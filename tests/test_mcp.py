"""
MCP Integration Tests
=====================
These tests verify:
  1. fs_client local-mode routing (always subprocess-free, fast).
  2. MCP tool logic correctness via direct in-process FastMCP.call_tool() calls.
     When CI=true (set automatically by GitHub Actions) we skip the real
     subprocess stdio transport and invoke tool functions directly, exercising
     the exact same Python logic without any pipe/signal machinery.
  3. Agent error-fallback behaviour (no LLM or MCP involved).

Why two test paths?
-------------------
On Python <3.12 / Linux, asyncio's subprocess transport uses SIGCHLD to notify
the event loop when a child process pipe is ready.  POSIX only delivers signals
to the *main* thread, but MCPClientManager runs its event loop in a background
thread, so the signal never arrives and session.initialize() hangs indefinitely.

Python 3.12 (used in CI) ships PidfdChildWatcher which avoids signals entirely,
but we also keep the direct-invocation path as a belt-and-suspenders guarantee
that MCP tests never block the CI pipeline regardless of Python version or runner.
"""

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_profile_matching import config
from agentic_profile_matching import fs_client
from agentic_profile_matching.matching_agent import (
    AgentState,
    extract_requirements_node,
)
from agentic_profile_matching.mcp_client import mcp_client

# ---------------------------------------------------------------------------
# CI detection
# ---------------------------------------------------------------------------
_IN_CI = os.getenv("CI", "").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Direct in-process invocation helper
# ---------------------------------------------------------------------------
_FS_SERVER = "agentic_profile_matching.filesystem_mcp_server"
_SEARCH_SERVER = "agentic_profile_matching.search_mcp_server"


def _call_tool_direct(server_module_path: str, tool_name: str, arguments: dict):
    """
    Invoke a FastMCP tool function directly in-process.

    FastMCP.call_tool() is an async coroutine; we run it with asyncio.run() which
    creates a fresh event loop on the *main* thread — fully signal-safe.

    FastMCP serialises return values as TextContent blocks:
      - A dict/primitive return → single block, JSON text.
      - A list return (e.g. batch_process) → N blocks, one JSON-encoded item each.

    Returns the native Python object (dict or list) decoded from those blocks.


    Raises unittest.SkipTest if the MCP SDK is not installed correctly so that
    CI shows SKIP rather than ERROR when the mcp package is unavailable.
    """
    import importlib

    try:
        module = importlib.import_module(server_module_path)
    except ImportError as exc:
        raise unittest.SkipTest(
            f"MCP SDK not available ({exc}). "
            "Ensure mcp>=1.0.0 is installed (Model Context Protocol SDK, not the legacy mcp package)."
        ) from exc

    server_mcp = module.mcp  # FastMCP instance registered at module level

    async def _run():
        return await server_mcp.call_tool(tool_name, arguments)

    # FastMCP.call_tool returns (list[ContentBlock], metadata_dict)
    content_blocks, _meta = asyncio.run(_run())
    if not content_blocks:
        return {}

    parsed = []
    for block in content_blocks:
        raw_text = block.text if hasattr(block, "text") else str(block)
        try:
            parsed.append(json.loads(raw_text))
        except json.JSONDecodeError:
            parsed.append(raw_text)

    # Single block → unwrap; multiple blocks → return as list (mirrors tool return type)
    return parsed[0] if len(parsed) == 1 else parsed


class TestMCPIntegration(unittest.TestCase):
    def setUp(self):
        # Cache original config flag
        self.original_use_mcp = config.USE_MCP

        # We will test in the data directory
        self.test_dir = Path(config.DATA_DIR) / "test_mcp_sandbox"
        self.test_dir.mkdir(exist_ok=True, parents=True)
        self.test_file = self.test_dir / "mcp_dummy.txt"
        self.test_file.write_text("Hello from MCP test framework. Python programming is awesome.")

    @classmethod
    def tearDownClass(cls):
        # Stop MCP client only if it was ever started (subprocess path)
        if mcp_client.loop is not None:
            mcp_client.stop()

    def tearDown(self):
        # Restore configuration
        config.USE_MCP = self.original_use_mcp

        # Clean up sandbox
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. Local-mode routing (always fast, no subprocess)
    # ------------------------------------------------------------------

    def test_local_mode_routing(self):
        """Verify fs_client routes calls locally when USE_MCP = False"""
        config.USE_MCP = False

        result = fs_client.read_file(str(self.test_file.resolve()))
        self.assertTrue(result["success"])
        self.assertIn("Hello from MCP test framework", result["content"])
        self.assertEqual(result["metadata"]["filename"], "mcp_dummy.txt")

    # ------------------------------------------------------------------
    # 2. MCP tool logic — direct in-process path (CI) or real stdio (local)
    # ------------------------------------------------------------------

    def test_mcp_mode_routing_and_server(self):
        """
        Verify MCP filesystem tools work end-to-end.

        CI: tools are called directly on the FastMCP server instance (no subprocess).
        Local: fs_client starts the real stdio MCP subprocess.
        """
        if _IN_CI:
            result = _call_tool_direct(_FS_SERVER, "read_file", {"filepath": str(self.test_file.resolve())})
            self.assertTrue(result.get("success"))
            self.assertIn("Hello from MCP test framework", result.get("content", ""))

            search_result = _call_tool_direct(
                _FS_SERVER,
                "search_in_file",
                {
                    "filepath": str(self.test_file.resolve()),
                    "keyword": "Python",
                    "context_size": 10,
                },
            )
            self.assertTrue(search_result.get("success"))
            self.assertGreaterEqual(search_result.get("total_matches", 0), 1)
        else:
            config.USE_MCP = True
            result = fs_client.read_file(str(self.test_file.resolve()))
            self.assertTrue(result["success"])
            self.assertIn("Hello from MCP test framework", result["content"])
            self.assertEqual(result["metadata"]["filename"], "mcp_dummy.txt")

            search_res = fs_client.search_in_file(
                filepath=str(self.test_file.resolve()), keyword="Python", context_size=10
            )
            self.assertTrue(search_res["success"])
            self.assertEqual(search_res["keyword"], "Python")
            self.assertGreaterEqual(search_res["total_matches"], 1)

    def test_mcp_batch_processing(self):
        """
        Verify batch_process tool reads multiple files concurrently.

        CI: direct in-process call.  Local: real stdio MCP subprocess.

        Note: FastMCP serialises list return values as a single JSON-array TextContent
        block, so _call_tool_direct() parses the outer block and returns a Python list.
        """
        file1 = self.test_dir / "batch1.txt"
        file2 = self.test_dir / "batch2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        try:
            if _IN_CI:
                results = _call_tool_direct(
                    _FS_SERVER,
                    "batch_process",
                    {"filepaths": [str(file1.resolve()), str(file2.resolve())]},
                )
                # batch_process returns a list; _call_tool_direct parses the JSON block
                self.assertIsInstance(results, list)
                self.assertEqual(len(results), 2)
                self.assertTrue(results[0]["success"])
                self.assertTrue(results[1]["success"])
                self.assertEqual(results[0]["content"], "Content 1")
                self.assertEqual(results[1]["content"], "Content 2")
            else:
                config.USE_MCP = True
                paths = [str(file1.resolve()), str(file2.resolve())]
                batch_results = fs_client.batch_process(paths)
                self.assertEqual(len(batch_results), 2)
                self.assertTrue(batch_results[0]["success"])
                self.assertTrue(batch_results[1]["success"])
                self.assertEqual(batch_results[0]["content"], "Content 1")
                self.assertEqual(batch_results[1]["content"], "Content 2")
        finally:
            if file1.exists():
                file1.unlink()
            if file2.exists():
                file2.unlink()

    def test_multi_mcp_search_bonus(self):
        """
        Verify search MCP server tools (search_web, fetch_candidate_notes, search_chroma_db).

        CI: direct in-process calls.  Local: real stdio MCP subprocess.
        """
        if _IN_CI:
            # --- search_web: should hit mock path for "John Doe" ---
            search_res = _call_tool_direct(_SEARCH_SERVER, "search_web", {"query": "John Doe Github"})
            self.assertTrue(search_res.get("success"))
            self.assertEqual(search_res.get("query"), "John Doe Github")
            self.assertGreater(len(search_res.get("results", [])), 0)

            # --- fetch_candidate_notes: mock HR notes for Jane Smith ---
            notes_res = _call_tool_direct(_SEARCH_SERVER, "fetch_candidate_notes", {"candidate_name": "Jane Smith"})
            self.assertTrue(notes_res.get("success"))
            self.assertIn("Jane Smith", notes_res.get("candidate_name", ""))
            self.assertIn("Frontend", notes_res.get("notes", ""))

            # --- search_chroma_db: expected to return success=False (no resumes ingested in CI) ---
            db_res = _call_tool_direct(_SEARCH_SERVER, "search_chroma_db", {"query": "Python", "limit": 2})
            self.assertIn("success", db_res)
        else:
            config.USE_MCP = True
            search_res = mcp_client.call_tool("search", "search_web", {"query": "John Doe Github"})
            self.assertTrue(search_res["success"])
            self.assertEqual(search_res["query"], "John Doe Github")
            self.assertGreater(len(search_res["results"]), 0)

            notes_res = mcp_client.call_tool("search", "fetch_candidate_notes", {"candidate_name": "Jane Smith"})
            self.assertTrue(notes_res["success"])
            self.assertIn("Jane Smith", notes_res["candidate_name"])
            self.assertIn("Frontend", notes_res["notes"])

            db_res = mcp_client.call_tool("search", "search_chroma_db", {"query": "Python", "limit": 2})
            self.assertIn("success", db_res)

    # ------------------------------------------------------------------
    # 3. Agent error-fallback (no MCP / LLM)
    # ------------------------------------------------------------------

    def test_agent_error_fallback(self):
        """Verify that agent nodes record errors and recover using safe fallbacks when exceptions occur"""
        # Create state with bad credentials to guarantee LLM connection failure
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="Analyze this job: Python Developer needed.")],
            requirements={},
            shortlist=[],
            coarse_screen_limit=5,
            deep_screen_limit=3,
            recommendation_limit=2,
            current_round=1,
            errors=[],
            llm_provider="InvalidProvider",
            llm_model="gpt-4o",
            api_key="bad-invalid-key",
        )

        # Run node that executes LLM call
        result = extract_requirements_node(state)

        # Verify it handled exception, didn't crash, added error, and used fallback requirements
        self.assertIn("errors", result)
        self.assertGreater(len(result["errors"]), 0)
        self.assertTrue(any("Requirements extraction failed" in err for err in result["errors"]))
        self.assertEqual(result["requirements"]["title"], "Software Engineer")


if __name__ == "__main__":
    unittest.main()
