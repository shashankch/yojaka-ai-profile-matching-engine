import os
import sys
import asyncio
import threading
import logging
import atexit
from typing import Dict, Any, Optional

from mcp import ClientSession, StdioServerParameters
from agentic_profile_matching import config
from mcp.client.stdio import stdio_client

logger = logging.getLogger("mcp_client")
logger.setLevel(logging.INFO)

# Ensure logs go to stderr to prevent stdout contamination if any prints occur
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


class MCPClientManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(MCPClientManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None

        # Track connections per server name
        self.servers_config: Dict[str, StdioServerParameters] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.contexts: Dict[str, Any] = {}

        # Configure local python environment command and paths from config
        fs_server_path = config.FILESYSTEM_SERVER_PATH
        search_server_path = config.SEARCH_SERVER_PATH
        python_exe = sys.executable or "python"
        env_vars = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.servers_config["filesystem"] = StdioServerParameters(
            command=python_exe, args=["-u", str(fs_server_path)], env=env_vars
        )
        self.servers_config["search"] = StdioServerParameters(
            command=python_exe, args=["-u", str(search_server_path)], env=env_vars
        )

        self._initialized = True

        # Register atexit handler to ensure subprocesses are always cleaned up
        atexit.register(self.stop)

    def start(self):
        """Starts the background event loop thread and initializes all server sessions."""
        with self._lock:
            if self.thread and self.thread.is_alive():
                return

            logger.info("Initializing MCP background event loop thread...")
            self.loop = asyncio.new_event_loop()

            # Start event loop in background daemon thread
            self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.thread.start()

            # Connect to servers
            future = asyncio.run_coroutine_threadsafe(self._connect_all(), self.loop)
            try:
                # Wait up to 10 seconds for initial connection
                future.result(timeout=10.0)
                logger.info("Successfully connected to all configured MCP servers.")
            except Exception as e:
                logger.error(f"Failed to establish connection to MCP servers: {e}")
                self.stop()
                raise e

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _connect_all(self):
        """Establish connections to all registered MCP servers concurrently."""
        tasks = []
        for server_name, params in self.servers_config.items():
            tasks.append(self._connect_server(server_name, params))
        await asyncio.gather(*tasks)

    async def _connect_server(self, name: str, params: StdioServerParameters):
        """Connect to a single MCP server."""
        logger.info(f"Connecting to MCP server '{name}' via stdio...")
        try:
            # Create stdio client context
            ctx = stdio_client(params)
            self.contexts[name] = ctx

            # Enter stdio transport
            read, write = await ctx.__aenter__()

            # Create session
            session = ClientSession(read, write)
            self.sessions[name] = session

            # Enter session context and initialize connection
            await session.__aenter__()
            await session.initialize()
            logger.info(f"Initialized MCP server session for '{name}'")

        except Exception as e:
            logger.error(f"Error connecting to server '{name}': {e}")
            raise e

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronously call a tool on a specific MCP server.

        Args:
            server_name: The target server name ('filesystem' or 'search').
            tool_name: The name of the tool to invoke.
            arguments: The dictionary of arguments to pass.
        """
        self.start()  # Lazy-start the connection thread if not running

        if server_name not in self.sessions:
            raise ValueError(f"MCP server session '{server_name}' is not connected.")

        session = self.sessions[server_name]

        # Execute the async coroutine thread-safely in our background loop
        future = asyncio.run_coroutine_threadsafe(session.call_tool(tool_name, arguments), self.loop)
        try:
            # Wait for execution to finish
            mcp_result = future.result(timeout=config.MCP_TIMEOUT)

            # The result from fastmcp tool call has a .content attribute which is a list of content blocks
            # We want to extract the main content. FastMCP tool execution returns CallToolResult.
            # If the tool returned a dictionary directly, FastMCP wraps it in a TextContent block containing JSON string.
            if hasattr(mcp_result, "content") and mcp_result.content:
                import json
                import ast

                parsed_blocks = []
                for block in mcp_result.content:
                    if hasattr(block, "text"):
                        text_content = block.text
                        try:
                            parsed_blocks.append(json.loads(text_content))
                        except Exception:
                            try:
                                parsed_blocks.append(ast.literal_eval(text_content))
                            except Exception:
                                parsed_blocks.append(text_content)
                if len(parsed_blocks) > 1:
                    return parsed_blocks
                elif len(parsed_blocks) == 1:
                    return parsed_blocks[0]
            return {"success": True, "result": mcp_result}

        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}' on '{server_name}': {e}")
            return {"success": False, "error": str(e)}

    def read_resource(self, server_name: str, uri: str) -> str:
        """
        Synchronously read a resource content from an MCP server.

        Args:
            server_name: The target server name ('filesystem' or 'search').
            uri: The resource URI (e.g. resumes://resume_john_doe.pdf).
        """
        self.start()

        if server_name not in self.sessions:
            raise ValueError(f"MCP server session '{server_name}' is not connected.")

        session = self.sessions[server_name]

        future = asyncio.run_coroutine_threadsafe(session.read_resource(uri), self.loop)

        try:
            mcp_result = future.result(timeout=config.MCP_TIMEOUT)
            # mcp_result is ReadResourceResult. Its contents are in .contents
            if hasattr(mcp_result, "contents") and mcp_result.contents:
                first_content = mcp_result.contents[0]
                if hasattr(first_content, "text"):
                    return first_content.text
            return str(mcp_result)
        except Exception as e:
            logger.error(f"Error reading resource '{uri}' from '{server_name}': {e}")
            raise e

    def stop(self):
        """Terminates all sessions, shuts down subprocesses, and stops the background loop."""
        with self._lock:
            if not self.loop:
                return

            def _safe_log(level: str, msg: str):
                try:
                    import sys

                    if getattr(sys, "stderr", None) is None or getattr(sys.stderr, "closed", False):
                        return
                    if getattr(sys, "stdout", None) is None or getattr(sys.stdout, "closed", False):
                        return
                    for h in logger.handlers:
                        if hasattr(h, "stream") and getattr(h.stream, "closed", False):
                            return
                    if level == "info":
                        logger.info(msg)
                    else:
                        logger.warning(msg)
                except Exception:
                    pass

            _safe_log("info", "Cleaning up MCP sessions and shutting down background processes...")

            async def _close_all():
                for name, session in list(self.sessions.items()):
                    try:
                        _safe_log("info", f"Closing session for '{name}'...")
                        await session.__aexit__(None, None, None)
                    except (Exception, RuntimeError):
                        pass

                for name, ctx in list(self.contexts.items()):
                    try:
                        _safe_log("info", f"Exiting stdio transport context for '{name}'...")
                        await ctx.__aexit__(None, None, None)
                    except (Exception, RuntimeError):
                        pass

                self.sessions.clear()
                self.contexts.clear()

            # Schedule close coroutine
            if self.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(_close_all(), self.loop)
                try:
                    future.result(timeout=5.0)
                except Exception as e:
                    _safe_log("warning", f"Timeout or error while closing sessions: {e}")

                # Stop the background event loop
                _safe_log("info", "Stopping background asyncio event loop...")
                try:
                    self.loop.call_soon_threadsafe(self.loop.stop)
                except Exception:
                    pass
            elif not self.loop.is_closed():
                try:
                    self.loop.run_until_complete(_close_all())
                except Exception as e:
                    _safe_log("warning", f"Error closing sessions on stopped loop: {e}")
                finally:
                    try:
                        self.loop.close()
                    except Exception:
                        pass
            else:
                self.sessions.clear()
                self.contexts.clear()

            if self.thread:
                self.thread.join(timeout=3.0)

            self.loop = None
            self.thread = None
            _safe_log("info", "MCP client manager successfully stopped.")


# Singleton Instance
mcp_client = MCPClientManager()
