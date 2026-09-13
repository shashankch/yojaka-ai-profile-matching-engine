# Yojaka AI (Agentic Profile Matching Engine): Engineering Conventions

This document defines the architectural guidelines, code quality standards, and design conventions for **Yojaka AI (Agentic Profile Matching Engine)**. These standards reflect production-grade software development expectations.

---

## 🏛️ 1. Architectural Principles & Layering

1. **Separation of Concerns (SoC)**
   - **Domain Logic**: RAG processing (`resume_rag.py`), hybrid matching algorithm (`job_matcher.py`), and graph state machine (`agent/`) must remain clean and decoupled from presentation layers (Streamlit UI) and RPC transport protocols (MCP servers).
   - **Service Layer**: Business operations spanning multiple domain components should be encapsulated in a dedicated service layer (e.g., `IngestionService`).

2. **Protocol-Driven Design & Abstractions**
   - High-level orchestrators must depend on interfaces (`typing.Protocol` or `abc.ABC`), not concrete implementations (e.g., `BaseVectorStore` instead of direct `ChromaDB` calls).
   - Concrete implementations (e.g., `ChromaVectorStore`, `QdrantVectorStore`) must be swappable via dependency injection without changing consumer code.

3. **Dependency Injection**
   - Classes must receive external dependencies (vector store, clients, LLM providers) via constructor arguments rather than instantiating them internally.

---

## 🛡️ 2. State Management & Type Safety

1. **Explicit Type Hinting**
   - All function parameters, return values, class attributes, and module constants must carry explicit Python 3.10+ type annotations.
   - Avoid `Any` wherever possible; use `Union`, `Optional`, `TypeVar`, or generic collections (`list[str]`, `dict[str, float]`).

2. **Structured Models with Pydantic V2**
   - Use Pydantic V2 `BaseModel` for config validation, external API payloads, and LLM output parsing.
   - LLM responses must be parsed and validated using `model_validate_json()` with fallback handling.

3. **Immutable Agent State**
   - Graph state schemas (e.g., `AgentState` in `agent/state.py`) must use `TypedDict` or Pydantic models.
   - State updates within graph nodes should return partial state dictionaries explicitly rather than mutating global/external references.

---

## ⚠️ 3. Resilience & Error Handling Boundaries

1. **No Silent Exception Swallowing**
   - Catch specific exceptions (`FileNotFoundError`, `JSONDecodeError`, `KeyError`, `httpx.HTTPError`).
   - Never use empty `except:` or `except Exception: pass` without logging or populating state warning flags.

2. **Graceful State Degradation**
   - If a non-critical tool node (e.g., Tavily API web search or external candidate notes fetching) fails, log a warning, set a warning flag in `AgentState`, and allow the core pipeline to complete.

3. **Deterministic LLM Fallbacks**
   - Any LLM invocation expecting JSON must include retry or default template fallback logic in case of schema validation failures or API rate limits.

---

## ⚡ 4. Concurrency & Protocol Integration

1. **Async / Sync Bridge Safety**
   - MCP servers operate over async stdio streams. When called from synchronous environments (like Streamlit UI or legacy tools), bridges must use thread-safe event loop execution patterns (e.g., `mcp_client.py` background loop runner).

2. **Resource Lifecycle Teardown**
   - Connections, subprocesses, vector store handles, and open file handles must be managed via context managers (`with` / `async with`) or explicit `.close()` / `.stop()` teardown hooks.

---

## 🧪 5. Testing & Code Quality Standards

1. **Test Coverage Expectations**
   - **Core Logic**: Math algorithms (e.g. 60/40 hybrid BM25 + vector ranking), graph router conditionals, and candidate assessment tools must have unit tests.
   - **Protocol Tests**: Mock external LLMs and MCP transport in unit tests (`pytest`).

2. **Linter & Formatter Integrity**
   - All code must pass `ruff check .` and `ruff format .` with zero errors or warnings before merging.
   - Maintain maximum line length limit of **100 characters**.

---

## 📝 6. Commit & Versioning Standards

1. **Conventional Commits**
   - Format: `<type>(<scope>): <short summary>`
   - Allowed Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`.
   - Example: `feat(observability): Phase 9.1 - Structured JSON Logging & Node Tracing (v0.9.1)`

2. **Semantic Versioning & Incremental Changelogs**
   - Follow `MAJOR.MINOR.PATCH` versioning tracked in `pyproject.toml` and documented chronologically in `CHANGELOG.md`.
   - Every subphase PR MUST increment the version in `pyproject.toml` and `CHANGELOG.md` with an explicit version section (e.g. `[0.8.2]`, `[0.8.3]`, `[0.9.1]`) and release date.

---

## 📊 7. Observability & Logging Standards

1. **Structured JSON Logging**
   - Use `get_logger(name)` from `agentic_profile_matching.observability`.
   - Unformatted bare `print()` calls in production modules (`nodes.py`, `job_matcher.py`, `resume_rag.py`, `services/`) are strictly prohibited.
   - All log records are output as single-line JSON objects with standard (`timestamp`, `level`, `logger`, `message`) and trace payload attributes (`event`, `node`, `elapsed_ms`).

2. **Node Latency Instrumentation**
   - All 9 LangGraph agent workflow nodes (`parse_input`, `extract_requirements`, `search_resumes`, `rank_candidates`, `deep_screen`, `recommendation`, `generate_report`, `adjust_requirements`, `conversational_query`) MUST be decorated with `@trace_node(node_name)`.

---

## 📚 8. Pre-PR Documentation Audit Protocol

Before submitting a Pull Request for ANY subphase:
1. **Quality Gate**: `ruff check src/ tests/`, `ruff format --check src/ tests/`, and `pytest tests/ -v` MUST pass with zero errors.
2. **CHANGELOG Sync**: `CHANGELOG.md` MUST record all added, changed, fixed, or security items under a new version tag (`[X.Y.Z] - YYYY-MM-DD`).
3. **README Sync**: `README.md` features overview, project directory tree, and setup commands MUST be updated to reflect all newly added modules, tools, and test suites.

---

## 🤖 9. Intent Routing & Structured LLM Standards

1. **LLM-Driven Intent Routing & Dynamic Anchors (ADR-009)**
   - Hardcoded substring / keyword matching arrays and static phrase lists for routing decisions are strictly prohibited.
   - Graph routing MUST use an LLM-driven architecture with dynamic anchor synthesis and query caching:
     - **Structural Fast Path & In-Memory LRU Cache (0ms)**: Multi-line raw JD pastes and repeated MD5-hashed queries resolve in sub-millisecond time.
     - **Primary Tier**: LLM structured intent classification (`with_structured_output(RouteDecision)`) with active session context.
     - **Secondary Tier (Fallback)**: Dynamic semantic embedding router evaluating cosine similarity against dynamic intent prototypes synthesized by the LLM (`generate_dynamic_intent_anchors`) with calibrated thresholds (0.20).

2. **Schema-Enforced Tool Outputs (ADR-010)**
   - All tool functions expecting structured outputs from LLMs MUST use `invoke_structured()` with explicit Pydantic V2 schemas.
   - String responses MUST be protected with balanced-bracket extraction and multi-tier auto-repair.

---

## 🔒 10. State Immutability & LangGraph Checkpoint Invariance (ADR-012)

1. **Functional State Updates Only**
   - Direct in-place mutation of dictionaries in `AgentState` is strictly prohibited.
   - Nodes must construct and return new dictionaries (`{**c, ...}`) when updating candidate assessment metadata.
   - All node functions must remain pure and idempotent, guaranteeing safe retry execution from any `MemorySaver` checkpoint.

2. **Accurate Numeric Typing**
   - Numeric scores must be typed as `float` across all state schemas to prevent precision truncation during hybrid ranking and sorting.

---

## 🛡️ 11. Credential Security & Stateless Execution Standards (ADR-011)

1. **Purge Credentials from Serializable State**
   - API keys, access tokens, and base URLs must NEVER be stored in `AgentState` or serialized in graph memory.
   - LLM instances and credential handles must be injected exclusively at runtime via `RunnableConfig` (`configurable["llm"]`) or an in-memory `CredentialStore`.

2. **Structured Log Sanitization**
   - Log formatters must apply automated redaction filters masking API keys, authorization headers, and bearer tokens from all log outputs.

---

## ⚡ 12. Concurrency & Resource Lifecycle Standards (ADR-014, ADR-015)

1. **Bounded Concurrency with Rate Limiting**
   - Parallel LLM screening loops must be constrained using bounded worker pools (`ThreadPoolExecutor`) and rate-limiting `Semaphore` locks to prevent HTTP 429 quota exhaustion.
   - Real-time progress events must be emitted to keep UI consumers responsive.

2. **Singleton Model Lifecycle & Dependency Injection**
   - Embedding models (`SentenceTransformer`) and search orchestrators (`JobMatcher`) must be managed as cached singletons (e.g. `@st.cache_resource` / DI container) to eliminate redundant model instantiation.

3. **Open/Closed Provider Registry**
   - LLM factories must use a modular `PROVIDER_REGISTRY` mapping with `register_provider()` extension hooks, eliminating procedural `if/elif` chains.

---

## 🧪 13. CI/CD Quality Gates & Static Type Enforcement

1. **Static Type Checking**
   - All code must pass `mypy src/ --strict-optional` before merging.

2. **Test Coverage Thresholds**
   - Pull requests must maintain a minimum automated test coverage of **75%** enforced via `pytest-cov --cov-fail-under=75`.

3. **Supply Chain & Secret Auditing**
   - CI workflows must include automated secret scanning (`gitleaks`) and dependency vulnerability auditing (`pip-audit`).

---

## 📂 14. Zero-Disk Ingestion & Generative Skill Expansion Standards (ADR-013, ADR-016)

1. **Zero-Disk In-Memory Stream Processing & Ephemeral Multi-Tenant Isolation (ADR-016)**
   - Uploaded resumes must be processed directly from in-memory byte buffers (`io.BytesIO`) using stream-compatible document parsers (`pymupdf.open(stream=stream_bytes, filetype="pdf")`, `docx.Document(io.BytesIO(stream_bytes))`).
   - Persisting unencrypted candidate resumes or uploaded PII to ephemeral local disk or shared persistent database collections is strictly prohibited.
   - The application uses `CompositeVectorStore`, isolating uploads into session-scoped in-memory collections (`uploads_{session_id}`) via `chromadb.EphemeralClient()`.
   - Chunks must be vectorized and upserted directly into the active session vector store with provenance URI formatted as `stream://{filename}?hash={content_hash}`.
   - Uploaded files must enforce a strict 10MB size bound before reading bytes into memory.

2. **Generative LLM Skill Expansion over Static Taxonomies (ADR-013)**
   - Do NOT maintain brittle manual static taxonomy YAML dictionaries for domain skill mapping.
   - Requirement extraction must generate dynamic structured skill expansions (`Dict[str, List[str]]`) at runtime.
   - `JobMatcher._skill_matches_candidate()` must evaluate semantic equivalence so specialized candidate technologies (e.g. AWS, GCP) satisfy parent requirements (e.g. Cloud) using word-boundary token matching to avoid false positives (e.g. Java vs JavaScript).

3. **Anti-XSS Candidate Data Sanitization (OWASP Standard)**
   - All candidate-derived tokens, names, education, experience, and paths interpolated into HTML UI templates (such as candidate cards in `app.py`) MUST be strictly sanitized using `html.escape(..., quote=True)` to neutralize script injection and markup tampering.



