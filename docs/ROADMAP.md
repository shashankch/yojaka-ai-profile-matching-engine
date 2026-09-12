# Yojaka AI (Agentic Profile Matching Engine): Project Roadmap

This document outlines the strategic milestones for **Yojaka AI (Agentic Profile Matching Engine)**. 

> 💡 **Implementation Note**: Architectural decisions, technical task breakdowns, and design patterns are formally documented in individual [Architecture Decision Records (ADRs)](adr/README.md) and [System Architecture](architecture.md).

---

## 📍 Implementation Milestones

### Phase 1: Environment & Foundational Setup ✅
- Core package environment initialization with LangGraph, Streamlit, Groq, and Google GenAI.
- Secure environment configuration via `.env` credentials.

### Phase 2: Ingestion & Vector Indexing Subsystem ✅
- Multi-format synthetic candidate dataset generation (PDF, DOCX, TXT).
- Document ingestion pipeline with local SentenceTransformers embeddings and ChromaDB vector indexing.

### Phase 3: LangGraph Agent & Conversational State Machine ✅
- 9-node `StateGraph` workflow with deterministic state transitions and `MemorySaver` checkpointing.
- End-to-end recruiter loop (parsing, search, deep screening, report generation, conversational adjustment).

### Phase 4: Assessment, Comparison & Screening Tools ✅
- Schema-validated requirement extraction and multi-profile side-by-side comparison matrices.
- Fact-grounded candidate interview question generation tailored to identified skill gaps.

### Phase 5: Streamlit Recruiter Dashboard ✅
- Interactive dual-pane recruiter UI (`app.py`) for live conversation and shortlist inspection.
- Conversational refinement workflow testing against recruitment scenarios.

### Phase 6: Model Context Protocol (MCP) Integration ✅
- FastMCP filesystem (`resumes://`) and candidate search stdio protocol servers.
- Thread-safe dual-mode gateway client (`USE_MCP=True/False`) supporting local in-process and JSON-RPC execution ([ADR-001](adr/ADR-001-mcp-dual-mode-gateway-architecture.md)).

### Phase 7: Modular Graph Decoupling & Linting Hygiene ✅
- Decomposed monolithic agent into a decoupled `agent/` package (`state.py`, `prompts.py`, `nodes.py`, `routers.py`).
- Automated pre-commit hooks and GitHub Actions CI quality gates with Ruff.

### Phase 8: Enterprise Abstractions & Background Workers ✅
- **Protocol Separation**: Dedicated `IngestionService` isolating business logic from MCP transport.
- **Storage Abstraction**: `BaseVectorStore` protocol with ChromaDB default and Qdrant stub ([ADR-003](adr/ADR-003-basevectorstore-structural-protocol.md), [ADR-005](adr/ADR-005-idempotent-upsert-ingestion.md)).
- **Retrieval Optimization**: Cached BM25Okapi sparse matrix with corpus fingerprinting ([ADR-004](adr/ADR-004-bm25-corpus-index-caching.md)).
- **State & Schema Safety**: TypedDict `AgentState` with Pydantic V2 LLM output contracts ([ADR-002](adr/ADR-002-using-typeddict-for-agentstate.md)).
- **Distributed Worker Queue**: Celery + Redis task queue with Docker Compose orchestration ([ADR-006](adr/ADR-006-celery-redis-task-queue.md)).

### Phase 9: Observability, Evaluation & Layout Ingestion ✅
- **Structured Observability**: Structured JSON logging and `@trace_node` latency instrumentation with opt-in Langfuse and OpenTelemetry backends ([ADR-007](adr/ADR-007-structured-json-logging-and-tracing.md)).
- **RAG Evaluation Suite**: Automated Recall@K, MRR, and LLM faithfulness benchmarks against ground-truth datasets.
- **Advanced Document Ingestion**: PyMuPDF (`pymupdf`) vertical column-sorted extraction with opt-in Unstructured.io support.

### Phase 10: Semantic Routing, Multi-Provider Scale & Architecture Baseline ✅
- **Multi-Factor Hybrid Scoring**: Min-max normalized 60/40 dense-sparse candidate ranking with grounded guardrails ([ADR-008](adr/ADR-008-multi-factor-hybrid-scoring-and-hierarchy.md)).
- **Tiered Intent Routing**: <2ms local semantic vector routing with LLM structured classification fallback ([ADR-009](adr/ADR-009-tiered-semantic-embedding-intent-routing.md)).
- **Multi-Provider & Indic LLMs**: Unified provider factory with Sarvam AI (`sarvam-105b`), Groq, Gemini, and OpenAI ([ADR-010](adr/ADR-010-multi-provider-sarvam-indic-llm.md)).
- **Architectural Documentation**: Published formal Architecture Decision Records catalog ([ADRs 001–016](adr/README.md)).

### Phase 11: Production Hardening, Security & State Invariance ✅
- **11.1 — Stateless Credential Isolation**: Purge API credentials from `AgentState`; inject via `RunnableConfig` (CWE-312 compliance) ([ADR-011](adr/ADR-011-stateless-credential-isolation.md)).
- **11.2 — Functional State Immutability**: Enforce copy-on-write candidate dict updates for safe LangGraph retries ([ADR-012](adr/ADR-012-functional-state-immutability.md)).
- **11.3 — Dynamic Skills Taxonomy**: Declarative taxonomy with canonical alias stemming (superseded by Generative Dynamic Expansion in Phase 15) ([ADR-013](adr/ADR-013-dynamic-skills-taxonomy-alias-normalization.md)).
- **11.4 — Floating-Point Precision**: Migrate `CandidateMatch.score` typing from `int` to `float` across state schemas.

### Phase 12: High-Throughput Screening & Resource Lifecycle ✅
- **12.1 — Async Candidate Deep Screening**: Concurrency-controlled worker pool (`ThreadPoolExecutor` + `Semaphore(2)`) reducing screening wall-clock time from ~75s to ~15s ([ADR-014](adr/ADR-014-concurrency-controlled-async-screening.md)).
- **12.2 — Singleton Model Caching**: Cache `JobMatcher` and embedder instances via `@st.cache_resource` and container DI.
- **12.3 — Open/Closed Provider Registry**: Extensible `PROVIDER_REGISTRY` mapping with runtime `register_provider()` API ([ADR-015](adr/ADR-015-open-closed-llm-provider-registry.md)).
- **12.4 — Modular UI Decomposition**: Refactor `app.py` into testable `ui/components/` and `ui/session_manager.py`.

### Phase 13: Enterprise CI/CD, Supply Chain Security & Quality Gates ✅
- **13.1 — Static Type Analysis**: Strict `mypy` type checking in CI.
- **13.2 — Test Coverage Enforcement**: Automated `pytest-cov` gate enforcing $\ge 75\%$ code coverage.
- **13.3 — Security Scanning**: Automated secret detection (`gitleaks`) and CVE vulnerability auditing (`pip-audit`).

### Phase 14: Engine Resilience, Cache Correctness & Retrieval Precision ✅
- Deterministic BM25 fingerprinting and global corpus hash validation.
- Guardrail boundary normalization and async subprocess lifecycle management.

### Phase 15: Modernization, Bug Remediation & In-Memory Resume Ingestion ✅
- **15.1 — Cold-Start Auto-Bootstrap**: Automatic vector store bootstrapping on cold starts for Streamlit Cloud deployment (`config.RESUMES_DIR` and `data/resumes/` auto-indexing).
- **15.2 — Zero-Disk In-Memory Resume Ingestion & Ephemeral PII Isolation**: Stream-based in-memory processing (`io.BytesIO`) without server disk writes; layered via `CompositeVectorStore` with session-scoped in-memory collections (`ChromaVectorStore(ephemeral=True)`), prior chunk pruning on update, anti-XSS HTML escaping, and 10MB upload guardrails ([ADR-016](adr/ADR-016-zero-disk-in-memory-resume-ingestion.md)).
- **15.3 — Dynamic Semantic Skill Expansion & Word-Boundary Matching**: Generative LLM requirement expansion replacing static taxonomy YAML dictionaries with case-insensitive synonym lookup and token/word-boundary matching preventing false-positive overlaps (e.g. Java vs JavaScript) ([ADR-013](adr/ADR-013-dynamic-skills-taxonomy-alias-normalization.md)).
- **15.4 — Modern PyMuPDF Transport Hygiene**: Upgraded to `import pymupdf` with single-pass page text extraction, eliminating duplicate parser passes and stdout deprecation warnings on MCP stdio JSON-RPC framing.
- **15.5 — Concurrency-Controlled Screening & Coordinated Rate Limiting**: Parallel `ThreadPoolExecutor` + `Semaphore(2)` screening with coordinated `THROTTLE_DELAY` inter-call spacing, robust unhandled exception fallback fields, and copy-on-write candidate dict updates ([ADR-012](adr/ADR-012-functional-state-immutability.md), [ADR-014](adr/ADR-014-concurrency-controlled-async-screening.md)).
- **15.6 — Progressive Checkpoint Indicators & Non-Duplicative Error Fallbacks**: Live progressive `st.status` node execution checkpoints with safe partial-progress error handling preventing duplicate re-invocations.
- **15.7 — Stateless Credential Isolation (CWE-312)**: Isolated credentials within `RunnableConfig` eliminating API secret leakage into checkpoints, with provider-specific environment key mapping (`GROQ_API_KEY`, `GEMINI_API_KEY`, `SARVAM_API_KEY`, `OPENAI_API_KEY`) ([ADR-011](adr/ADR-011-stateless-credential-isolation.md)).
- **15.8 — Brand Modernization**: Rebranded to **Yojaka AI (Agentic Profile Matching Engine)**.
- **15.10 — LLM-Driven Intent Routing, Dynamic Anchors & Semantic Caching**: Re-architected `route_input` in `agent/routers.py` into a modern 2026 LLM-driven router (`_classify_via_llm` as primary authority), completely eliminating static hardcoded anchor dictionaries and keyword arrays; added dynamic LLM anchor synthesis (`generate_dynamic_intent_anchors`), in-memory LRU query routing cache (`_ROUTING_CACHE`) for $0\text{ms}$ repeated queries, case-insensitive provider resolution in `config.py`, and direct in-process tool fallbacks in `nodes.py` ([ADR-009](adr/ADR-009-tiered-semantic-embedding-intent-routing.md)).

---

## 🚀 Future Backlog

- **Ephemeral S3 Signed URL Staging**: Pre-signed S3 bucket staging with automated lifecycle expiration for high-volume enterprise uploads.
- **Edge WASM Client-Side Vector Ingestion**: In-browser client-side document parsing and embedding vector generation via WebAssembly sandbox.
- **Reversible PII Anonymization & Redaction Vault**: Tokenized privacy vault masking personal identifiers (`[CANDIDATE_A]`) with deanonymization keys.
- **Indirect Prompt Injection Sanitizer**: Pre-screening heuristic and LLM scanner filtering adversarial prompt injection attempts in resume content.
- **Semantic Embedding Cache**: Redis-backed semantic vector query cache ($< 10\text{ms}$).
- **Multi-Agent Consensus Loop**: Independent Technical Architect and HR Sourcing screener debate before recommendation.
- **Bias & Fairness Auditing**: Automated inclusivity auditing for job descriptions and screening assessments.

---

<!-- References -->
[langgraph]: https://langchain-ai.github.io/langgraph/
[streamlit]: https://streamlit.io/
[ChromaDB]: https://www.trychroma.com/
[FastMCP]: https://github.com/modelcontextprotocol/python-sdk
[Ruff]: https://github.com/astral-sh/ruff
[Qdrant]: https://qdrant.tech/
[Celery]: https://docs.celeryq.dev/
[Redis]: https://redis.io/
[Langfuse]: https://langfuse.com/
[OpenTelemetry]: https://opentelemetry.io/
[PyMuPDF]: https://github.com/pymupdf/PyMuPDF
[Sarvam AI]: https://www.sarvam.ai/
