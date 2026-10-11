# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-10-10

### Added
- **HyDE Query Synthesis**: Hypothetical Document Embeddings service (`HyDEService`) synthesizing candidate profiles before dense vector retrieval to eliminate query-resume terminology mismatch ([ADR-019](adr/ADR-019-advanced-rag-hyde-parent-doc-and-local-inference.md)).
- **Hierarchical Parent-Document Chunking**: Implemented `ParentDocumentService` and thread-safe in-memory `ParentDocumentStore`, linking 150–250 token child chunks in ChromaDB to 1,000–1,500 character parent section blocks to eliminate context fragmentation during dual-rubric screening.
- **Air-Gapped Local Inference Engine**: Turnkey local inference service (`LocalInferenceService`) supporting Ollama and vLLM daemons via OpenAI-compatible endpoints with automated health probing for zero-cloud sovereign privacy deployments.
- **Pre-Retrieval Faceted Metadata Constraints**: Implemented `FacetedFilter` pruning candidates by experience years, education levels, and must-have skills with dynamic generative taxonomy expansions before hybrid scoring.
- **Architecture Diagram as Code**: Published publication-grade architecture diagram (`hyde_parent_doc_rag.png`) generated programmatically via Mingrammer `diagrams`.
- **ADR-019**: Published formal architecture decision record for advanced RAG architecture, parent-document chunking, and air-gapped local inference.
- **Test Suite Expansion**: Added 13 new unit and integration tests across HyDE, parent-document chunking, faceted filtering, and local inference daemons (140 tests total, 100% green pass rate).

### Fixed
- **Security Finding 6 (Stateless Credential Hygiene)**: Decoupled API keys from `AgentState` in node model factories, logging deprecation warnings and prioritizing runtime `RunnableConfig["configurable"]` credentials.
- **Heading Anchor Rendering**: Fixed attribute list heading syntax in `docs/CONVENTIONS.md` to render cleanly in standard markdown viewports.

## [1.5.0] - 2026-10-09

### Added
- **Calibrated Margin-Based Intent Routing**: Sub-2ms direct routing for high-confidence queries ($\Delta \ge 0.12$) at $0 token cost, with structured SLM escalation on ambiguity ([ADR-018](adr/ADR-018-calibrated-margin-routing-and-subgraphs.md)).
- **Native LangGraph 1.x Command Traversal**: Migrated workflow nodes to native `Command(goto=target, update={...})` primitives with destination routing, eliminating conditional edge boilerplate.
- **`RoutingCommand` Backward Compatibility**: Subclassed `Command` with mapping interfaces to maintain 100% test fixture compatibility.
- **Parallel Dual-Rubric Structured Evaluation**: Replaced sequential debate loops with concurrent structured scoring (Technical Architecture 60% + Domain Fit 40%) aggregated deterministically in ~1.2s.
- **Modular Typed Subgraphs**: Decomposed pipeline into isolated subgraphs (`JDAnalyzerSubgraph`, `TalentRetrievalSubgraph`, `DeepScreeningSubgraph`, `SynthesisSubgraph`) in `agent/subgraphs.py`.
- **ADR-018**: Published formal record on calibrated margin routing, native commands, and dual-rubric subgraphs.
- **Test Suite**: Expanded automated test suite to 127 tests (+11 tests covering margin gates, subgraphs, and dual-rubric evaluation).

## [1.4.0] - 2026-09-29

### Added
- **Layout-Aware Section Document Parsing**: PyMuPDF visual bounding block parsing for PDF, paragraph hierarchy for DOCX, and regex boundary analysis for TXT resumes, normalizing sections without mid-sentence truncation.
- **Anthropic Contextual Retrieval Prepending**: Prepends 50–80 word metadata banners to chunks before embedding and BM25 indexing while preserving clean candidate text for UI rendering.
- **Two-Stage Cross-Encoder Reranking**: Integrated `cross-encoder/ms-marco-MiniLM-L-6-v2` with Sigmoid calibration and Reciprocal Rank Fusion (RRF) for joint query-candidate attention ([ADR-017](adr/ADR-017-layout-parsing-contextual-retrieval-reranking.md)).
- **In-Memory Store Cloud Fallback**: Built pure NumPy `InMemoryVectorStore` fallback conforming to `BaseVectorStore` protocol for serverless/ephemeral container runtimes.
- **License Migration**: Updated project license to Apache License 2.0.

## [1.3.0] - 2026-09-28

### Added
- **Modular Presentation Layer**: Decomposed monolithic `app.py` into a `<75-LOC` conductor and testable `src/agentic_profile_matching/ui/` components.
- **Typewriter Response Streaming**: Integrated `st.write_stream()` token generator for fluid real-time chat output and dynamic execution status tracking.
- **Headless FastAPI Gateway Sidecar**: Added decoupled ASGI sidecar (`api/`) exposing REST endpoints (`/jobs/extract`, `/candidates/match`) and Server-Sent Events (SSE) streaming.
- **API & UI Test Suites**: Added unit and integration tests for UI components and FastAPI routes.

## [1.2.1] - 2026-09-13

### Changed
- **Zero-Scroll Sidebar**: Compacted profile statistics and layout margins to ensure constraints editor is fully visible above the fold.
- **Batched Vector Ingestion**: Accelerated cold-start indexing by ~10x using batched encoding and upserts.
- **Path Resolution**: Hardened path discovery across `config.py` for automated directory detection on cold boots.
- **Web Search Integration**: Added optional Tavily API key configuration with automatic DuckDuckGo fallback.

## [1.2.0] - 2026-09-12

### Added
- **Zero-Disk In-Memory Resume Ingestion**: Implemented stream-based parsing (`io.BytesIO`) without writing candidate files to disk ([ADR-016](adr/ADR-016-zero-disk-in-memory-resume-ingestion.md)).
- **Session-Scoped Ephemeral PII Isolation**: Integrated `CompositeVectorStore` layering persistent baseline vectors with session-isolated in-memory collections (`chromadb.EphemeralClient`).
- **Dynamic Generative Skill Expansion**: LLM-driven skill synonym generation replacing static taxonomies, with exact token word-boundary matching ([ADR-013](adr/ADR-013-dynamic-skills-taxonomy-alias-normalization.md)).
- **LLM-Driven Intent Routing & Caching**: Structured zero-shot intent classifier with dynamic anchor synthesis and sub-millisecond query cache ([ADR-009](adr/ADR-009-tiered-semantic-embedding-intent-routing.md)).
- **Anti-XSS Output Sanitization**: Applied `html.escape()` to candidate fields rendered in UI templates.

### Security
- **Stateless Credential Isolation**: Removed API keys and secrets from serialized `AgentState`, injecting them strictly via `RunnableConfig` ([ADR-011](adr/ADR-011-stateless-credential-isolation.md)).
- **Copy-on-Write State Immutability**: Enforced functional dictionary updates in graph nodes to prevent state mutation during retries ([ADR-012](adr/ADR-012-functional-state-immutability.md)).

## [1.1.0] - 2026-08-30

### Added
- **Multi-Provider LLM Gateway**: Added unified model provider registry with Sarvam AI (`sarvam-105b`), Groq, Gemini, and OpenAI support ([ADR-010](adr/ADR-010-multi-provider-sarvam-indic-llm.md)).
- **Schema-Enforced Outputs**: Added Pydantic V2 structured outputs and multi-tier JSON auto-repair for LLM tool invocations.

## [1.0.0] - 2026-08-12

### Added
- **Multi-Factor Hybrid Candidate Scoring**: Min-max normalized dense cosine similarity, BM25Okapi lexical retrieval, and qualification satisfaction ratio ([ADR-008](adr/ADR-008-multi-factor-hybrid-scoring-and-hierarchy.md)).
- **Grounded Recommendation Hierarchy**: Deterministic candidate screening status hierarchy enforcing mandatory safety guardrails.
- **ADR Catalog**: Published formal Architecture Decision Records ADR-001 through ADR-008.

## [0.9.0] - 2026-08-11

### Added
- **Enterprise Observability & Tracing**: Structured single-line JSON logging and `@trace_node` latency instrumentation supporting Langfuse and OpenTelemetry ([ADR-007](adr/ADR-007-structured-json-logging-and-tracing.md)).
- **RAG Evaluation Benchmarks**: Ground-truth benchmark suite evaluating Retrieval Recall@K, Mean Reciprocal Rank (MRR), and response faithfulness.

## [0.8.0] - 2026-08-06

### Added
- **Pluggable Vector Store Protocol**: Introduced `BaseVectorStore` structural protocol with ChromaDB and Qdrant implementations ([ADR-003](adr/ADR-003-basevectorstore-structural-protocol.md)).
- **BM25 Corpus Caching**: Sparse matrix index caching with MD5 fingerprint validation ([ADR-004](adr/ADR-004-bm25-corpus-index-caching.md)).
- **Idempotent Ingestion**: Section-scoped deterministic chunk keys preventing collection bloat ([ADR-005](adr/ADR-005-idempotent-upsert-ingestion.md)).
- **Asynchronous Task Queue**: Celery workers with Redis 7 message broker and Docker Compose orchestration ([ADR-006](adr/ADR-006-celery-redis-task-queue.md)).
- **Ingestion Service**: Decoupled document processing from transport gateways.

## [0.7.0] - 2026-07-30

### Changed
- Refactored monolithic agent script into decoupled `agent/` package (`state.py`, `prompts.py`, `nodes.py`, `routers.py`).
- Integrated Ruff linter and formatter into pre-commit and CI workflows.

## [0.6.0] - 2026-07-29

### Added
- **Model Context Protocol (MCP) Dual Gateway**: FastMCP filesystem and search servers operating over `stdio` JSON-RPC 2.0 with local in-process toggle ([ADR-001](adr/ADR-001-mcp-dual-mode-gateway-architecture.md)).

## [0.5.0] - 2026-07-28

### Added
- Created engineering conventions (`CONVENTIONS.md`) and contributing guidelines (`CONTRIBUTING.md`).

## [0.4.0] - 2026-07-26

### Added
- Core candidate evaluation tools for requirement extraction, comparison matrix, and interview question generation.

## [0.3.0] - 2026-07-25

### Added
- Initial Streamlit recruiter interface with dual-pane chat and shortlist inspection.

## [0.2.0] - 2026-07-24

### Added
- Synthetic candidate resume dataset generator and hybrid semantic-BM25 matcher.

## [0.1.0] - 2026-07-22

### Added
- Initial project scaffolding and foundational LangGraph agent state machine.
