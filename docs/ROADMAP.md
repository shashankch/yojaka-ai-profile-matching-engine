# 🗺️ Yojaka AI: Product Roadmap

This document outlines high-level implementation milestones and strategic release targets for **Yojaka AI (Agentic Profile Matching Engine)**.

> 📚 **Detailed Specifications**: For system architecture diagrams, mathematical formulations, and formal design decisions, see [**System Architecture**](architecture.md), [**Architecture Decision Records (ADRs)**](adr/index.md), and [**Changelog**](CHANGELOG.md).

---

## 📍 Implementation Milestones (Phases 1–19)

### Foundational Pipeline & Core State Machine (Phases 1–7 • v0.1.0) ✅
- **Agentic Core**: 9-node LangGraph `StateGraph` workflow with deterministic state transitions and `MemorySaver` checkpointing.
- **Tooling & Gateway**: Dual-mode tool execution gateway supporting local in-process and FastMCP JSON-RPC `stdio` servers ([ADR-001](adr/ADR-001-mcp-dual-mode-gateway-architecture.md)).
- **Candidate Vetting Tools**: Schema-validated requirement extraction, side-by-side comparison matrix, and tailored interview question generation.

### Enterprise Abstractions & Background Workers (Phase 8 • v0.8.0) ✅
- **Service Decoupling**: Dedicated `IngestionService` isolating core business logic from transport protocols.
- **Pluggable Storage**: `BaseVectorStore` structural protocol with ChromaDB default and Qdrant stub ([ADR-003](adr/ADR-003-basevectorstore-structural-protocol.md)).
- **Async Task Queue**: Distributed Celery workers with Redis 7 message broker and Docker Compose orchestration ([ADR-006](adr/ADR-006-celery-redis-task-queue.md)).

### Observability, Hybrid Retrieval & Multi-Provider Scale (Phases 9–10 • v1.0.0) ✅
- **Enterprise Observability**: Structured JSON logging and latency tracing with Langfuse and OpenTelemetry backends ([ADR-007](adr/ADR-007-structured-json-logging-and-tracing.md)).
- **Hybrid Scoring**: Min-max normalized dense vector cosine + BM25Okapi lexical retrieval with cached sparse index ([ADR-004](adr/ADR-004-bm25-corpus-index-caching.md), [ADR-008](adr/ADR-008-multi-factor-hybrid-scoring-and-hierarchy.md)).
- **Multi-Provider LLM Gateway**: Unified provider abstraction supporting Groq (Llama 3.3 70B), Google Gemini, Sarvam AI (105B Indic), and OpenAI ([ADR-010](adr/ADR-010-multi-provider-sarvam-indic-llm.md)).

### Security, State Invariance & Async Screening (Phases 11–12 • v1.1.0) ✅
- **Stateless Credential Isolation**: Credentials passed via `RunnableConfig`, eliminating secret exposure in checkpoints (CWE-312) ([ADR-011](adr/ADR-011-stateless-credential-isolation.md)).
- **Functional State Immutability**: Copy-on-write candidate dictionary updates ensuring safe LangGraph retries ([ADR-012](adr/ADR-012-functional-state-immutability.md)).
- **Concurrent Screening**: Parallel candidate screening with `ThreadPoolExecutor` and rate-limit semaphores ([ADR-014](adr/ADR-014-concurrency-controlled-async-screening.md)).

### Quality Gates & Engine Resilience (Phases 13–14 • v1.1.5) ✅
- **Automated Quality Gates**: Strict `mypy` static typing, `pytest-cov` ≥ 75% coverage gate, and `gitleaks`/`pip-audit` security scanning.
- **Deterministic Cache Correctness**: Global BM25 corpus fingerprinting and async subprocess lifecycle management.

### In-Memory Ingestion, Modern Routing & UI (Phase 15 • v1.2.0) ✅
- **Zero-Disk In-Memory Ingestion**: Stream-based file parsing (`io.BytesIO`) without disk persistence, layered with ephemeral session stores ([ADR-016](adr/ADR-016-zero-disk-in-memory-resume-ingestion.md)).
- **Dynamic Semantic Skill Expansion**: LLM requirement expansion with word-boundary matching, replacing brittle static taxonomies ([ADR-013](adr/ADR-013-dynamic-skills-taxonomy-alias-normalization.md)).
- **LLM Intent Routing & Caching**: Structured zero-shot intent router with dynamic anchor synthesis and sub-millisecond query caching ([ADR-009](adr/ADR-009-tiered-semantic-embedding-intent-routing.md)).
- **UI Ergonomics**: Compact zero-scroll recruiter sidebar with instant repository path auto-discovery.

### Modular UI Architecture & Headless API Sidecar (Phase 16 • v1.3.0) ✅
- **Modular Presentation Layer**: Monolithic `app.py` decomposed into an executive `<75-LOC` conductor and modular `ui/` components.
- **Streamlit-Native Streaming**: Real-time typewriter output via `st.write_stream()` and dynamic intent-aware execution tracking.
- **Headless FastAPI Gateway**: Additive ASGI sidecar exposing REST and Server-Sent Events (SSE) streaming endpoints.

### Layout Parsing, Contextual Retrieval & Two-Stage Reranking (Phase 17 • v1.4.0) ✅
- **Layout-Aware Section Parsing**: PyMuPDF visual bounding block extraction preserving work experience integrity without boundary truncation.
- **Anthropic Contextual Retrieval Prepending**: Document-level metadata banners prepended to chunks before dense and BM25 embedding.
- **Two-Stage Hybrid Reranking**: Fine-grained Cross-Encoder joint reranking (`ms-marco-MiniLM-L-6-v2`) with Sigmoid calibration and Reciprocal Rank Fusion.
- **Serverless Cloud Resilience**: Zero-disk pure NumPy `InMemoryVectorStore` fallback ensuring 100% crash-free ephemeral execution.

### Calibrated Margin Routing, Native Commands & Dual-Rubric Subgraphs (Phase 18 • v1.5.0) ✅
- **Calibrated Margin Routing**: Mathematical embedding margin scoring ($\Delta \ge 0.12$) routing confident queries in < 2ms at $0 LLM cost while escalating ambiguity to structured LLMs ([ADR-018](adr/ADR-018-calibrated-margin-routing-and-subgraphs.md)).
- **LangGraph Native Commands**: Refactored workflow nodes to native LangGraph `Command(goto=..., update={...})` primitives with destination routing, eliminating conditional edge boilerplate.
- **Parallel Dual-Rubric Structured Evaluation**: Concurrent structured scoring (Technical Architecture Competence 60% + Talent Sourcing Fit 40%) with deterministic mathematical aggregation, eliminating slow debate loops while delivering committee-grade scorecards.
- **Modular Typed Subgraphs**: Pipeline decomposed into isolated, independently testable subgraphs (`JDAnalyzerSubgraph`, `TalentRetrievalSubgraph`, `DeepScreeningSubgraph`, `SynthesisSubgraph`).

### Advanced RAG Architecture & Air-Gapped Local Inference (Phase 19 • v1.6.0) ✅
- **HyDE (Hypothetical Document Embeddings)**: Query expansion synthesizing realistic candidate profile summaries from job descriptions before vector search to bridge vocabulary mismatch ([ADR-019](adr/ADR-019-advanced-rag-hyde-parent-doc-and-local-inference.md)).
- **Hierarchical Parent-Document Chunking**: Fine-grained 150–250 token child vector retrieval linked to 1,000–1,500 character parent sections in `ParentDocumentStore`, eliminating context fragmentation during screening.
- **Air-Gapped Local Inference**: Turnkey offline model execution via Ollama and vLLM daemons for GDPR sovereign cloud compliance and zero cloud API costs.
- **Pre-Retrieval Faceted Filtering**: Direct vector and lexical metadata constraints for experience years and education criteria with dynamic skill expansion support.

---

## 🚀 Future Milestones (Phases 20–22)

### Phase 20: Resume Threat Security, Zero-Trust PII Redaction Vault & Multi-Tenancy (v1.7.0)
- **Indirect Prompt Injection Defense (OWASP LLM01)**: Input sanitization heuristics, canary detection, and strict XML boundary tags neutralizing hidden resume injection payloads.
- **Zero-Trust PII Tokenization Vault**: Pre-screening entity tokenization replacing personal candidate identifiers (`[CANDIDATE_A]`) for EEOC blind hiring meritocracy.
- **Reversible Recruiter Vault**: AES-256 encrypted candidate identity mapping revealed only upon authorized recruiter outreach with immutable audit logging.
- **Hardware-Partitioned Multi-Tenancy**: Isolated collection namespaces across vector stores (`tenant_{org_id}_resumes`) and session checkpoints.

### Phase 21: Unit Economics, Semantic Caching & Budget Circuit Breakers (v1.8.0)
- **Pool-Aware Semantic Evaluation Cache**: Sub-5ms response caching for semantically equivalent recruiter queries, guarded by pool-version hashes to prevent stale candidate rankings.
- **Granular Cost & Token Budgeting**: Predictive `tiktoken` calculation, real-time per-query and per-tenant cost accounting, daily spend caps, and runaway loop circuit breakers.
- **Job Description Inclusivity Scanner**: Automated open-source linguistic analysis flagging exclusionary or demographic-biased wording and generating auditable competency-grounded rationale trails.

### Phase 22: Production Infrastructure, Containerization & Continuous Evals (v2.0.0)
- **Multi-Stage Hardened Production Dockerfile**: Minimal non-root container image (<250MB) isolating build dependencies for ASGI sidecar and workers.
- **Docker Compose Multi-Service Topology**: Single-command orchestration for FastAPI sidecar, Streamlit UI, Celery worker, Redis 7, vector services, and self-hosted Langfuse.
- **Continuous Ragas & DeepEval CI/CD Quality Gates**: Automated RAG Triad benchmarks (Context Precision $\ge 0.88$, Recall@K, Faithfulness $\ge 0.90$) blocking pull requests on score regression.
- **Deep Health Probes & Rate Limiting**: `/healthz` and `/readyz` probing database, Redis, and workers with `slowapi` rate-limiting shield.
