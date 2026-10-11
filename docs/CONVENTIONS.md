# Yojaka AI: Engineering Conventions & Guidelines

This document outlines the core architectural principles, coding standards, and quality conventions for **Yojaka AI**. These guidelines keep our codebase modular, reliable, and straightforward for contributors.

---

## 🏛️ 1. Architectural Principles

1. **Separation of Concerns**:
    - **Domain Logic**: RAG retrieval (`resume_rag.py`), hybrid matching (`job_matcher.py`), and LangGraph state machines (`agent/`) must remain decoupled from presentation layers (Streamlit UI) and transport protocols (FastAPI / MCP).
    - **Service Layer**: High-level orchestrations spanning multiple domain boundaries belong in `src/agentic_profile_matching/services/` (e.g., `IngestionService`).

2. **Protocol-Driven Interfaces & Dependency Injection**:
    - Orchestrators depend on abstractions (`typing.Protocol` or `abc.ABC`), not hardcoded implementations (e.g., `BaseVectorStore` instead of direct `ChromaDB` dependencies).
    - Components receive clients and model references via constructor arguments or runtime config rather than instantiating them internally.

---

## 🛡️ 2. State Management & Type Safety

1. **Strict Type Annotations**:
    - All functions, methods, class attributes, and module constants must use explicit Python 3.10+ type hints.
    - Avoid `Any` where possible. Use explicit generics (`list[str]`, `dict[str, float]`) and `Optional`/`Union`.
    - Numeric match scores must be typed as `float` across all schemas to avoid truncation.

2. **Pydantic V2 for Validation & Structured LLM Extraction**:
    - Use Pydantic V2 `BaseModel` for request/response payloads, config validation, and structured LLM tool outputs.
    - Parse LLM structured responses using schema-enforced methods with deterministic fallbacks.

3. **LangGraph State Immutability**:
    - Graph state schemas in `agent/state.py` use `TypedDict` or Pydantic models.
    - **Pure updates**: Never mutate state dictionaries in place. Return fresh partial state dictionaries (`{**existing, "key": value}`) from graph nodes so checkpoints remain deterministic.

---

## 🔒 3. Security & Resilience

1. **Zero Secret Leakage in Serialized State**:
    - API keys, credentials, and tokens must **never** be written to `AgentState` or stored in serializable checkpoints.
    - Inject credentials dynamically via runtime context (`RunnableConfig`) or environment variables.
    - Structured loggers sanitize sensitive values and authorization tokens.

2. **Input & UI Sanitization (OWASP)**:
    - Candidate-derived content (names, experiences, education) rendered in HTML templates (such as candidate cards) must be escaped with `html.escape()` to prevent cross-site scripting (XSS).

3. **Graceful Degradation & Error Boundaries**:
    - Catch specific exceptions (`FileNotFoundError`, `KeyError`, `httpx.HTTPError`) rather than bare `except Exception: pass`.
    - Non-critical node failures (e.g., secondary tool or enrichment calls) should log structured warnings and populate state flags without crashing the primary evaluation pipeline.

---

## 📊 4. Observability & Logging

1. **Structured JSON Logs**:
    - Use `get_logger(__name__)` from `agentic_profile_matching.observability`.
    - Bare `print()` calls in core/service modules are prohibited.
    - All log records emit structured JSON containing `timestamp`, `level`, `logger`, `message`, and trace payloads.

2. **Latency Instrumentation**:
    - Workflow nodes in `agent/workflow.py` and subgraphs are instrumented with `@trace_node("node_name")` to profile execution latency.

---

## 🧪 5. Testing & Code Quality Gates

Before opening a pull request, verify all local quality gates:

```bash
# 1. Lint and format checks
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# 2. Test suite execution
uv run pytest tests/ -v
```

- **Mock External Services**: Unit tests must not make live billable LLM or external network calls; use pytest fixtures and mock responses.
- **Deterministic Math**: Scoring algorithms (e.g., hybrid BM25 + vector weighting, calibrated margin routing) must be covered with explicit assertions.

---

## 📝 6. Commit & Versioning Standards

1. **Conventional Commits**:
    - Format: `<type>(<scope>): <summary>`
    - Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
    - Example: `feat(routing): calibrated margin routing with subgraphs (v0.18.0)`

2. **Changelog & Semantic Versioning**:
    - Increment `version` in `pyproject.toml` and record changes in `CHANGELOG.md` under the corresponding version section (`[MAJOR.MINOR.PATCH] - YYYY-MM-DD`).

---

<a id="diagrams-maintenance-protocol"></a>
## 🎨 7. Architecture & Diagrams Maintenance Protocol

To ensure documentation remains accurate and publication-grade as the system evolves:

1. **Diagrams as Code (`scripts/generate_diagrams.py`)**:
    - Architecture diagrams are defined programmatically using Mingrammer `diagrams`. Manual graphic edits or ad-hoc image uploads are prohibited.
    - When introducing new architectural components, subgraphs, storage engines, or routing mechanisms, update the corresponding generator function in `scripts/generate_diagrams.py`.

2. **Visual Standards**:
    - Use clean planar orientations (`direction="LR"` for pipelines, `"TB"` for hierarchies).
    - Use official high-resolution brand icons from `docs/assets/icons/`.
    - Use the centralized `cluster_attr()` helper with generous padding (≥28pt) to eliminate node label collision.

3. **Regeneration & Verification**:
    ```bash
    # Regenerate diagrams
    uv run python scripts/generate_diagrams.py

    # Verify documentation links and build integrity
    uv run mkdocs build --strict
    ```
