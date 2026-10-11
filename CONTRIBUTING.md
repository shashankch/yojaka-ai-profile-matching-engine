# Contributing to Yojaka AI (Agentic Profile Matching Engine)

Thank you for your interest in contributing to **Yojaka AI (Agentic Profile Matching Engine)**! We welcome contributions that improve agent capabilities, enhance test coverage, optimize RAG retrieval performance, harden security, or refine developer tooling.

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.10+ (tested on Python 3.11, 3.12, and 3.14)
- **Package Manager**: Standard `pip` (or `uv` for high-performance dependency management)
- **Git**: For version control and submitting Pull Requests
- **API Keys**: At least one supported LLM inference provider:
    - Groq (`GROQ_API_KEY`)
    - Google Gemini (`GEMINI_API_KEY`)
    - Sarvam AI (`SARVAM_API_KEY`)
    - OpenAI (`OPENAI_API_KEY`)
- **Live Search Key (Optional)**: Tavily AI (`TAVILY_API_KEY`) for real-time external web searching and candidate portfolio verification.

---

## 🛠️ Local Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shashankch/yojaka-ai-profile-matching-engine.git
   cd yojaka-ai-profile-matching-engine
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   # Using Python venv
   python3 -m venv .venv
   source .venv/bin/activate

   # Or using uv (recommended)
   uv venv
   source .venv/bin/activate
   ```

3. **Install Package in Editable Mode with Dev Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure Environment Secrets**
   Copy the example environment configuration:
   ```bash
   cp .env.example .env
   ```
   Add your active API credentials to `.env`:
   ```env
   GROQ_API_KEY="your-groq-api-key"
   GEMINI_API_KEY="your-gemini-api-key"
   SARVAM_API_KEY="your-sarvam-api-key"
   OPENAI_API_KEY="your-openai-api-key"
   TAVILY_API_KEY="your-tavily-api-key"  # Optional: for live web searching
   USE_MCP=False  # Set to True to test Model Context Protocol JSON-RPC mode
   ```

---

## 🧪 Running Tests & Quality Gates

Before submitting any code changes, ensure all unit tests pass and code quality checks pass.

1. **Run Full Test Suite**
   ```bash
   pytest tests/ -v
   ```

2. **Run Linter & Formatter Integrity**
   We enforce code formatting via [Ruff](https://astral.sh/ruff):
   ```bash
   # Check for lint violations
   ruff check .

   # Automatically format code
   ruff format .
   ```

3. **Run Static Type Checking (CI Gate)**
   ```bash
   mypy src/
   ```

4. **Install Local Pre-Commit Hooks (Recommended)**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

---

## 🏃 Running the Pipeline & Web Application Locally

Verify full end-to-end functionality locally:

```bash
# 1. Generate local mock candidate dataset (34 profiles)
python -m agentic_profile_matching.generate_dataset

# 2. Ingest candidate resumes into ChromaDB & BM25 index
python -m agentic_profile_matching.resume_rag

# 3. Launch interactive Streamlit GUI with live node checkpoints & in-memory resume upload
streamlit run src/agentic_profile_matching/app.py

# 4. Run automated test scenarios suite
python -m agentic_profile_matching.run_scenarios
```

---

## 📐 Architectural Guidelines & Conventions

All contributions must strictly comply with the project standards documented in:

- 📖 **[Engineering Conventions](docs/CONVENTIONS.md)** (`docs/CONVENTIONS.md`): Core architectural principles, state immutability (ADR-012), stateless credential isolation (ADR-011), structured JSON logging, and diagrams-as-code maintenance protocol.
- 🏛️ **[Technical Architecture](docs/architecture.md)** (`docs/architecture.md`): System architecture, cascading screening funnel, component boundaries, and dataflows.
- 📋 **[Architecture Decision Records](docs/adr/README.md)** (`docs/adr/README.md`): Formal design records (ADR-001 through ADR-019).

---

## 📝 Commit & PR Guidelines

1. **Conventional Commits**
   Format: `<type>(<scope>): <short summary>`

    - `feat(ingestion): add in-memory stream resume parsing`
    - `fix(matching): resolve semantic skill expansion evaluation`
    - `refactor(state): enforce copy-on-write immutability in deep screen`
    - `docs(adr): document zero-disk stream ingestion architecture`
    - `test(matcher): add test cases for cloud skill expansion`

2. **Pre-PR Checklist**
    - [ ] `pytest tests/ -v` passes (all unit and integration tests green).
    - [ ] `ruff check .` and `ruff format --check .` pass with 0 warnings.
    - [ ] Version incremented in `pyproject.toml` and release notes added to `CHANGELOG.md`.
    - [ ] Documentation updated in `README.md`, `docs/ROADMAP.md`, or `docs/adr/` if architectural changes were introduced.
    - [ ] Architecture diagrams updated in `scripts/generate_diagrams.py` and regenerated via `python scripts/generate_diagrams.py` if design/arch/flow changes were introduced.
    - [ ] Documentation site verified cleanly via `mkdocs build --strict`.

Thank you for helping make **Yojaka AI** better! 🌟

