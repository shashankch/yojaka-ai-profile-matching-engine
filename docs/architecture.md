# Yojaka AI (Agentic Profile Matching Engine): Technical Architecture

This document provides the comprehensive technical architecture, dataflow sequence diagrams, mathematical scoring formulations, state machine specifications, and system designs for **Yojaka AI (Agentic Profile Matching Engine)**.

---

## 1. C4 Architecture — Progressive Abstraction

The system architecture is modeled using the **C4 Model** (Context → Containers → Components) to communicate clearly at each level of abstraction: from executive business context down to engineering component boundaries.

---

### Level 1 — System Context (C4-L1)

> _Who uses the system and what external systems does it depend on?_

```mermaid
graph TB
    classDef person fill:#08427B,color:#fff,stroke:#073B6F,rx:50
    classDef system fill:#1168BD,color:#fff,stroke:#0E5CA6
    classDef external fill:#999,color:#fff,stroke:#7A7A7A
    classDef externalDb fill:#555,color:#fff,stroke:#444

    Recruiter(["👤 Recruiter / Hiring Manager\n──────────────────\nSends job descriptions,\nuploads candidate resumes,\nreviews shortlists & candidate\ncomparison reports interactively"])

    Engine["⚙️ Yojaka AI (Agentic Profile Matching Engine)\n──────────────────────────────────\nCoordinates 3-round cascading candidate\nscreening: coarse ranking → LLM deep audit\n→ grounded hiring recommendations + QGen\n──────────────────────────────────\n[Python 3.12+ · LangGraph · Streamlit · FastMCP]"]

    LLMs["🤖 LLM Inference Providers\n──────────────────\nGroq · Llama 3.3 70B\nGoogle Gemini 2.0 Pro\nSarvam AI 105B (Indic)\nOpenAI GPT-4o\n──────────────────\n[REST / HTTPS]"]

    APM["📊 APM & Observability Platform\n──────────────────\nLangfuse · OpenTelemetry\nDatadog · AWS CloudWatch\n──────────────────\n[OTLP / HTTPS]"]

    Corpus[("📁 Resume Document Store\n──────────────────\nPDF · DOCX · TXT\nZero-Disk Memory Streams / Disk\n──────────────────\n[PyMuPDF / stdio / BytesIO]")]

    Recruiter -->|"Submits JD, uploads resumes,\nreviews ranked shortlists\n[HTTPS · Port 8501]"| Engine
    Engine -->|"Requirements extraction,\ndeep profile audits, QGen\n[REST · JSON / HTTPS]"| LLMs
    Engine -->|"Emits structured JSON logs,\n@trace_node OTLP spans\n[OTLP / HTTPS]"| APM
    Engine -->|"Ingests layout-sorted chunks\n& in-memory resume streams\n[PyMuPDF / fs_tools / io.BytesIO]"| Corpus

    class Recruiter person
    class Engine system
    class LLMs,APM external
    class Corpus externalDb
```

---

### Level 2 — Container Diagram (C4-L2)

> _What are the main deployable units, data stores, and how do they communicate?_

```mermaid
graph TB
    classDef person fill:#08427B,color:#fff,stroke:#073B6F
    classDef container fill:#1168BD,color:#fff,stroke:#0E5CA6
    classDef database fill:#2E7D32,color:#fff,stroke:#1B5E20
    classDef queue fill:#6A1B9A,color:#fff,stroke:#4A148C
    classDef external fill:#999,color:#fff,stroke:#7A7A7A
    classDef boundary fill:none,stroke:#CCC,stroke-dasharray:6,color:#555

    Recruiter(["👤 Recruiter"])

    subgraph Stack["  📦 Yojaka AI Stack  "]
        UI["🖥️ Streamlit Presentation Layer\n─────────────────────\nDual-pane chat workspace\nSidebar & Tab In-Memory Resume Ingestion\nLive st.status node checkpoints\nComparison matrix + report export\n─────────────────────\n[Python 3.12+ · Streamlit · Port 8501]"]


        Agent["🧠 LangGraph Agentic Workflow Engine\n─────────────────────\n9-Node StateGraph with TypedDict AgentState\nMemorySaver checkpointing per thread_id\nTimered intent router (Tier 1 + Tier 2)\n─────────────────────\n[Python 3.12 · LangGraph 0.2 · agent/]"]

        Gateway["🔌 Dual-Mode Tool Gateway\n─────────────────────\nfs_client.py bridges direct in-process calls\nand stdio JSON-RPC 2.0 FastMCP transport\nToggled via USE_MCP env flag (ADR-001)\n─────────────────────\n[Python · FastMCP · mcp_client.py]"]

        Workers["⚙️ Async Background Task Workers\n─────────────────────\nasync_ingest_directory: bulk PDF chunking\nasync_deep_screen_candidate: LLM batch audit\nRate-limited via task routing keys\n─────────────────────\n[Celery 5.4 · Python · ADR-006]"]

        Redis[("🟥 Redis Task Broker & State Backend\n─────────────────────\nCelery task queue (FIFO)\nWorker heartbeat & execution state\n─────────────────────\n[Redis 7 Alpine · Port 6379]")]

        VectorDB[("🗄️ Vector & Sparse Retrieval Index\n─────────────────────\nCompositeVectorStore: Layered hybrid storage\nBase: Read-only persistent disk pool\nEphemeral: Session-scoped in-memory PII uploads\nBM25Okapi: Sparse keyword index with invalidation\nAll upserts are idempotent (ADR-005, ADR-016)\n─────────────────────\n[ChromaDB / Qdrant · In-Memory BM25]")]
    end

    LLM_APIs["🤖 LLM Inference APIs\n[Groq · Gemini · Sarvam · OpenAI]"]
    APM_Ext["📊 APM Backends\n[Langfuse · OpenTelemetry]"]

    Recruiter -->|"Browser HTTP"| UI
    UI -->|"execute_graph(input, thread_id)\n[In-Process / Thread]"| Agent
    Agent -->|"Invokes tool operations\n[Python call / FastMCP stdio]"| Gateway
    Gateway -->|"Vector similarity + BM25 queries\n[In-Memory / SQLite / HTTP]"| VectorDB
    Agent -->|"apply_async() → Celery task\n[AMQP-over-Redis]"| Workers
    Workers -->|"BRPOP / BLPOP task messages\n[Redis Protocol]"| Redis
    Workers -->|"Upsert document chunks"| VectorDB
    Agent -->|"with_structured_output()\n[HTTPS / REST]"| LLM_APIs
    Agent -.->|"@trace_node spans\n[OTLP / JSON]"| APM_Ext

    class Recruiter person
    class UI,Agent,Gateway,Workers container
    class Redis,VectorDB database
    class LLM_APIs,APM_Ext external
```

---

### Level 3 — Component Diagram: LangGraph Agentic Core (C4-L3)

> _What are the internal components of the LangGraph Workflow Engine and how do they collaborate?_

```mermaid
graph TB
    classDef router fill:#E65100,color:#fff,stroke:#BF360C
    classDef node fill:#1565C0,color:#fff,stroke:#0D47A1
    classDef state fill:#2E7D32,color:#fff,stroke:#1B5E20
    classDef tool fill:#6A1B9A,color:#fff,stroke:#4A148C
    classDef external fill:#999,color:#fff,stroke:#7A7A7A

    subgraph AgentCore["  🧠 LangGraph Agentic Workflow Engine (agent/)  "]

        subgraph Routing["  🔀 LLM-Driven Intent Routing (ADR-009)  "]
            FastPath["⚡ Structural Fast-Path & Query Cache\nRaw multi-line JD paste (0ms)\nIn-memory LRU query cache (0ms)"]
            PrimaryLLM["🧠 Primary Tier: LLM Intent Classifier\nwith_structured_output(RouteDecision)\nFull natural language understanding\nZero hardcoded strings or rigid anchors"]
            DynamicAnchors["🔵 Secondary Tier: Dynamic Intent Anchors\nSentenceTransformer cosine similarity against\nLLM-synthesized intent prototypes (fallback)"]
        end

        subgraph ParseNodes["  📥 Input & Requirements Nodes  "]
            ParseInput["parse_input_node\nPre-processes message,\ninitialises session fields"]
            ExtractReq["extract_requirements_node\nPydantic V2 structured extraction\nof JobRequirements from raw JD text"]
            AdjustReq["adjust_requirements_node\nApplies recruiter refinements\n(must-have updates, exp adjustments)"]
        end

        subgraph RetrievalNodes["  🔍 Retrieval & Ranking Nodes  "]
            SearchRes["search_resumes_node\nExecutes hybrid JobMatcher query\n(dense vector + BM25 sparse)"]
            RankCand["rank_candidates_node  [Round 1]\nMulti-factor score (0–100)\nSlices coarse Top 10 shortlist"]
        end

        subgraph ScreenNodes["  🧪 LLM Screening & Guardrail Nodes  "]
            DeepScreen["deep_screen_node  [Round 2]\nLLM full-resume audit per profile\nExtracts strengths, gaps, status\n(copy-on-write, ADR-012)"]
            Recommend["recommendation_node  [Round 3]\nGrounded status hierarchy\nHard safety guardrails:\nmissing skills / exp deficit overrides\nInterview QGen (3–5 tailored Qs)"]
        end

        subgraph SynthNodes["  📋 Synthesis & Conversation Nodes  "]
            GenReport["generate_report_node\nCompiles markdown comparison\nmatrix and audit report"]
            ConvQuery["conversational_query_node\nReAct loop: compare, Q&A,\nweb search, candidate notes"]
        end

        StateStore["🗃️ TypedDict AgentState\n+ MemorySaver Checkpoint\n─────────────────────\nAgentState per thread_id\nrequirements · shortlist · messages\nNo credentials in state (ADR-011)"]
    end

    subgraph Tools["  🔧 Tool & Service Layer  "]
        JobMatcher["JobMatcher\nHybrid scoring engine\n(job_matcher.py)"]
        IngestionSvc["IngestionService\nDocument chunking\n+ vector upsert"]
        MCPTools["MCP Tool Handlers\n(fs_client.py / mcp_client.py)"]
    end

    LLMProvider["🤖 LLM Provider\n(config.py PROVIDER_REGISTRY)\n[Groq · Gemini · Sarvam · OpenAI]"]

    ParseInput --> FastPath
    FastPath -->|"raw multi-line JD"| ExtractReq
    FastPath -->|"cache miss"| PrimaryLLM
    PrimaryLLM -->|"sourcing / new JD"| ExtractReq
    PrimaryLLM -->|"constraint refinement"| AdjustReq
    PrimaryLLM -->|"tech / search / compare"| ConvQuery
    PrimaryLLM -.->|"offline / fallback"| DynamicAnchors
    DynamicAnchors --> ExtractReq
    DynamicAnchors --> AdjustReq
    DynamicAnchors --> ConvQuery
    ExtractReq --> SearchRes
    AdjustReq --> SearchRes
    SearchRes --> RankCand
    RankCand -->|"Top 10 → Top 5"| DeepScreen
    DeepScreen --> Recommend
    Recommend --> GenReport
    GenReport --> StateStore
    ConvQuery --> StateStore

    SearchRes --> JobMatcher
    GenReport --> MCPTools
    ConvQuery --> MCPTools
    MCPTools --> IngestionSvc
    DeepScreen --> LLMProvider
    Recommend --> LLMProvider
    ExtractReq --> LLMProvider

    class FastCheck,Tier1,Tier2 router
    class ParseInput,ExtractReq,AdjustReq,SearchRes,RankCand,DeepScreen,Recommend,GenReport,ConvQuery node
    class StateStore state
    class JobMatcher,IngestionSvc,MCPTools tool
    class LLMProvider external
```

---

### Core Architectural Principles
1. **Separation of Concerns (SoC)**: Presentation logic (Streamlit), agent orchestration (LangGraph), business services (`IngestionService`), and transport protocols (FastMCP) remain completely decoupled.
2. **Protocol-Driven Abstraction**: Vector store operations conform to the `BaseVectorStore` structural protocol (`typing.Protocol`), enabling zero-code changes when switching between ChromaDB, Qdrant, or cloud vector stores.
3. **Idempotency by Design**: Ingestion keys are deterministically generated from document names and section indices (`{file}_chunk_{idx}`), guaranteeing safe, duplicate-free re-indexing.
4. **Cascading Cost & Latency Optimization**: Dense embedding and BM25 scoring narrow large candidate pools down to top contenders locally ($0$ LLM cost), running compute-intensive LLM deep audits strictly on the top-ranked candidates ($O(N) \to O(K)$).

---

## 2. End-to-End Execution Sequence

The sequence diagram below traces the end-to-end execution lifecycle from initial job description input through coarse ranking, LLM deep screening, hiring recommendation, and conversational constraint refinement:

```mermaid
sequenceDiagram
    autonumber
    actor Recruiter as Recruiter / Hiring Manager
    participant UI as Streamlit UI
    participant Ingestion as IngestionService
    participant Agent as LangGraph Orchestrator
    participant Parser as parse_input_node
    participant Search as search_resumes_node
    participant Matcher as JobMatcher (Hybrid Engine)
    participant DeepScreen as deep_screen_node
    participant Recommend as recommendation_node
    participant LLM as External LLM (Groq / Gemini)

    opt Zero-Disk In-Memory Resume Ingestion & Ephemeral Isolation (ADR-016)
        Recruiter->>UI: Uploads candidate resume(s) via Tab 2
        UI->>UI: Enforce 10MB limit & compute content digest
        UI->>Ingestion: ingest_stream(filename, bytes) into ephemeral_store
        Ingestion->>Matcher: Upsert vectorized chunks to ephemeral ChromaDB
        Ingestion-->>UI: Return IngestionResult(success=True, chunks=N)
        UI-->>Recruiter: Updated live talent pool count & candidate table
    end

    Recruiter->>UI: Submit Job Description ("Looking for Python Dev with 3+ yrs exp")
    UI->>Agent: execute_graph(user_input, thread_id)
    Agent->>Parser: Classify input type
    Parser-->>Agent: route -> extract_requirements
    Agent->>LLM: extract_requirements(jd_text)
    LLM-->>Agent: JobRequirements(skills=['Python'], min_exp=3)
    
    Agent->>Search: search_resumes_node(requirements)
    Search->>Matcher: match(query, min_exp=3, must_have=['Python'])
    Matcher->>Matcher: 1. Vector Search + Min-Max Scaling (50%)
    Matcher->>Matcher: 2. Stop-word Filtered BM25 (35%)
    Matcher->>Matcher: 3. Skills/Exp Satisfaction (15%)
    Matcher-->>Search: Top 10 Shortlisted Candidates (Coarse Filter)
    
    Agent->>DeepScreen: deep_screen_node(Top 5 Candidates)
    loop For Each Top Candidate
        DeepScreen->>LLM: deep_screen_prompt(Candidate Resume + JD)
        LLM-->>DeepScreen: Strengths, Gaps, Status ("Strong Hire")
    end
    
    Agent->>Recommend: recommendation_node()
    Recommend->>Recommend: Apply Hard Safety Guardrails (Check Missing Skills/Exp)
    Recommend->>LLM: generate_custom_questions(Gaps)
    LLM-->>Recommend: 3-5 Tailored Technical Interview Questions
    
    Agent->>UI: Render Shortlist Cards, Comparison Matrix & Audit Reports
    UI-->>Recruiter: Interactive Recruiter Dashboard
    
    opt Conversational Refinement
        Recruiter->>UI: "Make Kubernetes a must-have skill"
        UI->>Agent: execute_graph("Make Kubernetes a must-have", thread_id)
        Agent->>Parser: Route -> adjust_requirements
        Agent->>Search: Re-run Search & Scoring
        Agent->>UI: Render Updated Shortlist + Ranking Changes Explanation
    end

    opt General Tech Inquiry / Web Search (Zero Candidate Screening)
        Recruiter->>UI: "Tell me about graph engineering in 2026"
        UI->>Agent: execute_graph("Tell me about graph engineering in 2026", thread_id)
        Agent->>Parser: Route via LLM / Cache -> conversational_query
        Agent->>ConvQuery: conversational_query_node()
        ConvQuery->>LLM: Answer tech question / search external web
        ConvQuery-->>Agent: Comprehensive 2026 tech analysis
        Agent->>UI: Render chat message (Candidate shortlist unchanged)
        UI-->>Recruiter: Direct answer without screening cards
    end
```

---

## 3. LangGraph Agent Workflow & State Machine

### A. Graph State Design (`AgentState`)
The agent state schema (`agent/state.py`) is defined as a Python `TypedDict`, avoiding serialization overhead during checkpoint transitions while enforcing strict Pydantic V2 schemas at LLM boundaries:

```python
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class JobRequirements(TypedDict, total=False):
    title: str
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    min_experience_years: int
    education_level: str
    other_constraints: List[str]

class CandidateMatch(TypedDict, total=False):
    candidate_id: str
    name: str
    score: int
    matched_skills: List[str]
    missing_skills: List[str]
    experience_years: int
    education: str
    relevance_excerpts: List[str]
    strengths: List[str]
    gaps: List[str]
    improvement_suggestions: str
    screening_status: str
    screening_reasoning: str
    interview_questions: List[str]

class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    requirements: JobRequirements
    shortlist: List[CandidateMatch]
    previous_shortlist: List[CandidateMatch]
    ranking_explanation: str
    coarse_screen_limit: Optional[int]
    deep_screen_limit: Optional[int]
    recommendation_limit: Optional[int]
    current_round: int
    final_report: str
    feedback_pending: bool
    user_feedback: str
    errors: List[str]
```

### B. State Graph Topology & LLM-Driven Intent Routing (ADR-009)
The workflow is implemented as a 9-node `StateGraph` compiled with `MemorySaver` in-memory checkpointing for persistent session tracking via `thread_id`. 

Incoming recruiter messages are classified using an **LLM-Driven Intent Router with Dynamic Anchors and Query Caching** (`agent/routers.py`):
1. **Structural Fast Path (0ms)**: Multi-line pasted JDs or empty initial state route immediately to `extract_requirements`.
2. **In-Memory LRU Query Cache (0ms)**: Normalizes and MD5-hashes repeated queries, yielding sub-millisecond execution for frequent questions and intent patterns.
3. **Primary Tier — LLM Intent Classifier**: Invokes the LLM via `with_structured_output(RouteDecision)` with active session context. Provides zero-shot generalisation across natural language phrasing without hardcoded string arrays.
4. **Secondary Tier — Dynamic Semantic Embedding Router (Fallback)**: When LLM APIs are offline or unreachable, calculates cosine similarity against dynamic intent prototypes synthesized by the LLM (`generate_dynamic_intent_anchors`) or rich semantic descriptions (`all-MiniLM-L6-v2`) with a tuned threshold ($0.20$).

```mermaid
graph TD
    START([Start / Recruiter Input]) --> parse_input[parse_input_node]
    
    parse_input --> Router{"🔀 LLM-Driven Intent Router<br/>(Cache Hit 0ms ➔ Primary: LLM RouteDecision<br/>➔ Fallback: Dynamic Semantic Embeddings)"}
    
    Router -- "extract_requirements<br/>(New Search / JD)" --> extract_req[extract_requirements_node]
    Router -- "conversational_query<br/>(Compare / Q&A / Web)" --> conv_query[conversational_query_node]
    Router -- "adjust_requirements<br/>(Refine / Filter)" --> adjust_req[adjust_requirements_node]
    
    extract_req --> search_resumes[search_resumes_node]
    adjust_req --> search_resumes
    
    search_resumes --> rank_candidates[rank_candidates_node]
    rank_candidates --> deep_screen[deep_screen_node]
    deep_screen --> recommend[recommendation_node]
    recommend --> gen_report[generate_report_node]
    
    gen_report --> END([End / Recruiter View])
    conv_query --> END
```

### C. Node Responsibilities & Specifications

| Node | Purpose | Inputs | Outputs | Error Boundary / Fallback |
|:---|:---|:---|:---|:---|
| **`parse_input_node`** | Pre-processes incoming message and initializes session state | `messages[-1]` | Clean state | Preserves existing state on empty inputs |
| **`extract_requirements_node`** | Extracts structured requirements from raw JDs via `invoke_structured` | Raw JD string | `JobRequirements` | Pydantic validation fallback |
| **`adjust_requirements_node`** | Updates active requirements based on recruiter comments/slider changes | Active requirements + comment | Modified `JobRequirements` | Retains previous requirements on failure |
| **`conversational_query_node`** | Answers free-form recruiter questions via ReAct loop with MCP tools | User question + context | Response message | Direct context answering without tools |
| **`search_resumes_node`** | Queries ChromaDB and BM25 index via `JobMatcher` | `JobRequirements` | Raw candidate matches | Returns empty list if no matches found |
| **`rank_candidates_node` (Round 1)** | Applies multi-factor hybrid scoring and slices top $N$ candidates | Candidate matches | Shortlist (Top 10) | Preserves existing sort order |
| **`deep_screen_node` (Round 2)** | Sequential LLM audit via `invoke_structured` extracting strengths/gaps | Shortlist (Top 5) | Enriched `CandidateMatch` | Default strengths/gaps on LLM failure |
| **`recommendation_node` (Round 3)** | Applies safety guardrails and synthesizes interview questions | Enriched Shortlist | Final status + questions | Status override on skill/exp deficits |
| **`generate_report_node`** | Compiles markdown comparison matrix and chat response | Full Shortlist | `final_report` markdown | Generates fallback text summary |

---

## 4. Hybrid Search & Multi-Factor Scoring Engine

Candidate matching in `job_matcher.py` combines dense semantic search (ChromaDB), sparse lexical search (BM25 Okapi), and hard qualification constraints into a deterministic **0–100 Match Score**.

```mermaid
graph LR
    subgraph Inputs ["Query & Filters"]
        JD["Job Query / Requirements"]
        Filters["Must-Have Skills & Min Exp"]
    end

    subgraph Scoring ["Multi-Factor Scoring Pipeline"]
        Dense["1. Dense Vector Search<br/>Min-Max Scaling (1.0 to 0.5)"]
        Sparse["2. BM25 Sparse Search<br/>Stop-Word Filtered + Normalized"]
        Quals["3. Qualification Ratios<br/>Skill Match % + Experience Ratio"]
    end

    subgraph Output ["Final Candidate Score (0-100)"]
        Combined["Dynamic Weighted Sum<br/>50% Hybrid + 35% Skills + 15% Exp"]
    end

    JD --> Dense
    JD --> Sparse
    Filters --> Quals
    Dense --> Combined
    Sparse --> Combined
    Quals --> Combined
```

### A. Algorithmic Breakdown

1. **Min-Max Normalized Vector Similarity**:
   - *Problem*: Raw cosine distance in embedding spaces clusters tightly (e.g. `0.35`–`0.55`), causing top candidates to receive artificially deflated scores.
   - *Engineering Solution*: Map the closest vector in the retrieved batch to `1.0` (100% similarity) and scale linearly down to `0.50` for the furthest vector:
     ```python
     # Scales nearest match to 1.0 and furthest to 0.50
     norm_sim = 1.0 - 0.5 * ((dist - min_dist) / max(1e-5, max_dist - min_dist))
     semantic_score = max(0.0, min(1.0, norm_sim))
     ```

2. **Stop-Word Filtered BM25 Keyword Search**:
   - *Problem*: High-frequency recruiting noise tokens (`"looking"`, `"for"`, `"years"`, `"experience"`) distort BM25 term frequency calculations.
   - *Engineering Solution*: Filter queries against a curated stop-word set before querying the cached in-memory `BM25Okapi` sparse matrix:
     ```python
     stop_words = {"the", "and", "for", "with", "looking", "years", "exp", ...}
     tokens = [w for w in re.findall(r"\b\w+\b", query.lower()) if len(w) > 1 and w not in stop_words]
     bm25_score = bm25_index.get_scores(tokens)[chunk_idx] / max_bm25_score
     ```

3. **Dynamic Weight Allocation**:
   The engine automatically adapts component weights based on whether explicit must-have skills or experience bounds are supplied:

   | Scenario | Raw Hybrid (60% Dense + 40% Sparse) | Skill Match Ratio | Experience Satisfaction |
   |:---|:---:|:---:|:---:|
   | **With Must-Have Skills** | **50%** | **35%** | **15%** |
   | **General Semantic Search (No Must-Haves)** | **85%** | **0%** | **15%** |
   | **Zero Constraints Specified** | **100%** | **0%** | **0%** |

---

## 5. Tooling Layer & Model Context Protocol (MCP)

The engine implements a **Dual-Mode Gateway Architecture** (ADR-001) toggled dynamically via `config.USE_MCP`:

```mermaid
graph LR
    Agent[LangGraph Nodes] --> Gateway[fs_client.py Gateway]
    
    subgraph DirectMode ["Local Mode (USE_MCP=False)"]
        Gateway --> InProcess["Direct in-process call<br/>(fs_tools.py, job_matcher.py)"]
    end

    subgraph MCPMode ["MCP Protocol Mode (USE_MCP=True)"]
        Gateway --> Manager["mcp_client.py ClientSession Manager"]
        Manager --> StdioTransport["stdio JSON-RPC 2.0 Transport"]
        StdioTransport --> FSServer["filesystem_mcp_server.py (FastMCP)"]
        StdioTransport --> SearchServer["search_mcp_server.py (FastMCP)"]
    end
```

### Exposed Protocol Tools & Resources
- **Filesystem Server (`filesystem_mcp_server.py`)**:
  - Tools: `list_files`, `read_file`, `search_in_file`.
  - Resources: `resumes://all`, `resumes://{filename}`.
  - Background Watcher: `FileSystemWatcher` triggers auto-ingestion on directory file modifications.
- **Search Server (`search_mcp_server.py`)**:
  - Tools: `search_web` (Tavily API search with fallback mock profiles), `search_candidates` (ChromaDB vector lookup), `fetch_candidate_notes` (Internal HR screening file fetcher).

---

## 6. Distributed Task Queue (Celery + Redis)

For production deployments handling bulk document parsing and parallel LLM audits, the engine provides an asynchronous task processing layer via **Celery** and **Redis** (ADR-006):

```mermaid
graph LR
    subgraph Producers ["Task Producers"]
        WebUI["Streamlit Web App"]
        MCPServer["MCP Tool Handlers"]
    end

    subgraph Broker ["Redis 7 Broker & State Backend"]
        RedisQueue["Redis Queue (Port 6379)<br/>Task Serialization & Celery Result Backend"]
    end

    subgraph Workers ["Celery Worker Pool"]
        Worker1["Celery Worker Process 1<br/>(async_ingest_directory)"]
        Worker2["Celery Worker Process 2<br/>(async_deep_screen_candidate)"]
    end

    subgraph Storage ["Persistent Stores"]
        VectorDB["ChromaDB / Qdrant"]
    end

    Producers -->|"delay() / apply_async()"| RedisQueue
    RedisQueue --> Workers
    Workers --> VectorDB
```

- **`async_ingest_directory`**: Background document chunking, PyMuPDF extraction, and idempotent vector upserting.
- **`async_deep_screen_candidate`**: Parallel LLM candidate audits with rate-limited task batching.
- **Docker Compose Topology**: Multi-container setup orchestrating `app` (Streamlit port 8501), `redis` (port 6379), and `celery_worker`.

---

## 7. Production Observability & Evaluation

### A. Structured JSON Logging & Node Tracing
Every workflow execution is instrumented via structured JSON logging and the `@trace_node` decorator:

```json
{
  "timestamp": "2026-08-14 17:00:12,345",
  "level": "INFO",
  "logger": "agentic_profile_matching.node.rank_candidates",
  "message": "Completed node execution: rank_candidates in 1.24ms",
  "event": "node_end",
  "node": "rank_candidates",
  "elapsed_ms": 1.24,
  "thread_id": "session-4247cca1"
}
```

### B. Pluggable Observability Backends
- **Langfuse (`OBSERVABILITY_BACKEND=langfuse`)**: Captures complete execution traces, token usage, LLM input/output payloads, and latency waterfalls.
- **OpenTelemetry (`OBSERVABILITY_BACKEND=opentelemetry`)**: Emits standard OTLP spans for enterprise APM platforms (Datadog, Dynatrace, AWS CloudWatch).

### C. RAG Evaluation Benchmark Suite
Automated evaluation pipelines (`tests/eval/`) run against ground-truth scenarios (`data/eval_scenarios.json`) measuring:
1. **Retrieval Recall@K**: Proportion of ground-truth relevant profiles retrieved in the top $K$ candidates.
2. **Mean Reciprocal Rank (MRR)**: Precision of the top-ranked ground-truth profile position.
3. **Response Faithfulness**: Verification that LLM screening summaries strictly reflect extracted candidate resume text with zero hallucinated skills or experiences.

---

---

## 8. AI Safety, Guardrails & Production Resilience

1. **Grounded Recommendation Hierarchy**: Preserves qualitative LLM deep screening status assignments (`"Strong Hire"`, `"Borderline Hire"`, `"Rejected / No-Hire"`) while enforcing deterministic overrides if mandatory qualifications are unmet.
2. **Fact-Grounded Prompt Engineering**: Passes explicit candidate ground-truth metrics (`Experience Years`, `Matched Skills`, `Missing Skills`) into all report explanation prompts, forbidding ungrounded claims.
3. **Exponential Backoff & Rate Limit Handling**: All LLM invocations are wrapped in `execute_with_retry` catching HTTP 429 exceptions with exponential backoff (up to 5 retries).
4. **Token Truncation Budgeting**: Candidate resume inputs are capped at 12,000 characters (~3,000 tokens) with concise structured JSON output constraints, eliminating TPM/RPM exhaustion.
5. **Ingestion Idempotency**: Deterministic chunk IDs (`{filename}_chunk_{idx}`) prevent vector store bloat across repeated ingestion runs.

---

## 9. Stateless Credential Architecture & Checkpoint Isolation (ADR-011)

To comply with **CWE-312** standards and eliminate credential exfiltration in checkpoint snapshots or APM logs:
1. **Zero-Credential State**: `AgentState` contains strictly domain metadata (`requirements`, `shortlist`, `messages`). API keys are never stored in graph state.
2. **Runtime Configuration Injection**: Model instances are created at entry boundaries and supplied via `RunnableConfig` (`config["configurable"]["llm"]`) or an in-memory thread-keyed store.
3. **Automated Secret Redaction**: Log formatters apply regex token masking (`sk-.*`, `gsk_.*`, `AIzaSy.*`) across all console and trace channels.

---

## 10. Functional State Immutability & Idempotent Node Invariance (ADR-012)

LangGraph's state machine requires functional, side-effect-free node transitions:
1. **Immutable Candidate Dictionaries**: Candidate dicts in `AgentState.shortlist` are never mutated in-place.
2. **Copy-on-Write State Updates**: Screening nodes construct new profile records (`{**c, "strengths": ..., "gaps": ...}`) and return fresh list slices.
3. **Deterministic Checkpointing**: Guarantees that time-travel debugging, step restarts, and human-in-the-loop interrupts resume from clean, uncorrupted checkpoints.

---

## 11. Dynamic Generative LLM Skill Expansion & Semantic Equivalence Engine (ADR-013)

```mermaid
graph LR
    JobDesc["📄 Recruiter Job Description"] --> Extractor["LLM Requirements Extractor"]
    Extractor --> Expansions["🧠 Generative Skill Expansions<br/>('Cloud' ➔ ['AWS', 'GCP', 'Azure', 'K8s'])"]
    Resume["📄 Candidate Resume Text"] --> SectionParser["Open-Ended Regex Section Parser<br/>(Captures 'SKILLS:' & 'TECHNICAL SKILLS:')"]
    Expansions --> Matcher["JobMatcher Semantic Evaluator"]
    SectionParser --> Matcher
    Matcher --> Satisfied["✅ Candidate Evaluated with Conceptual Equivalence"]
```

1. **Generative Query Expansion**: Replaces brittle static manual YAML taxonomies with LLM-driven runtime expansion (`JobRequirements.skill_expansions`), dynamically associating parent skills with their ecosystem technologies.
2. **Semantic Equivalence Verification**: `JobMatcher._skill_matches_candidate()` ensures candidates with equivalent specialized tooling (e.g. AWS or GCP) satisfy general competencies (e.g. Cloud).
3. **Open-Ended Section Indexing**: Captures emerging tools and unlisted frameworks directly into candidate vector metadata via regex section parsing in `resume_rag.py`.

---

## 12. Concurrency-Controlled Asynchronous Candidate Screening (ADR-014)

```mermaid
graph TD
    Shortlist["Top 5 Shortlisted Profiles"] --> Dispatcher["Concurrent Screening Dispatcher"]
    Dispatcher --> Semaphore["🚦 Bounded Semaphore (max_concurrent=2)"]
    Semaphore --> Worker1["LLM Audit Worker 1"]
    Semaphore --> Worker2["LLM Audit Worker 2"]
    Worker1 --> Aggregator["State Aggregator (Copy-on-Write)"]
    Worker2 --> Aggregator
    Aggregator --> Output["Screened Candidate State"]
```

1. **Parallel Worker Pool**: Uses bounded `ThreadPoolExecutor` workers to audit multiple candidate profiles simultaneously.
2. **RPM/TPM Rate-Limit Shield**: Concurrency `Semaphore` restricts simultaneous inference calls to prevent HTTP 429 errors from Groq, Gemini, or OpenAI.
3. **Latency Gain**: Reduces Round 2 deep screening wall-clock time from **~75s to ~15–20s** while maintaining structured output validation.

---

## 13. Zero-Disk In-Memory Resume Ingestion Architecture (ADR-016)

```mermaid
graph LR
    Upload["📁 Recruiter Upload (.pdf, .docx, .txt)"] --> BytesBuffer["In-Memory Stream (io.BytesIO)"]
    BytesBuffer --> StreamParser["PyMuPDF / docx Buffer Parser<br/>(Zero Disk Writes)"]
    StreamParser --> ChunkEmbed["Text Chunking & Dense Embeddings"]
    ChunkEmbed --> StoreUpsert["Vector Store (BaseVectorStore)<br/>URI: stream://{filename}"]
```

1. **Complete Server-Side Disk Isolation**: Ingests files directly from byte streams without writing unencrypted documents to the server filesystem (`/tmp`).
2. **Ephemeral Cloud & Multi-Tenant Safety**: Designed for read-only containers (Streamlit Cloud, ECS, Lambda), completely eliminating file-leakage vulnerabilities (GDPR, SOC2).
3. **Deterministic Chunk Attribution**: Ingested candidates are indexed with `stream://{filename}` provenance, seamlessly searchable alongside pre-seeded repository profiles.

---

## 14. Environment Variables & Runtime Configuration Reference

Yojaka AI enforces twelve-factor application principles, managing external integrations, model credentials, background broker connections, and protocol flags via environment variables:

| Variable | Type | Default | Component Layer | Description & Security Rationale |
|:---|:---:|:---|:---|:---|
| `GROQ_API_KEY` | `Secret` | `""` | LLM Inference Gateway | Authentication for Groq Llama 3.3 70B inference. Stored statelessly; never serialized in graph state (ADR-011). |
| `GEMINI_API_KEY` | `Secret` | `""` | LLM Inference Gateway | Authentication for Google Gemini 2.0 Pro / Flash. Masked in APM logs. |
| `SARVAM_API_KEY` | `Secret` | `""` | Indic Language Models | Authentication for Sarvam AI 105B Indic LLM endpoints (ADR-010). |
| `OPENAI_API_KEY` | `Secret` | `""` | LLM Inference Gateway | Authentication for OpenAI GPT-4o / GPT-4o-mini models. |
| `TAVILY_API_KEY` | `Secret` | `""` | External Web Search | Real-time web search for live tech trends, external packages, and candidate portfolios. Optional; engine degrades to local mock notes if omitted. |
| `USE_MCP` | `bool` | `False` | Dual Tool Gateway | Toggles FastMCP stdio JSON-RPC 2.0 servers (`True`) vs direct in-process execution (`False`) (ADR-001). |
| `MCP_TIMEOUT` | `float` | `30.0` | Dual Tool Gateway | Subprocess JSON-RPC communication timeout in seconds before fallback triggers. |
| `OBSERVABILITY_BACKEND` | `str` | `"none"` | APM & Telemetry | Selects APM backend: `"none"`, `"langfuse"`, or `"opentelemetry"` (ADR-007). |
| `LANGFUSE_PUBLIC_KEY` | `Secret` | `""` | APM & Telemetry | Public tracking key for Langfuse prompt/run analytics. |
| `LANGFUSE_SECRET_KEY` | `Secret` | `""` | APM & Telemetry | Secret tracking key for Langfuse authenticated ingestion. |
| `LANGFUSE_HOST` | `str` | `"https://cloud.langfuse.com"` | APM & Telemetry | Self-hosted or cloud URL for Langfuse APM server. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `str` | `""` | APM & Telemetry | gRPC or HTTP collector endpoint for OpenTelemetry span export. |
| `REDIS_URL` | `URL` | `"redis://localhost:6379/0"` | Task Queue & Caching | Redis 7 broker for Celery async worker queue (ADR-006). |
| `CELERY_BROKER_URL` | `URL` | `REDIS_URL` | Task Queue Broker | Celery task message broker connection string. |
| `CELERY_RESULT_BACKEND` | `URL` | `REDIS_URL` | Task Queue Backend | Celery asynchronous task result storage backend. |
| `EMBEDDING_MODEL` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | RAG Vector Storage | Dense sentence transformer model used for ChromaDB vector embeddings. |
| `VECTOR_DB_PATH` | `Path` | `"./chroma_db"` | RAG Vector Storage | Filesystem path for persistent baseline ChromaDB vector storage. |
| `TOP_K` | `int` | `10` | Matching Engine | Maximum candidates retrieved during vector coarse filtering. |
| `RESUME_TRUNCATION_LIMIT` | `int` | `12000` | Safety Guardrails | Maximum resume character count per candidate (~3,000 tokens) passed to deep screening LLM prompts. |
| `THROTTLE_DELAY` | `float` | `0.5` | Safety Guardrails | Rate-limiting delay (seconds) between sequential LLM inference calls to prevent HTTP 429 quota exhaustion. |


