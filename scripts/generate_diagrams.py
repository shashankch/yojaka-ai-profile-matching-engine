#!/usr/bin/env python3
"""Yojaka AI: Architecture & Dataflow Diagrams Generator.

Uses Diagrams (diagrams as code - https://diagrams.mingrammer.com/) to generate
high-resolution PNG architecture and workflow diagrams for public documentation
and the MkDocs site.

Every diagram strictly adheres to the real system implementation (Phases 1-17),
uses official high-resolution brand logos, has clean non-colliding edges, and maintains
balanced, publication-grade aspect ratios.
"""

from pathlib import Path
import shutil
from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python
from diagrams.onprem.client import User, Client, Users
from diagrams.onprem.database import Qdrant
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.monitoring import Datadog
from diagrams.onprem.security import Vault, Trivy
from diagrams.generic.network import Firewall
from diagrams.generic.storage import Storage
from diagrams.aws.ml import Textract, Kendra

# Output directory for rendered diagram assets
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
OUTPUT_DIR = DOCS_DIR / "assets" / "diagrams"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Latest official brand icons directory
ICONS_DIR = DOCS_DIR / "assets" / "icons"

ICON_STREAMLIT = str(ICONS_DIR / "streamlit.png")
ICON_LANGGRAPH = str(ICONS_DIR / "langgraph.png")
ICON_CHROMA = str(ICONS_DIR / "chroma.png")
ICON_GROQ = str(ICONS_DIR / "groq.png")
ICON_HF = str(ICONS_DIR / "huggingface.png")
ICON_ANTHROPIC = str(ICONS_DIR / "anthropic.png")
ICON_CELERY = str(ICONS_DIR / "celery.png")
ICON_GEMINI = str(ICONS_DIR / "gemini.png")
ICON_OPENAI = str(ICONS_DIR / "openai.png")

# Publication-grade graph styling
BASE_GRAPH_ATTR = {
    "dpi": "300",
    "bgcolor": "#ffffff",
    "pad": "0.5",
    "nodesep": "0.8",
    "ranksep": "0.9",
    "fontname": "Helvetica-Bold",
    "fontsize": "15",
    "fontcolor": "#0f172a",
    "compound": "true",
    "splines": "spline",
}

BASE_NODE_ATTR = {
    "fontname": "Helvetica",
    "fontsize": "10",
    "fontcolor": "#0f172a",
    "labelloc": "b",
}

BASE_EDGE_ATTR = {
    "fontname": "Helvetica-Bold",
    "fontsize": "9",
    "fontcolor": "#334155",
    "color": "#475569",
    "penwidth": "1.3",
}


def cluster_attr(
    bgcolor: str,
    border_color: str,
    labelloc: str = "t",
    margin: str | None = None,
) -> dict[str, str]:
    """Returns cluster styling with generous padding to prevent text-border collision."""
    if margin is None:
        margin = "35" if labelloc == "b" else "28"
    attr = {
        "bgcolor": bgcolor,
        "color": border_color,
        "style": "rounded",
        "margin": margin,
    }
    if labelloc == "b":
        attr["labelloc"] = "b"
    return attr


def generate_system_architecture():
    """Generates clean, planar 2D system architecture overview in LR orientation."""
    output_path = OUTPUT_DIR / "system_architecture"
    with Diagram(
        "Yojaka AI: System Architecture Overview",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "ranksep": "1.2", "nodesep": "0.8"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Presentation & Ingress (ADR-016)",
            graph_attr=cluster_attr("#f8fafc", "#94a3b8"),
        ):
            recruiter = User("Recruiter\n(Browser UI)")
            api_client = Client("Headless Client\n(REST / SSE)")
            st_ui = Custom("Streamlit App\n(Port 8501)", ICON_STREAMLIT)
            api_gw = Fastapi("FastAPI Gateway\n(Port 8000)")
            recruiter >> Edge(label="HTTPS", color="#2563eb") >> st_ui
            api_client >> Edge(label="HTTP REST", color="#2563eb") >> api_gw

        with Cluster(
            "Core Agentic Orchestration",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            conductor = Custom("StateGraph Engine\n(LangGraph)", ICON_LANGGRAPH)
            router = Python("Intent Router\n(Fast-Path Cache)")
            screening = Python("Cascading Screening\n(Funnel Orchestrator)")
            conductor >> router >> screening

        with Cluster(
            "Storage & Worker Infrastructure",
            graph_attr=cluster_attr("#faf5ff", "#d8b4fe", labelloc="b"),
        ):
            vector_store = Custom("ChromaDB Index\n(In-Memory RAM)", ICON_CHROMA)
            qdrant_store = Qdrant("Qdrant Vector DB\n(Persistent)")
            celery_workers = Custom("Celery Worker Pool\n(Async Tasks)", ICON_CELERY)
            redis_broker = Redis("Redis 7 Broker\n(Port 6379 FIFO)")
            apm = Datadog("Langfuse / APM\n(OTel Tracing)")
            redis_broker >> celery_workers
            celery_workers >> qdrant_store
            celery_workers >> vector_store

        with Cluster(
            "Hybrid Retrieval & Deep Screening",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            hybrid_search = Kendra("Coarse Hybrid Search\n(Dense + BM25Okapi)")
            cross_encoder = Custom("Cross-Encoder Rerank\n(ms-marco-MiniLM)", ICON_HF)
            deep_llm = Custom("Deep LLM Audit\n(Llama 3.3 / GPT-4o / Gemini 3.8)", ICON_GROQ)
            report = Custom("Comparison Matrix\n(Decision Synthesis)", ICON_STREAMLIT)
            hybrid_search >> cross_encoder >> deep_llm >> report

        st_ui >> Edge(label="execute", color="#2563eb") >> conductor
        api_gw >> Edge(label="stream", color="#2563eb") >> conductor
        screening >> Edge(label="Search Query", color="#059669") >> hybrid_search
        screening >> Edge(label="apply_async", color="#7c3aed") >> redis_broker
        conductor >> Edge(label="Telemetry", style="dashed", color="#0891b2") >> apm
        vector_store >> Edge(label="Index Read", style="dashed", color="#059669") >> hybrid_search


def generate_cascading_screening_funnel():
    """Generates 3-stage cascading screening funnel [O(N) -> O(K)]."""
    output_path = OUTPUT_DIR / "cascading_screening_funnel"
    with Diagram(
        "3-Stage Cascading Screening Funnel [O(N) -> O(K)]",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "nodesep": "0.9", "ranksep": "1.3"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        raw_corpus = Storage("Candidate Resumes\n(N Profiles)")

        with Cluster(
            "Stage 1: Coarse Filtering (0 LLM Cost, O(N))",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            s1_hybrid = Kendra("Dense + BM25\nHybrid Index")
            s1_rerank = Custom("Cross-Encoder Rerank\n(ms-marco-MiniLM)", ICON_HF)
            s1_hybrid >> s1_rerank

        with Cluster(
            "Stage 2: Deep Profile Analysis (~3k Tokens/Profile, O(K))",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            s2_guard = Python("Semaphore(2) Barrier\n(Bounded Concurrency)")
            s2_groq = Custom("Primary: Groq LPU\n(Llama 3.3 70B, ~300 T/s)", ICON_GROQ)
            s2_gemini = Custom("Fallback: Google Gemini\n(Gemini 3.8 Flash)", ICON_GEMINI)
            s2_guard >> Edge(label="Slot 1 & 2 Audits", color="#2563eb") >> s2_groq
            s2_groq >> Edge(label="429 Fallback", style="dashed", color="#ea580c") >> s2_gemini

        with Cluster(
            "Stage 3: Decision & Synthesis (1 LLM Call)",
            graph_attr=cluster_attr("#faf5ff", "#d8b4fe"),
        ):
            s3_safety = Trivy("Demographic Parity\n& Safety Guardrails")
            s3_decide = Custom("Recommendation & QGen\n(OpenAI GPT-4o / Gemini 3.8)", ICON_OPENAI)
            s3_safety >> s3_decide

        matrix = Custom("Interactive Matrix\n& Recruiter Report", ICON_STREAMLIT)

        raw_corpus >> Edge(label="Candidate Pool", color="#059669") >> s1_hybrid
        s1_rerank >> Edge(label="Top 10 Slice", color="#2563eb") >> s2_guard
        s2_groq >> Edge(label="Audited Top 5", color="#7c3aed") >> s3_safety
        s2_gemini >> Edge(color="#7c3aed", style="dashed") >> s3_safety
        s3_decide >> Edge(label="Render", color="#2563eb") >> matrix


def generate_c4_level1_system_context():
    """Generates C4 Level 1 - System Context diagram in clean top-to-bottom layout."""
    output_path = OUTPUT_DIR / "c4_level1_system_context"
    with Diagram(
        "C4 Level 1: System Context Diagram",
        filename=str(output_path),
        show=False,
        direction="TB",
        graph_attr={**BASE_GRAPH_ATTR, "ranksep": "0.75", "nodesep": "0.7"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Stakeholders & Consumers",
            graph_attr=cluster_attr("#f8fafc", "#94a3b8"),
        ):
            recruiter = User("Recruiter / Hiring Mgr\n(Browser UI)")
            ats_client = Client("Enterprise ATS / HRIS\n(REST API Client)")

        engine = Custom(
            "Yojaka AI Platform\n(Agentic Matching Engine)",
            ICON_LANGGRAPH,
            width="1.9",
            height="1.8",
            fixedsize="true",
        )

        with Cluster(
            "External Ecosystem & Pluggable LLMs (Provider-Agnostic)",
            graph_attr=cluster_attr("#ffffff", "#cbd5e1", labelloc="b"),
        ):
            groq = Custom("Groq Cloud LPU\n(Llama 3.3 70B)", ICON_GROQ)
            gemini = Custom("Google Gemini\n(Gemini 3.8 Flash / 3.8 Pro)", ICON_GEMINI)
            openai = Custom("OpenAI API\n(GPT-4o / o3-mini)", ICON_OPENAI)
            anthropic = Custom("Anthropic Claude\n(Claude 3.7 Sonnet)", ICON_ANTHROPIC)
            apm = Datadog("Langfuse / APM\n(OTel Tracing)")
            corpus = Storage("Resume Storage\n(Zero-Disk Stream)")

        recruiter >> Edge(label="HTTPS / UI", color="#2563eb") >> engine
        ats_client >> Edge(label="REST / SSE", color="#2563eb") >> engine

        engine >> Edge(label="Fast Inference", color="#ea580c") >> groq
        engine >> Edge(label="Multimodal & Fallback", color="#2563eb") >> gemini
        engine >> Edge(label="Deep Reasoning", color="#10b981") >> openai
        engine >> Edge(label="Contextual RAG", color="#d97706") >> anthropic
        engine >> Edge(label="Telemetry", style="dashed", color="#0891b2") >> apm
        engine >> Edge(label="Stream Read", color="#059669") >> corpus


def generate_c4_level2_container():
    """Generates C4 Level 2 - Container Diagram in clean 5-cluster left-to-right flow."""
    output_path = OUTPUT_DIR / "c4_level2_container"
    with Diagram(
        "C4 Level 2: Container Diagram",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "ranksep": "1.3", "nodesep": "0.9"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Presentation & Ingress (ADR-016)",
            graph_attr=cluster_attr("#f8fafc", "#94a3b8"),
        ):
            recruiter = User("Recruiter\n(Browser UI)")
            client_app = Client("API Client\n(REST / SSE)")
            st_ui = Custom("Streamlit UI\n(Port 8501)", ICON_STREAMLIT)
            api_gw = Fastapi("FastAPI Gateway\n(Port 8000)")
            recruiter >> Edge(label="HTTPS 8501", color="#2563eb") >> st_ui
            client_app >> Edge(label="REST 8000", color="#2563eb") >> api_gw

        with Cluster(
            "Agentic Core & Orchestration",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            agent = Custom("LangGraph Engine\n(StateGraph Core)", ICON_LANGGRAPH)
            gateway = Python("Dual-Mode Gateway\n(FastMCP / Direct)")
            agent >> Edge(label="Tool Calls", color="#059669") >> gateway

        with Cluster(
            "Cloud Inference & Observability",
            graph_attr=cluster_attr("#fff7ed", "#fdba74"),
        ):
            llm_groq = Custom("Groq Cloud LPU\n(Llama 3.3 70B Primary)", ICON_GROQ)
            llm_gemini = Custom("Google Gemini\n(Gemini 3.8 Flash Fallback)", ICON_GEMINI)
            apm_ext = Datadog("APM & Tracing\n(Langfuse)")
            llm_groq >> Edge(label="Fallback", style="dashed", color="#ea580c") >> llm_gemini

        with Cluster(
            "Distributed Queue & Execution",
            graph_attr=cluster_attr("#faf5ff", "#d8b4fe", labelloc="b"),
        ):
            redis = Redis("Redis 7 Broker\n(Port 6379 FIFO)")
            celery = Custom("Celery Workers\n(Async Pool)", ICON_CELERY)
            redis >> Edge(label="Task Feeds", color="#7c3aed") >> celery

        with Cluster(
            "Vector Storage Tier",
            graph_attr=cluster_attr("#f0fdf4", "#86efac", labelloc="b"),
        ):
            chroma_store = Custom("ChromaDB Vector Store\n(In-Memory RAM)", ICON_CHROMA)
            qdrant_store = Qdrant("Qdrant Vector DB\n(Persistent)")

        st_ui >> Edge(label="execute", color="#2563eb") >> agent
        api_gw >> Edge(label="stream", color="#2563eb") >> agent
        agent >> Edge(label="Inference", color="#ea580c") >> llm_groq
        agent >> Edge(label="Telemetry", style="dashed", color="#0891b2") >> apm_ext
        agent >> Edge(label="apply_async", color="#7c3aed") >> redis

        celery >> Edge(color="#059669") >> chroma_store
        celery >> Edge(color="#059669") >> qdrant_store
        gateway >> Edge(label="Search Resumes", color="#059669") >> chroma_store


def generate_c4_level3_component_core():
    """Generates C4 Level 3 - Component Diagram: LangGraph Agentic Core."""
    output_path = OUTPUT_DIR / "c4_level3_component_core"
    with Diagram(
        "C4 Level 3: LangGraph Agentic Core Component Diagram",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "ranksep": "1.1", "nodesep": "0.8"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Tier 1: Input & Intent Routing (ADR-009)",
            graph_attr=cluster_attr("#fff7ed", "#fdba74"),
        ):
            parse_in = Python("parse_input_node\n(Sanitize Payload)")
            fast_path = Python("Fast-Path Cache\n(0ms Exact Match)")
            router = Python("Intent Router\n(Classifier)")
            parse_in >> fast_path >> router

        with Cluster(
            "Tier 2: Requirements & Dialogue",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            extract_req = Textract("extract_requirements\n(LLM Extraction)")
            adjust_req = Python("adjust_requirements\n(Criterion Adjustment)")
            conv_query = Python("conversational_query\n(Direct QA)")

        with Cluster(
            "Tier 3: Retrieval & Re-ranking",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            search_res = Kendra("search_resumes\n(Dense + BM25Okapi)")
            rank_cand = Custom("rank_candidates\n(HF Cross-Encoder)", ICON_HF)
            search_res >> rank_cand

        with Cluster(
            "Tier 4: Deep Screening & Synthesis",
            graph_attr=cluster_attr("#faf5ff", "#d8b4fe"),
        ):
            deep_screen = Custom("deep_screen\n(Groq Llama 3.3 Audit)", ICON_GROQ)
            recommend = Custom("recommendation\n(GPT-4o / Gemini 3.8)", ICON_OPENAI)
            gen_report = Custom("generate_report\n(Recruiter Matrix)", ICON_STREAMLIT)
            deep_screen >> recommend >> gen_report

        state_store = Custom("AgentState Checkpoint\n(Session Memory)", ICON_LANGGRAPH)

        # Routing connections
        router >> Edge(label="new_search", color="#2563eb") >> extract_req
        router >> Edge(label="refine", color="#eab308") >> adjust_req
        router >> Edge(label="chat", color="#0891b2") >> conv_query

        # Connected evaluation paths (NO dead ends)
        extract_req >> Edge(color="#059669") >> search_res
        adjust_req >> Edge(label="Refined", color="#059669") >> search_res
        rank_cand >> Edge(label="Top 10", color="#2563eb") >> deep_screen

        gen_report >> Edge(label="Done", color="#7c3aed") >> state_store
        conv_query >> Edge(label="Answer", color="#0891b2") >> state_store


def generate_execution_lifecycle():
    """Generates end-to-end execution lifecycle diagram."""
    output_path = OUTPUT_DIR / "execution_lifecycle"
    with Diagram(
        "End-to-End Execution Lifecycle & Dataflow",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "nodesep": "0.8", "ranksep": "1.1"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "1. Ingestion Phase (Zero-Disk)",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            stream_buf = Storage("BytesIO Stream\n(Zero Disk I/O)")
            parser = Textract("PyMuPDF Parser\n(Bounding Boxes)")
            chroma = Custom("ChromaDB RAM Index", ICON_CHROMA)
            stream_buf >> parser >> chroma

        with Cluster(
            "2. Query & Intent Phase",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            recruiter = User("Recruiter")
            router = Python("Intent Router\n(Fast-Path)")
            extract = Textract("Requirements Extractor")
            recruiter >> router >> extract

        with Cluster(
            "3. Retrieval & Rerank Phase",
            graph_attr=cluster_attr("#fff7ed", "#fdba74"),
        ):
            search = Kendra("Hybrid Search\n(Dense + BM25)")
            rerank = Custom("Cross-Encoder Rerank\n(ms-marco-MiniLM)", ICON_HF)
            search >> rerank

        with Cluster(
            "4. Audit & Synthesis Phase",
            graph_attr=cluster_attr("#faf5ff", "#d8b4fe"),
        ):
            deep = Custom("LLM Deep Audit\n(Groq Llama 3.3 70B)", ICON_GROQ)
            decide = Custom("Recommendation &\nInterview QGen (GPT-4o)", ICON_OPENAI)
            deep >> decide

        matrix = Custom("Comparison Matrix\n(Streamlit UI)", ICON_STREAMLIT)

        # Cross-Cluster Pipeline Flow
        chroma >> Edge(label="Index Query", color="#059669") >> search
        extract >> Edge(label="JD Query", color="#2563eb") >> search
        rerank >> Edge(label="Top 10 Slice", color="#2563eb") >> deep
        decide >> Edge(label="Render", color="#2563eb") >> matrix


def generate_state_graph_topology():
    """Generates LangGraph StateGraph topology diagram with balanced layout."""
    output_path = OUTPUT_DIR / "state_graph_topology"
    with Diagram(
        "LangGraph State Machine Topology & Intent Routing (ADR-009)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "ranksep": "1.1", "nodesep": "0.8"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        start = User("User Request / JD")
        parse_input = Python("parse_input_node")
        router = Python("Intent Router\n(Cache + Fast-Path)")

        with Cluster(
            "Routing Branches",
            graph_attr=cluster_attr("#f8fafc", "#cbd5e1"),
        ):
            extract_req = Textract("extract_requirements\n(New Search)")
            adjust_req = Python("adjust_requirements\n(Refine Criteria)")
            conv_query = Python("conversational_query\n(Direct QA)")

        with Cluster(
            "Candidate Evaluation Pipeline",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            search_resumes = Kendra("search_resumes\n(Dense + BM25Okapi)")
            rank_candidates = Custom("rank_candidates\n(Cross-Encoder)", ICON_HF)
            deep_screen = Custom("deep_screen\n(Groq Llama 3.3 Audit)", ICON_GROQ)
            recommend = Custom("recommendation\n(GPT-4o / Gemini 3.8)", ICON_OPENAI)
            gen_report = Custom("generate_report\n(Recruiter Matrix)", ICON_STREAMLIT)
            search_resumes >> rank_candidates >> deep_screen >> recommend >> gen_report

        end = Client("Recruiter Response / UI")

        start >> parse_input >> router
        router >> Edge(label="new_search", color="#2563eb") >> extract_req
        router >> Edge(label="refine_search", color="#eab308") >> adjust_req
        router >> Edge(label="chat_query", color="#0891b2") >> conv_query

        # Fully connected graph without dead ends
        extract_req >> Edge(color="#059669") >> search_resumes
        adjust_req >> Edge(label="Updated Criteria", color="#059669") >> search_resumes
        gen_report >> Edge(label="Structured Report", color="#2563eb") >> end
        conv_query >> Edge(label="Chat Reply", color="#0891b2") >> end


def generate_hybrid_search_scoring():
    """Generates hybrid search and multi-factor scoring pipeline diagram without text overlap."""
    output_path = OUTPUT_DIR / "hybrid_search_scoring"
    with Diagram(
        "Hybrid Search & Multi-Factor Scoring Engine (0-100 Score)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "nodesep": "1.0", "ranksep": "1.4"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Job Description Query Ingress",
            graph_attr=cluster_attr("#f8fafc", "#94a3b8"),
        ):
            jd_input = Textract("Parsed Job Description\n(Schema & Requirements)")

        with Cluster(
            "Stage 1: Multi-Factor Hybrid Scorers (Pre-Indexed Corpus)",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            dense = Custom("Dense Semantic (50%)\nChromaDB (all-MiniLM)", ICON_CHROMA)
            bm25 = Kendra("Sparse Lexical (35%)\nBM25Okapi Term Index")
            quals = Python("Hard Constraints (15%)\nRule & Metadata Filter")

        with Cluster(
            "Stage 2: Score Fusion & Cross-Encoder Calibration",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            fusion = Python("Score Fusion Engine\n(Weighted 0-100 Composite)")
            rerank = Custom("Cross-Encoder Reranker\n(ms-marco-MiniLM)", ICON_HF)
            fusion >> Edge(label="Top 25 Candidates", color="#2563eb") >> rerank

        output = Custom("Calibrated Top 10 Shortlist\n(LangGraph State)", ICON_LANGGRAPH)

        jd_input >> Edge(label="Query Vector", color="#059669") >> dense
        jd_input >> Edge(label="Query Terms", color="#059669") >> bm25
        jd_input >> Edge(label="Hard Constraints", color="#059669") >> quals

        dense >> Edge(label="50%", color="#2563eb") >> fusion
        bm25 >> Edge(label="35%", color="#2563eb") >> fusion
        quals >> Edge(label="15%", color="#2563eb") >> fusion

        rerank >> Edge(label="Calibrated Shortlist", color="#2563eb") >> output


def generate_mcp_dual_gateway():
    """Generates clean MCP dual-mode gateway architecture diagram (ADR-001)."""
    output_path = OUTPUT_DIR / "mcp_dual_gateway"
    with Diagram(
        "Dual-Mode Tool Gateway Architecture (ADR-001: FastMCP vs Direct)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "nodesep": "1.0", "ranksep": "1.3"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        nodes = Custom("LangGraph Nodes\n(agent/nodes.py)", ICON_LANGGRAPH)
        gateway = Python("Dual-Mode Gateway\n(USE_MCP Toggle)")

        with Cluster(
            "Direct Mode (USE_MCP=False: In-Process Tools)",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            in_proc_fs = Storage("fs_tools.py\n(Direct Memory/Disk Read)")
            in_proc_match = Kendra("job_matcher.py\n(In-Process Scoring)")

        with Cluster(
            "FastMCP Mode (USE_MCP=True: Microservice Subprocesses)",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            mcp_client = Python("FastMCP Client Wrapper\n(fastmcp_client.py)")
            transport = Python("Subprocess stdio\n(JSON-RPC 2.0 IPC)")
            search_server = Kendra("search_server\n(Tool: search_resumes)")
            fs_server = Storage("filesystem_server\n(Tool: read_resume)")
            mcp_client >> Edge(label="stdio", color="#2563eb") >> transport
            transport >> Edge(color="#2563eb") >> search_server
            transport >> Edge(color="#2563eb") >> fs_server

        nodes >> Edge(label="Invoke Tool", color="#059669") >> gateway
        gateway >> Edge(label="In-Proc Read", color="#16a34a") >> in_proc_fs
        gateway >> Edge(label="In-Proc Match", color="#16a34a") >> in_proc_match
        gateway >> Edge(label="Remote RPC", color="#2563eb") >> mcp_client


def generate_celery_redis_task_queue():
    """Generates Celery and Redis distributed task queue diagram with official modern Celery icon."""
    output_path = OUTPUT_DIR / "celery_redis_task_queue"
    with Diagram(
        "Distributed Task Queue Architecture (ADR-006: Celery + Redis 7)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "nodesep": "1.0", "ranksep": "1.3"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Task Producers",
            graph_attr=cluster_attr("#f8fafc", "#94a3b8"),
        ):
            web_ui = Custom("Streamlit Web App\n(Port 8501)", ICON_STREAMLIT)
            fastapi_app = Fastapi("FastAPI Gateway\n(Port 8000)")

        broker = Redis("Redis 7 Task Broker\n(Port 6379, FIFO Queues)")

        with Cluster(
            "Celery Worker Pool",
            graph_attr=cluster_attr("#faf5ff", "#d8b4fe"),
        ):
            w1 = Custom("Worker 1: Bulk Ingest\n(async_ingest_directory)", ICON_CELERY)
            w2 = Custom("Worker 2: Deep Audit\n(async_deep_screen)", ICON_CELERY)

        with Cluster(
            "Persistence & Vector Indices",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            chroma_store = Custom("ChromaDB Vector Store\n(In-Memory)", ICON_CHROMA)
            qdrant_store = Qdrant("Qdrant Vector DB\n(Persistent)")

        web_ui >> Edge(label="apply_async", color="#7c3aed") >> broker
        fastapi_app >> Edge(label="apply_async", color="#7c3aed") >> broker
        broker >> Edge(label="Queue: ingest", color="#7c3aed") >> w1
        broker >> Edge(label="Queue: screen", color="#7c3aed") >> w2
        w1 >> Edge(label="Index Embeddings", color="#059669") >> chroma_store
        w2 >> Edge(label="Persist State", color="#059669") >> qdrant_store


def generate_generative_skill_expansion():
    """Generates generative LLM skill expansion diagram."""
    output_path = OUTPUT_DIR / "generative_skill_expansion"
    with Diagram(
        "Dynamic Generative Skill Expansion & Semantic Equivalence (ADR-013)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "nodesep": "1.0", "ranksep": "1.3"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Job Description Pipeline",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            jd = Textract("Job Description Text")
            extractor = Python("Skill Extractor\n(Structured Pydantic)")
            expansion = Custom("Generative Expansion\n(Cloud -> AWS, GCP, K8s)", ICON_GROQ)
            jd >> extractor >> expansion

        with Cluster(
            "Candidate Resume Pipeline",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            resume = Storage("Candidate Resume\n(PDF / DOCX)")
            parser = Textract("Layout Section Parser\n(PyMuPDF Blocks)")
            parsed_skills = Python("Captured Profile Skills\n[AWS, EKS, Terraform]")
            resume >> parser >> parsed_skills

        matcher = Kendra("Semantic Equivalence Matcher\n(_skill_matches_candidate)")
        result = Custom("Verified Skill Match Matrix\n(Equivalent Tech Identified)", ICON_LANGGRAPH)

        expansion >> Edge(label="Expanded Tech", color="#2563eb") >> matcher
        parsed_skills >> Edge(label="Candidate Tech", color="#059669") >> matcher
        matcher >> Edge(label="Validated Score", color="#2563eb") >> result


def generate_concurrency_controlled_screening():
    """Generates concurrency-controlled asynchronous screening diagram (ADR-014)."""
    output_path = OUTPUT_DIR / "concurrency_controlled_screening"
    with Diagram(
        "Concurrency-Controlled Asynchronous Screening (ADR-014)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "nodesep": "1.0", "ranksep": "1.3"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        candidates = Users("Top Shortlisted\nCandidates (Top 5)")
        dispatcher = Python("Concurrent Dispatcher\n(ThreadPoolExecutor)")
        barrier = Python("Semaphore(2) Barrier\n(Rate Limit Guard)")

        with Cluster(
            "Bounded Parallel Audit Workers (Max 2 Concurrency)",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            worker1 = Python("Worker Thread 1\n(Candidate A Audit)")
            worker2 = Python("Worker Thread 2\n(Candidate B Audit)")

        with Cluster(
            "LLM Deep Audit Inference Engine",
            graph_attr=cluster_attr("#faf5ff", "#d8b4fe"),
        ):
            primary_llm = Custom("Primary LLM: Groq LPU\n(Llama 3.3 70B, ~300 T/s)", ICON_GROQ)
            fallback_llm = Custom("Fallback: Google Gemini\n(Gemini 3.8 Flash)", ICON_GEMINI)
            primary_llm >> Edge(label="Fallback (429 / Error)", style="dashed", color="#ea580c") >> fallback_llm

        aggregator = Python("Copy-on-Write Aggregator\n(State Immutability Guard)")
        output = Custom("Screened Candidate State\n(LangGraph AgentState)", ICON_LANGGRAPH)

        candidates >> dispatcher >> barrier
        barrier >> Edge(label="Slot 1", color="#2563eb") >> worker1
        barrier >> Edge(label="Slot 2", color="#2563eb") >> worker2

        worker1 >> Edge(label="Audit Request A", color="#7c3aed") >> primary_llm
        worker2 >> Edge(label="Audit Request B", color="#7c3aed") >> primary_llm

        primary_llm >> Edge(label="Audit Results", color="#059669") >> aggregator
        aggregator >> Edge(label="Immutable State", color="#2563eb") >> output


def generate_zero_disk_in_memory_ingestion():
    """Generates zero-disk in-memory ingestion architecture diagram (ADR-016)."""
    output_path = OUTPUT_DIR / "zero_disk_in_memory_ingestion"
    with Diagram(
        "Zero-Disk In-Memory Resume Ingestion Architecture (ADR-016)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "nodesep": "1.0", "ranksep": "1.3"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Stage 1: Ephemeral Stream Ingestion",
            graph_attr=cluster_attr("#f8fafc", "#94a3b8"),
        ):
            upload = User("Recruiter Upload\n(.pdf, .docx, .txt)")
            ram_stream = Storage("io.BytesIO Buffer\n(Zero Disk Writes, RAM)")
            parser = Textract("PyMuPDF Buffer Parser\n(fitz.open(stream=...))")
            upload >> Edge(label="Multipart POST", color="#2563eb") >> ram_stream
            ram_stream >> Edge(label="Stream Read", color="#059669") >> parser

        with Cluster(
            "Stage 2: Contextual Enrichment & Embedding",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            enricher = Custom("Contextual Enricher\n(Claude 3.7 Sonnet)", ICON_ANTHROPIC)
            embed = Custom("Tensor Encoder\n(all-MiniLM-L6-v2)", ICON_HF)
            store = Custom("CompositeVectorStore\n(Ephemeral RAM Index)", ICON_CHROMA)
            enricher >> Edge(color="#059669") >> embed
            embed >> Edge(label="RAM Vectors", color="#059669") >> store

        parser >> Edge(label="Section Blocks", color="#059669") >> enricher


def generate_enterprise_security_guardrails():
    """Generates enterprise security and blind hiring strategy diagram."""
    output_path = OUTPUT_DIR / "enterprise_security_guardrails"
    with Diagram(
        "Enterprise Security, Blind Hiring & Guardrail Strategy (2026)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "nodesep": "1.0", "ranksep": "1.3"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "1. Ingestion & Security Guardrails",
            graph_attr=cluster_attr("#fef2f2", "#fca5a5"),
        ):
            untrusted = Storage("Untrusted Resume\n(Hostile PDF / DOCX)")
            firewall = Firewall("Prompt Injection Sanitizer\n(OWASP LLM01 Defense)")
            pii_vault = Vault("Reversible PII Vault\n(Names -> [CANDIDATE_A])")
            untrusted >> firewall >> pii_vault

        with Cluster(
            "2. Blind Evaluation & Fairness Audit",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            blind_screen = Custom("Blind Screening Pipeline\n(LangGraph State Machine)", ICON_LANGGRAPH)
            audit = Trivy("Demographic Parity Audit\n(AIR >= 0.8 / 4/5ths Rule)")
            blind_screen >> audit

        with Cluster(
            "3. Authorized Delivery",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            deanon = Vault("UI Rehydration Vault\n(Authorized Recruiter View)")
            report = Custom("Grounded Recruiter Report\n(Streamlit / API Matrix)", ICON_STREAMLIT)
            deanon >> report

        pii_vault >> Edge(label="Anonymized Payload", color="#dc2626") >> blind_screen
        audit >> Edge(label="Fairness Cleared", color="#16a34a") >> deanon


def generate_dual_surface_ui_api():
    """Generates dual-surface UI and FastAPI architecture diagram with balanced LR proportions."""
    output_path = OUTPUT_DIR / "dual_surface_ui_api"
    with Diagram(
        "Dual-Surface Architecture: Modular UI & Headless FastAPI Gateway (Phase 16)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "ranksep": "1.3", "nodesep": "0.9"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Clients & Integrators",
            graph_attr=cluster_attr("#f8fafc", "#94a3b8"),
        ):
            recruiter = User("Recruiter Browser\n(Interactive GUI)")
            client_app = Client("External API / CI\n(Headless Automation)")

        with Cluster(
            "Modular Presentation Surfaces (Phase 16)",
            graph_attr=cluster_attr("#fff7ed", "#fdba74"),
        ):
            with Cluster("Streamlit UI (ui/)", graph_attr=cluster_attr("#f8fafc", "#e2e8f0", margin="20")):
                st_app = Custom("Streamlit App (<75 LOC)\n(ui/streamlit_app.py)", ICON_STREAMLIT)
                st_runner = Python("ui/runner.py\n(Streaming Execution)")
                st_app >> st_runner

            with Cluster("FastAPI Gateway (api/)", graph_attr=cluster_attr("#f8fafc", "#e2e8f0", margin="20")):
                api_app = Fastapi("FastAPI Gateway\n(api/main.py)")
                api_runner = Python("Async Pipeline Runner\n(Background Task)")
                api_app >> api_runner

        with Cluster(
            "Shared Engine & Storage Core",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            workflow = Custom("LangGraph Engine\n(agent/workflow.py)", ICON_LANGGRAPH)
            store = Custom("CompositeVectorStore\n(Chroma + Qdrant)", ICON_CHROMA)
            workflow >> store

        recruiter >> Edge(label="HTTPS 8501", color="#ea580c") >> st_app
        client_app >> Edge(label="REST / SSE 8000", color="#2563eb") >> api_app
        st_runner >> Edge(label="In-Process Invoke", color="#ea580c") >> workflow
        api_runner >> Edge(label="Async Invoke", color="#2563eb") >> workflow


def generate_layout_parsing_two_stage_rerank():
    """Generates layout-aware parsing and two-stage reranking diagram with clean 2-stage flow."""
    output_path = OUTPUT_DIR / "layout_parsing_two_stage_rerank"
    with Diagram(
        "Layout-Aware Parsing, Contextual Retrieval & Two-Stage Reranking (Phase 17)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "ranksep": "1.3", "nodesep": "0.8"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "Stage 1: Layout-Aware Ingestion & Contextual Enrichment",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            doc = Storage("Resume PDF\n(Multi-Column)")
            parser = Textract("PyMuPDF Section Parser\n(Bounding Blocks)")
            enricher = Custom("Contextual Enricher\n(Claude 3.7 Sonnet)", ICON_ANTHROPIC)
            index = Custom("ChromaDB Hybrid Store\n(Vector + BM25)", ICON_CHROMA)
            doc >> parser >> enricher >> index

        with Cluster(
            "Stage 2: Two-Stage Hybrid Retrieval & Cross-Encoder Reranking",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd", labelloc="b"),
        ):
            query = Textract("Job Description Query\n(Extracted Schema)")
            stage1 = Kendra("Coarse Hybrid Search\n(Dense + BM25Okapi)")
            stage2 = Custom("Cross-Encoder Rerank\n(ms-marco-MiniLM)", ICON_HF)
            top_ranked = Custom("Calibrated Shortlist\n(Deep Screen Input)", ICON_LANGGRAPH)
            query >> stage1 >> stage2 >> top_ranked

        index >> Edge(label="Candidate Chunks", color="#059669") >> stage1


def generate_calibrated_margin_subgraphs():
    """Generates Calibrated Margin Routing & Dual-Rubric Subgraphs architecture diagram (Phase 18)."""
    output_path = OUTPUT_DIR / "calibrated_margin_subgraphs"
    with Diagram(
        "Calibrated Margin Routing, Native Commands & Dual-Rubric Subgraphs (Phase 18 / ADR-018)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "ranksep": "1.3", "nodesep": "0.8"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "1. Calibrated Margin Ingress & Routing",
            graph_attr=cluster_attr("#f8fafc", "#cbd5e1"),
        ):
            user = User("Recruiter Ingress\n(Raw Prompt / JD)")
            router = Python("Margin Router\n(Top1 - Top2 >= 0.12)")
            fast_path = Custom("Fast-Path Command\n(Direct <2ms · $0 Cost)", ICON_LANGGRAPH)
            llm_gate = Custom("Structured LLM Gate\n(Ambiguity Fallback)", ICON_GROQ)
            user >> router
            router >> Edge(label="Margin >= 0.12", color="#059669") >> fast_path
            router >> Edge(label="Margin < 0.12", color="#d97706") >> llm_gate

        with Cluster(
            "2. Modular Subgraphs Pipeline (Dual-Rubric Screening)",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            jd_subgraph = Textract("JD Analyzer Subgraph\n(Requirements & Skills)")
            retrieval = Kendra("Talent Retrieval Subgraph\n(Hybrid + Cross-Encoder)")
            rubric_a = Custom("Rubric A: Technical (60%)\nArchitecture & Systems", ICON_GROQ)
            rubric_b = Custom("Rubric B: Domain (40%)\nTrajectory & Culture", ICON_GEMINI)
            aggregator = Python("Deterministic Aggregator\n(0.6A + 0.4B)")

            fast_path >> Edge(color="#2563eb") >> jd_subgraph
            llm_gate >> Edge(color="#2563eb") >> jd_subgraph
            jd_subgraph >> Edge(label="Extracted Schema", color="#2563eb") >> retrieval
            retrieval >> Edge(label="Candidate Chunks", color="#2563eb") >> rubric_a
            retrieval >> Edge(label="Candidate Chunks", color="#2563eb") >> rubric_b
            rubric_a >> Edge(color="#7c3aed") >> aggregator
            rubric_b >> Edge(color="#7c3aed") >> aggregator

        with Cluster(
            "3. Synthesis & Recruiter Presentation",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            synthesis = Custom("Synthesis Subgraph\n(QGen, Safety & Matrix)", ICON_OPENAI)
            out = Custom("Recruiter Dashboard\n(Streamlit & FastAPI)", ICON_STREAMLIT)

            aggregator >> Edge(label="Composite Shortlist", color="#059669") >> synthesis
            synthesis >> Edge(label="Final Report & Scores", color="#059669") >> out


def generate_hyde_parent_doc_rag():
    """Generates Advanced RAG Architecture: HyDE Query Synthesis, Parent-Document Chunking & Local Inference (Phase 19 / ADR-019)."""
    output_path = OUTPUT_DIR / "hyde_parent_doc_rag"
    with Diagram(
        "Advanced RAG Architecture, Parent-Doc Chunking & Air-Gapped Local Inference (Phase 19 / ADR-019)",
        filename=str(output_path),
        show=False,
        direction="LR",
        graph_attr={**BASE_GRAPH_ATTR, "ranksep": "1.3", "nodesep": "0.8"},
        node_attr=BASE_NODE_ATTR,
        edge_attr=BASE_EDGE_ATTR,
    ):
        with Cluster(
            "1. Ingestion & Hierarchical Parent-Child Decomposition",
            graph_attr=cluster_attr("#f8fafc", "#cbd5e1"),
        ):
            doc_in = Storage("Raw Resumes / JDs\n(PDF / DOCX / TXT)")
            parent_store = Storage("ParentDocumentStore\n(1000-1500 chars parent blocks)")
            child_split = Python("ParentDocumentService\n(150-250 token child chunks)")
            chroma_child = Custom("Vector Collection\n(ChromaDB Child Embeddings)", ICON_CHROMA)
            bm25_index = Python("Sparse BM25 Index\n(BM25Okapi Exact Terms)")

            doc_in >> parent_store
            doc_in >> child_split
            child_split >> Edge(label="Parent Pointer UUID", color="#475569") >> parent_store
            child_split >> Edge(label="Index Embeddings", color="#2563eb") >> chroma_child
            child_split >> Edge(label="Index Tokens", color="#475569") >> bm25_index

        with Cluster(
            "2. Query Processing & Pre-Retrieval Filtering",
            graph_attr=cluster_attr("#eff6ff", "#93c5fd"),
        ):
            recruiter = User("Recruiter Query / JD\n(Prompt + Constraints)")
            metadata_filter = Python("FacetedFilter\n(Exp, Edu & Must-Haves)")
            hyde_gen = Custom("HyDEService\n(Synthesized Profile Bio)", ICON_GROQ)
            dense_embed = Custom("Query Embedder\n(SentenceTransformer)", ICON_HF)

            recruiter >> Edge(label="Criteria Metadata", color="#2563eb") >> metadata_filter
            recruiter >> Edge(label="Query Text", color="#2563eb") >> hyde_gen
            hyde_gen >> Edge(label="Hypothetical Candidate", color="#7c3aed") >> dense_embed
            recruiter >> Edge(label="Exact Terms", color="#d97706") >> bm25_index

        with Cluster(
            "3. Two-Stage Rerank & Air-Gapped Local Screening",
            graph_attr=cluster_attr("#f0fdf4", "#86efac"),
        ):
            hybrid_merge = Kendra("Hybrid Score Merge\n(0.5 Dense + 0.5 Sparse)")
            reranker = Custom("Cross-Encoder Rerank\n(bge-reranker-base)", ICON_HF)
            local_llm = Python("Local Daemon / Air-Gapped\n(Ollama / vLLM / Local)")
            deep_screen = Custom("Dual-Rubric Deep Screen\n(Tech 60% + Domain 40%)", ICON_LANGGRAPH)
            dashboard = Custom("Recruiter Dashboard\n(Streamlit & FastAPI)", ICON_STREAMLIT)

            dense_embed >> Edge(color="#2563eb") >> chroma_child
            chroma_child >> Edge(color="#2563eb") >> hybrid_merge
            bm25_index >> Edge(color="#2563eb") >> hybrid_merge
            metadata_filter >> Edge(label="Candidate Filter Mask", color="#059669") >> hybrid_merge
            hybrid_merge >> Edge(label="Top-K Candidates", color="#2563eb") >> reranker
            reranker >> Edge(label="Enrich via Parent UUID", color="#059669") >> parent_store
            parent_store >> Edge(label="Full Section Blocks", color="#059669") >> deep_screen
            local_llm >> Edge(label="Zero-Cloud Tokens", color="#10b981") >> deep_screen
            deep_screen >> Edge(label="Verified Shortlist & Matrix", color="#059669") >> dashboard


def preserve_state_machine_assets():
    """Copies the canonical state_machine assets without regenerating them."""
    canonical_png = DOCS_DIR / "state_machine.png"
    if canonical_png.exists():
        shutil.copy2(canonical_png, OUTPUT_DIR / "state_machine.png")
        print("  -> Preserved canonical docs/state_machine.png intact.")


def main():
    """Generates all 18 architecture and dataflow diagrams."""
    print("🎨 Generating Yojaka AI Architecture & Dataflow Diagrams with Diagrams (diagrams as code)...")

    diagram_generators = [
        ("System Architecture Overview", generate_system_architecture),
        ("3-Stage Cascading Funnel", generate_cascading_screening_funnel),
        ("C4 Level 1 - System Context", generate_c4_level1_system_context),
        ("C4 Level 2 - Container Diagram", generate_c4_level2_container),
        ("C4 Level 3 - Component Core", generate_c4_level3_component_core),
        ("Execution Lifecycle", generate_execution_lifecycle),
        ("StateGraph Topology", generate_state_graph_topology),
        ("Hybrid Search & Scoring", generate_hybrid_search_scoring),
        ("MCP Dual-Mode Gateway", generate_mcp_dual_gateway),
        ("Celery + Redis Task Queue", generate_celery_redis_task_queue),
        ("Dynamic Skill Expansion", generate_generative_skill_expansion),
        ("Concurrency-Controlled Screening", generate_concurrency_controlled_screening),
        ("Zero-Disk In-Memory Ingestion", generate_zero_disk_in_memory_ingestion),
        ("Enterprise Security Guardrails", generate_enterprise_security_guardrails),
        ("Dual-Surface UI & API", generate_dual_surface_ui_api),
        ("Layout Parsing & Two-Stage Rerank", generate_layout_parsing_two_stage_rerank),
        ("Calibrated Margin & Subgraphs", generate_calibrated_margin_subgraphs),
        ("HyDE & Parent-Doc Advanced RAG", generate_hyde_parent_doc_rag),
    ]

    for name, gen_fn in diagram_generators:
        print(f"  -> Rendering {name}...")
        gen_fn()

    preserve_state_machine_assets()
    print("✅ All 17 architecture diagrams generated successfully in docs/assets/diagrams/")
    print("ℹ️ Note: docs/state_machine.png and docs/state_machine.mermaid are strictly preserved intact from LangGraph.")


if __name__ == "__main__":
    main()
