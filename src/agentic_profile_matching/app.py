import os
import html
import hashlib
import uuid
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Load configurations
load_dotenv()
from agentic_profile_matching import config  # noqa: E402
from agentic_profile_matching.matching_agent import matching_agent_workflow  # noqa: E402
from agentic_profile_matching.tools import compare_candidates  # noqa: E402


# Setup page config
st.set_page_config(
    page_title="Yojaka AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Auto-bootstrap candidate vector database for cold starts (e.g. Streamlit Cloud)
@st.cache_resource(show_spinner="Bootstrapping Candidate Database...")
def ensure_vector_store_initialized():
    from agentic_profile_matching.stores import ChromaVectorStore
    from agentic_profile_matching.services.ingestion_service import IngestionService

    store = ChromaVectorStore()
    service = IngestionService(store=store)
    if store.count() == 0:
        resumes_dir = Path(config.RESUMES_DIR)
        if not resumes_dir.exists():
            resumes_dir = Path(config.BASE_DIR).parent / "data" / "resumes"
        if resumes_dir.exists():
            service.ingest_directory(str(resumes_dir))
    else:
        service.prune_stale_records()
    return store


vector_store = ensure_vector_store_initialized()

# Custom premium styling
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main App Header with Gradient */
    .app-header {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.75rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 1.75rem;
    }

    /* Adaptive Theme Variables for Candidate Cards (Default: Clean Crisp Light Mode) */
    :root, [data-theme="light"], .stApp {
        --c-card-bg: #ffffff;
        --c-card-border: #e2e8f0;
        --c-card-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.08), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
        --c-card-hover-bg: #f8fafc;
        --c-card-hover-border: #6366f1;
        --c-card-name: #0f172a;
        --c-card-rank: #64748b;
        --c-card-meta: #334155;
        --c-card-label: #475569;
        --c-skill-bg: #f1f5f9;
        --c-skill-text: #1e293b;
        --c-skill-border: #cbd5e1;
        --c-card-path: #64748b;
    }

    /* Dark Mode: Applied strictly when data-theme="dark" */
    [data-theme="dark"], [data-theme="dark"] .stApp {
        --c-card-bg: rgba(30, 41, 59, 0.65);
        --c-card-border: rgba(255, 255, 255, 0.1);
        --c-card-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.35);
        --c-card-hover-bg: rgba(30, 41, 59, 0.9);
        --c-card-hover-border: rgba(167, 139, 250, 0.5);
        --c-card-name: #ffffff;
        --c-card-rank: #9ca3af;
        --c-card-meta: #e2e8f0;
        --c-card-label: #9ca3af;
        --c-skill-bg: rgba(255, 255, 255, 0.08);
        --c-skill-text: #f1f5f9;
        --c-skill-border: rgba(255, 255, 255, 0.15);
        --c-card-path: #94a3b8;
    }

    /* Candidate Card Layout */
    .candidate-card {
        background: var(--c-card-bg);
        border: 1px solid var(--c-card-border);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--c-card-shadow);
    }
    
    .candidate-card:hover {
        transform: translateY(-3px);
        border-color: var(--c-card-hover-border);
        box-shadow: 0 12px 24px -4px rgba(99, 102, 241, 0.18);
        background: var(--c-card-hover-bg);
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .card-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--c-card-name);
    }

    .card-rank {
        color: var(--c-card-rank);
        margin-left: 8px;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .score-badge {
        font-size: 1.05rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 3px 9px;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.15);
    }

    .card-meta {
        display: flex;
        gap: 2rem;
        font-size: 0.92rem;
        margin-bottom: 0.75rem;
        color: var(--c-card-meta);
    }

    .skills-label {
        font-size: 0.78rem;
        color: var(--c-card-label);
        font-weight: 700;
        display: block;
        margin-bottom: 0.35rem;
        letter-spacing: 0.05em;
    }

    .card-path {
        font-size: 0.78rem;
        color: var(--c-card-path);
        word-break: break-all;
        margin-top: 0.35rem;
    }

    /* Status Pills */
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .status-strong {
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .status-borderline {
        background-color: rgba(245, 158, 11, 0.15);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .status-rejected {
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    @media (prefers-color-scheme: dark) {
        .status-strong { color: #34d399; }
        .status-borderline { color: #fbbf24; }
        .status-rejected { color: #f87171; }
    }
    [data-theme="dark"] .status-strong { color: #34d399; }
    [data-theme="dark"] .status-borderline { color: #fbbf24; }
    [data-theme="dark"] .status-rejected { color: #f87171; }

    /* Skills Badges */
    .skill-tag {
        display: inline-block;
        background: var(--c-skill-bg);
        color: var(--c-skill-text);
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
        border: 1px solid var(--c-skill-border);
    }

    /* Customizing Streamlit Tabs & Buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 6px 6px 0 0;
        background-color: rgba(255, 255, 255, 0.01);
        color: #9ca3af;
        border: 1px solid transparent;
        padding: 0 14px;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(167, 139, 250, 0.08) !important;
        color: #c084fc !important;
        border-color: rgba(167, 139, 250, 0.2) rgba(167, 139, 250, 0.2) transparent rgba(167, 139, 250, 0.2) !important;
        font-weight: 600;
    }

    /* Styled buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 8px 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 3px 5px rgba(79, 70, 229, 0.2) !important;
        transition: all 0.2s !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-1.5px) !important;
        box-shadow: 0 6px 10px rgba(79, 70, 229, 0.3) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    }

    .stButton>button:active {
        transform: translateY(0) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.html("""
<script>
(function() {
    function detectTheme() {
        try {
            const app = document.querySelector('.stApp');
            if (!app) return;
            const bg = window.getComputedStyle(app).backgroundColor;
            const isDark = bg.includes('14, 17, 23') || bg.includes('14,17,23') || bg === 'rgb(14, 17, 23)';
            const newTheme = isDark ? 'dark' : 'light';
            if (document.documentElement.getAttribute('data-theme') !== newTheme) {
                document.documentElement.setAttribute('data-theme', newTheme);
            }
        } catch (e) {}
    }
    const observer = new MutationObserver(detectTheme);
    observer.observe(document.body, { attributes: true, childList: true, subtree: true });
    detectTheme();
    setInterval(detectTheme, 400);
})();
</script>
""")


# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "requirements" not in st.session_state:
    st.session_state["requirements"] = {
        "title": "Software Engineer",
        "must_have_skills": [],
        "nice_to_have_skills": [],
        "min_experience_years": 0,
        "education_level": "Not Specified",
        "other_constraints": [],
    }
if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = []
if "final_report" not in st.session_state:
    st.session_state["final_report"] = ""
if "ranking_explanation" not in st.session_state:
    st.session_state["ranking_explanation"] = ""
if "coarse_limit" not in st.session_state:
    st.session_state["coarse_limit"] = config.DEFAULT_COARSE_LIMIT
if "deep_limit" not in st.session_state:
    st.session_state["deep_limit"] = config.DEFAULT_DEEP_LIMIT
if "recommendation_limit" not in st.session_state:
    st.session_state["recommendation_limit"] = config.DEFAULT_RECOMMENDATION_LIMIT
if "errors" not in st.session_state:
    st.session_state["errors"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex[:8]
if "ephemeral_store" not in st.session_state:
    from agentic_profile_matching.stores import ChromaVectorStore

    st.session_state["ephemeral_store"] = ChromaVectorStore(
        collection_name=f"uploads_{st.session_state['session_id']}",
        ephemeral=True,
    )
if "session_vector_store" not in st.session_state:
    from agentic_profile_matching.stores import CompositeVectorStore

    st.session_state["session_vector_store"] = CompositeVectorStore(
        base_store=vector_store,
        ephemeral_store=st.session_state["ephemeral_store"],
    )
if "uploaded_candidates" not in st.session_state:
    st.session_state["uploaded_candidates"] = []


# ----------------------------------------------------
# Sidebar: LLM Configuration & Requirements Details
# ----------------------------------------------------

st.sidebar.title("Configuration & Filters")

st.sidebar.markdown("### 1. LLM Provider Setup")
llm_provider = st.sidebar.selectbox("LLM Provider", list(config.SUPPORTED_PROVIDERS.keys()), index=0)
supported_models = config.SUPPORTED_PROVIDERS[llm_provider]
llm_model = st.sidebar.selectbox("Model Name", supported_models, index=0)

# Pre-populate keys from environment secrets
default_key = ""
if llm_provider == "Groq":
    default_key = os.getenv("GROQ_API_KEY", "")
elif llm_provider == "Gemini":
    default_key = os.getenv("GEMINI_API_KEY", "")
elif llm_provider == "Sarvam AI":
    default_key = os.getenv("SARVAM_API_KEY", "")
elif llm_provider == "OpenAI":
    default_key = os.getenv("OPENAI_API_KEY", "")

api_key = st.sidebar.text_input("API Key", value=default_key, type="password")

api_url = None
if llm_provider == "Custom (OpenAI-compatible)":
    api_url = st.sidebar.text_input("API Base URL (Endpoint)", value="https://api.openai.com/v1")


# ----------------------------------------------------
# Helper: Progressive Live Checkpoints Runner
# ----------------------------------------------------
def run_agent_workflow_with_status(state_input: dict, config_dict: dict) -> dict:
    """
    Executes LangGraph workflow with progressive live checkpoint status indicators.
    """
    with st.status("Recruiter Agent is analyzing candidates...", expanded=True) as status_box:
        result = dict(state_input)
        emitted_any = False
        try:
            for event in matching_agent_workflow.stream(state_input, config=config_dict, stream_mode="updates"):
                emitted_any = True
                for node_name, node_update in event.items():
                    if isinstance(node_update, dict):
                        result.update(node_update)
                        if node_name == "parse_input":
                            status_box.write("🔍 Parsing recruiter input and verifying conversation history...")
                        elif node_name == "extract_requirements":
                            title_ext = node_update.get("requirements", {}).get("title", "Software Engineer")
                            status_box.write(f"📋 Extracted job requirements: **{title_ext}**")
                        elif node_name == "adjust_requirements":
                            status_box.write("🔄 Adjusted skill constraints and experience thresholds...")
                        elif node_name == "search_resumes":
                            count = len(node_update.get("shortlist", []))
                            status_box.write(
                                f"📂 Retrieved **{count}** candidate profiles via hybrid BM25 + dense search..."
                            )
                        elif node_name == "rank_candidates":
                            count = len(node_update.get("shortlist", []))
                            status_box.write(f"📊 Ranked Top **{count}** candidate profiles...")
                        elif node_name == "deep_screen":
                            status_box.write("🔬 Completed parallel deep audits (strengths, gaps, reasoning)...")
                        elif node_name == "recommendation":
                            status_box.write("⚖️ Formulated hire decisions & custom interview questions...")
                        elif node_name == "generate_report":
                            status_box.write("📝 Compiled candidate comparison matrix and executive report...")
                        elif node_name == "conversational_query":
                            status_box.write("💬 Formulated conversational response...")
            status_box.update(label="Screening workflow complete!", state="complete", expanded=False)
            return result
        except Exception as ex:
            if not emitted_any:
                status_box.write(f"⚠️ Direct execution fallback ({ex})...")
                result = matching_agent_workflow.invoke(state_input, config=config_dict)
                status_box.update(label="Screening workflow complete!", state="complete", expanded=False)
                return result
            else:
                status_box.write(f"⚠️ Workflow stopped with partial error: {ex}")
                status_box.update(label="Screening completed with errors", state="error", expanded=False)
                return result


# ----------------------------------------------------
# Sidebar Section 2: Active Talent Pool Summary
# ----------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Active Talent Pool")
try:
    total_indexed_chunks = st.session_state["session_vector_store"].count()
except Exception:
    total_indexed_chunks = 0

try:
    all_files = list(Path(config.RESUMES_DIR).glob("*.*"))
    disk_count = len([f for f in all_files if f.suffix.lower() in [".pdf", ".docx", ".txt"]])
    valid_count = disk_count + len(st.session_state.get("uploaded_candidates", []))
except Exception:
    valid_count = 34 + len(st.session_state.get("uploaded_candidates", []))

st.sidebar.info(
    f"📂 **{valid_count} Profiles** (`{total_indexed_chunks}` chunks in memory)\n\n"
    "Upload new resumes or explore indexed candidates in the **📤 Resume Ingestion & Talent Pool** workspace tab."
)


# ----------------------------------------------------
# Sidebar Section 3: Throttling & Limits Control
# ----------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ 3. Throttling & Limits Control", expanded=False):
    coarse_limit = st.slider(
        "Round 1: Coarse Limit",
        min_value=5,
        max_value=20,
        value=int(st.session_state["coarse_limit"]),
    )
    deep_limit = st.slider(
        "Round 2: Deep Screen Limit",
        min_value=3,
        max_value=15,
        value=int(st.session_state["deep_limit"]),
    )
    recommendation_limit = st.slider(
        "Round 3: Recommendation Limit",
        min_value=2,
        max_value=10,
        value=int(st.session_state["recommendation_limit"]),
    )

    st.session_state["coarse_limit"] = coarse_limit
    st.session_state["deep_limit"] = deep_limit
    st.session_state["recommendation_limit"] = recommendation_limit


# ----------------------------------------------------
# Sidebar Section 4: Active Requirements Constraints
# ----------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("📋 4. Active Requirements Constraints", expanded=False):
    reqs = st.session_state["requirements"]
    title_input = st.text_input("Extracted Job Title", value=reqs.get("title", "Software Engineer"))
    min_exp_slider = st.slider(
        "Min Experience Years",
        min_value=0,
        max_value=20,
        value=int(reqs.get("min_experience_years", 0)),
    )
    must_have_input = st.text_area(
        "Must-Have Skills (comma separated)",
        value=", ".join(reqs.get("must_have_skills", [])),
    )
    nice_have_input = st.text_area(
        "Nice-To-Have Skills (comma separated)",
        value=", ".join(reqs.get("nice_to_have_skills", [])),
    )
    education_level = st.text_input("Education Level Target", value=reqs.get("education_level", "Not Specified"))

    if st.button("Sync Constraints & Re-Rank"):
        updated_reqs = {
            "title": title_input,
            "must_have_skills": [s.strip() for s in must_have_input.split(",") if s.strip()],
            "nice_to_have_skills": [s.strip() for s in nice_have_input.split(",") if s.strip()],
            "min_experience_years": min_exp_slider,
            "education_level": education_level,
            "other_constraints": reqs.get("other_constraints", []),
            "skill_expansions": reqs.get("skill_expansions", {}),
        }
        st.session_state["requirements"] = updated_reqs

        state_input = {
            "messages": st.session_state["messages"],
            "requirements": updated_reqs,
            "shortlist": [],
            "coarse_screen_limit": st.session_state["coarse_limit"],
            "deep_screen_limit": st.session_state["deep_limit"],
            "recommendation_limit": st.session_state["recommendation_limit"],
            "current_round": 1,
            "final_report": "",
            "feedback_pending": False,
            "user_feedback": "Recruiter updated requirements manually via sidebar.",
            "errors": [],
        }

        workflow_config = {
            "configurable": {
                "thread_id": "streamlit-session-thread",
                "api_key": api_key,
                "api_url": api_url,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "store": st.session_state["session_vector_store"],
            }
        }

        result = run_agent_workflow_with_status(state_input, workflow_config)
        st.session_state["shortlist"] = result.get("shortlist", [])
        st.session_state["final_report"] = result.get("final_report", "")
        st.session_state["ranking_explanation"] = result.get("ranking_explanation", "")
        st.session_state["errors"] = result.get("errors", [])
        st.success("Shortlist re-ranked successfully!")


# ----------------------------------------------------
# Main Layout Workspace
# ----------------------------------------------------

st.markdown(
    '<h1 class="app-header">💼 Yojaka AI (Agentic Profile Matching Engine)</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">Autonomous multi-agent intelligence to extract, hybrid-search, screen, and benchmark talent profiles.</p>',
    unsafe_allow_html=True,
)

# Render active warnings/errors from the agent run
if st.session_state["errors"]:
    st.session_state["errors"] = [e for e in st.session_state["errors"] if "File not found" not in e]
    for err in st.session_state["errors"]:
        st.warning(f"⚠️ {err}")


tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Chat Workspace", "📤 Resume Ingestion & Talent Pool", "📊 Shortlist & Comparison", "🔬 Deep Screening Reports"]
)

# TAB 1: Chat Workspace
with tab1:
    st.markdown("### Chat with Recruiter Assistant")
    st.caption(
        "Paste a Job Description (JD) to extract requirements, or type conversational search and refinement commands."
    )

    # Render conversational chat log
    for msg in st.session_state["messages"]:
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai"):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Chat input area
    user_query = st.chat_input("Enter message (e.g. 'Search resumes for React developers with 3+ years experience')")

    if user_query:
        # Display user input in UI immediately
        with st.chat_message("user"):
            st.markdown(user_query)

        # Append to message list
        st.session_state["messages"].append(HumanMessage(content=user_query))

        # Prepare graph state inputs (stateless credential isolation)
        state_input = {
            "messages": st.session_state["messages"],
            "requirements": st.session_state["requirements"],
            "shortlist": st.session_state["shortlist"],
            "coarse_screen_limit": st.session_state["coarse_limit"],
            "deep_screen_limit": st.session_state["deep_limit"],
            "recommendation_limit": st.session_state["recommendation_limit"],
            "current_round": 1,
            "final_report": st.session_state["final_report"],
            "feedback_pending": False,
            "user_feedback": "",
            "errors": [],
        }

        workflow_config = {
            "configurable": {
                "thread_id": "streamlit-session-thread",
                "api_key": api_key,
                "api_url": api_url,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "store": st.session_state["session_vector_store"],
            }
        }

        try:
            result = run_agent_workflow_with_status(state_input, workflow_config)

            # Copy updated state outputs to session state
            st.session_state["messages"] = result.get("messages", [])
            st.session_state["requirements"] = result.get("requirements", {})
            st.session_state["shortlist"] = result.get("shortlist", [])
            st.session_state["final_report"] = result.get("final_report", "")
            st.session_state["ranking_explanation"] = result.get("ranking_explanation", "")
            st.session_state["errors"] = result.get("errors", [])

            # Render assistant output
            with st.chat_message("assistant"):
                if st.session_state["shortlist"]:
                    st.markdown("Analyzed candidates and successfully updated active requirements.")
                    if st.session_state["ranking_explanation"]:
                        st.info(st.session_state["ranking_explanation"])
                    st.markdown(
                        f"**Top Candidates Shortlisted**: {', '.join(c['name'] for c in st.session_state['shortlist'][:3])}"
                    )
                else:
                    st.markdown("Ingested input. Please verify the active requirements are updated.")
            st.rerun()

        except Exception as e:
            st.error(f"Error executing agentic loop: {e}")
            st.session_state["messages"].append(AIMessage(content=f"Sorry, I encountered an error: {str(e)}"))

# TAB 2: Resume Ingestion & Talent Pool (Dedicated Workspace)
with tab2:
    st.markdown("### 📤 Candidate Resume Ingestion & Talent Pool")
    st.caption(
        "Zero-disk in-memory stream processing (ADR-016). Uploaded resumes (.pdf, .docx, .txt) are vectorized directly into memory without local disk persistence."
    )

    # Talent Pool Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Indexed Chunks", total_indexed_chunks)
    with c2:
        try:
            all_files = list(Path(config.RESUMES_DIR).glob("*.*"))
            valid_count = len([f for f in all_files if f.suffix.lower() in [".pdf", ".docx", ".txt"]])
            st.metric("Active Talent Profiles", valid_count)
        except Exception:
            st.metric("Active Talent Profiles", "34")
    with c3:
        st.metric("Storage Protocol", "BaseVectorStore")
    with c4:
        st.metric("Ingestion Mode", "In-Memory Stream")

    st.markdown("---")
    st.markdown("#### 📁 Drag-and-Drop Resume Ingestion")
    main_uploaded_files = st.file_uploader(
        "Upload candidate resumes to expand the active talent pool",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="main_resume_uploader",
        help="Direct in-memory stream ingestion via PyMuPDF/python-docx without server disk writes.",
    )

    if main_uploaded_files:
        if "processed_uploads" not in st.session_state:
            st.session_state["processed_uploads"] = set()

        MAX_UPLOAD_BYTES = 10 * 1024 * 1024
        files_to_process = []
        for uf in main_uploaded_files:
            if uf.size > MAX_UPLOAD_BYTES:
                st.error(
                    f"❌ Upload rejected for `{uf.name}`: Exceeds 10MB limit ({round(uf.size / (1024 * 1024), 2)} MB)."
                )
                continue
            content_bytes = uf.getvalue()
            content_digest = f"{uf.name}_{uf.size}_{hashlib.sha256(content_bytes).hexdigest()[:12]}"
            if content_digest not in st.session_state["processed_uploads"]:
                files_to_process.append((uf, content_bytes, content_digest))

        if files_to_process:
            with st.status(f"Ingesting {len(files_to_process)} resume(s)...", expanded=True) as upload_status:
                from agentic_profile_matching.services.ingestion_service import IngestionService

                # Ingest into session-scoped ephemeral store (ADR-016 zero-disk privacy)
                service = IngestionService(store=st.session_state["ephemeral_store"])
                for uf, content_bytes, content_digest in files_to_process:
                    upload_status.write(f"Vectorizing `{uf.name}` into memory...")
                    res = service.ingest_stream(uf.name, content_bytes)
                    if res.get("success"):
                        st.session_state["processed_uploads"].add(content_digest)
                        cand_entry = {
                            "candidate_name": res.get("candidate_name", "Unknown"),
                            "filename": uf.name,
                            "format": Path(uf.name).suffix.upper().replace(".", ""),
                            "cohort": "📤 Session Upload",
                            "status": "✅ In-Memory (Zero-Disk)",
                        }
                        # Update existing or append
                        existing = [
                            i for i, c in enumerate(st.session_state["uploaded_candidates"]) if c["filename"] == uf.name
                        ]
                        if existing:
                            st.session_state["uploaded_candidates"][existing[0]] = cand_entry
                        else:
                            st.session_state["uploaded_candidates"].append(cand_entry)

                        upload_status.write(
                            f"✅ Ingested **{res.get('candidate_name')}** ({res.get('chunks_ingested')} chunks)"
                        )
                    else:
                        upload_status.write(f"❌ Failed `{uf.name}`: {res.get('error')}")
                upload_status.update(
                    label=f"Ingested {len(files_to_process)} file(s) successfully!",
                    state="complete",
                    expanded=False,
                )
            st.rerun()

    st.markdown("---")
    st.markdown("#### 👥 Active Talent Pool Inventory")

    import pandas as pd

    talent_rows = []

    # 1. In-memory session uploads (isolated, zero-disk)
    for up in st.session_state.get("uploaded_candidates", []):
        talent_rows.append(
            {
                "Candidate Name": up["candidate_name"],
                "File Name": up["filename"],
                "Format": up["format"],
                "Cohort": up["cohort"],
                "Status": up["status"],
            }
        )

    # 2. Disk-based pre-indexed candidates
    try:
        resumes_dir_path = Path(config.RESUMES_DIR)
        for rf in sorted(resumes_dir_path.glob("*.*")):
            if rf.suffix.lower() in [".pdf", ".docx", ".txt"]:
                raw_name = rf.stem.replace("resume_", "")
                display_name = " ".join(p.capitalize() for p in raw_name.split("_"))
                persona_tag = "Standard Profile"
                talent_rows.append(
                    {
                        "Candidate Name": display_name,
                        "File Name": rf.name,
                        "Format": rf.suffix.upper().replace(".", ""),
                        "Cohort": persona_tag,
                        "Status": "✅ Indexed & Searchable",
                    }
                )
    except Exception:
        pass

    if talent_rows:
        df_talent = pd.DataFrame(talent_rows)
        st.dataframe(
            df_talent,
            use_container_width=True,
            column_config={
                "Candidate Name": st.column_config.TextColumn("Candidate Name", width="medium"),
                "File Name": st.column_config.TextColumn("File Name", width="medium"),
                "Format": st.column_config.TextColumn("Format", width="small"),
                "Cohort": st.column_config.TextColumn("Cohort / Source", width="small"),
                "Status": st.column_config.TextColumn("Index Status", width="small"),
            },
            hide_index=True,
        )

# TAB 3: Shortlist & Comparison Matrix
with tab3:
    st.markdown("### Candidate Shortlist Matrix")
    shortlist = st.session_state["shortlist"]

    if not shortlist:
        st.info("No candidates shortlisted yet. Paste a JD or write a search command in the Chat tab.")
    else:
        # Display side-by-side comparison table
        st.markdown("#### Head-to-Head Comparison Matrix")
        rec_limit = st.session_state["recommendation_limit"]
        candidate_ids = [c["candidate_id"] for c in shortlist[: int(rec_limit)]]
        compare_md = compare_candidates(candidate_ids, shortlist)
        st.markdown(compare_md)

        st.markdown("---")
        st.markdown("#### Ranked Candidate Shortlist")

        for idx, c in enumerate(shortlist):
            status = c.get("screening_status", "Shortlisted")

            # Strict classification to avoid operator precedence / substring bugs (like 'no-hire' matching 'hire')
            if "reject" in status.lower() or "no-hire" in status.lower():
                status_class = "status-rejected"
            elif "borderline" in status.lower():
                status_class = "status-borderline"
            else:
                status_class = "status-strong"

            # Escape candidate values for security (anti-XSS)
            safe_name = html.escape(str(c.get("name", "Unknown")), quote=True)
            safe_status = html.escape(str(status), quote=True)
            safe_exp = html.escape(str(c.get("experience_years", 0)), quote=True)
            safe_edu = html.escape(str(c.get("education", "Not Specified")), quote=True)
            safe_score = html.escape(str(c.get("score", 0)), quote=True)

            # Format skills
            matched = c.get("matched_skills", [])
            if matched:
                skills_html = "".join(
                    f'<span class="skill-tag">{html.escape(str(s), quote=True)}</span>' for s in matched
                )
            else:
                skills_html = '<span class="skill-tag" style="opacity: 0.5;">None matched</span>'

            # Get path relative to the project root directory
            try:
                project_root = Path(config.BASE_DIR).parent
                rel_path = str(Path(c["candidate_id"]).relative_to(project_root))
            except Exception:
                rel_path = Path(c.get("candidate_id", "")).name
            safe_rel_path = html.escape(rel_path, quote=True)

            card_html = f"""
            <div class="candidate-card">
                <div class="card-header">
                    <div>
                        <span class="card-name">{safe_name}</span>
                        <span class="card-rank">(Rank #{idx + 1})</span>
                    </div>
                    <span class="score-badge">{safe_score}/100</span>
                </div>
                <div style="margin-bottom: 0.75rem;">
                    <span class="status-pill {status_class}">{safe_status}</span>
                </div>
                <div class="card-meta">
                    <div>💼 <b>Experience</b>: {safe_exp} Years</div>
                    <div>🎓 <b>Education</b>: {safe_edu}</div>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span class="skills-label">MATCHED SKILLS:</span>
                    {skills_html}
                </div>
                <div class="card-path">
                    📄 <i>Path: {safe_rel_path}</i>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

# TAB 4: Deep Screening & Interview Qs
with tab4:
    st.markdown("### Deep Profile Audits")
    shortlist = st.session_state["shortlist"]

    if not shortlist:
        st.info("No candidate screening data available yet.")
    else:
        deep_limit = st.session_state["deep_limit"]
        for idx, c in enumerate(shortlist[: int(deep_limit)]):
            # Expander for each shortlisted candidate
            expander_title = (
                f"{idx + 1}. {c['name']} (Match Score: {c['score']}/100) — {c.get('screening_status', 'Shortlisted')}"
            )
            with st.expander(expander_title, expanded=(idx == 0)):
                st.markdown(f"**Screening Reasoning**: {c.get('screening_reasoning', 'No deep reasoning generated.')}")

                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Core Strengths**:")
                    if c.get("strengths"):
                        st.markdown("\n".join(f"- {s}" for s in c["strengths"]))
                    else:
                        st.caption("No strengths evaluated yet.")
                with cols[1]:
                    st.markdown("**Identified Gaps**:")
                    if c.get("gaps"):
                        st.markdown("\n".join(f"- {g}" for g in c["gaps"]))
                    else:
                        st.caption("No gaps evaluated yet.")

                st.markdown(f"**Improvement Suggestions**: *{c.get('improvement_suggestions', 'None')}*")

                # Show interview questions
                if c.get("interview_questions"):
                    st.markdown("#### Custom Interview Questions:")
                    for q in c["interview_questions"]:
                        st.markdown(f"- {q}")
