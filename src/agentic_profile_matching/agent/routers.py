import numpy as np
from typing import Dict, List, Literal, Optional, Tuple, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_profile_matching.agent.state import AgentState
from agentic_profile_matching.observability import get_logger

logger = get_logger("agentic_profile_matching.agent.routers")


class RouteDecision(BaseModel):
    intent: Literal["extract_requirements", "adjust_requirements", "conversational_query"] = Field(
        description="The target workflow branch for this user input."
    )
    reasoning: str = Field(default="", description="Short 1-sentence reasoning for the routing classification.")


# Rich Semantic Intent Signatures used as baseline vector centroids for zero-shot embedding routing
DEFAULT_INTENT_DESCRIPTIONS: Dict[str, List[str]] = {
    "extract_requirements": [
        "Search for candidates, source resumes, find developers, engineers, and technical talent.",
        "Extract skills, qualifications, and requirements from job descriptions for hiring.",
        "Find or match candidates matching job requirements, tech stack, and experience.",
        "Looking for a full stack developer with React and Node.js or backend engineers.",
        "Search for candidates with Java and Python experience.",
    ],
    "adjust_requirements": [
        "Adjust, modify, update, tighten, or relax existing skill requirements and constraints.",
        "Change experience threshold, make skills mandatory must-have, or remove nice-to-have skills.",
        "Exclude candidates without Docker experience.",
        "Make Python a mandatory must-have skill and increase experience.",
        "Filter by degree, update job title, or refine constraints on current search.",
    ],
    "conversational_query": [
        "Ask general technical questions, software engineering concepts, or industry technology trends.",
        "Perform an internet search, search Google, browse the web, or check online documentation.",
        "Compare active candidates, analyze candidate differences, or explain ranking scores and notes.",
        "Why is candidate Alice ranked higher than Bob?",
        "Tell me about graph engineering in 2026.",
        "Greetings, assistant capabilities, help, or conversational questions.",
    ],
}


# In-memory storage for dynamically LLM-synthesized anchors
_DYNAMIC_INTENT_ANCHORS: Optional[Dict[str, List[str]]] = None

# In-memory query routing cache for sub-millisecond repeated queries (query_hash, has_reqs, has_shortlist) -> intent
_ROUTING_CACHE: Dict[Tuple[str, bool, bool], str] = {}

# Cached embedder and pre-computed anchor embeddings
_EMBEDDER = None
_ANCHOR_EMBEDDINGS: Optional[Dict[str, np.ndarray]] = None


def generate_dynamic_intent_anchors(llm: Optional[Any] = None) -> Dict[str, List[str]]:
    """
    Dynamically synthesizes diverse natural language query exemplars across intent categories
    using an LLM, eliminating the need for rigid static anchor dictionaries in source code.
    """
    global _DYNAMIC_INTENT_ANCHORS, _ANCHOR_EMBEDDINGS
    try:
        if llm is None:
            from agentic_profile_matching import config as app_config
            import os

            api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
            if api_key:
                llm = app_config.get_llm_model(api_key=api_key)

        if llm is not None:

            class DynamicAnchors(BaseModel):
                extract_requirements: List[str] = Field(
                    description="5 diverse queries asking to search/source resumes or providing job descriptions"
                )
                adjust_requirements: List[str] = Field(
                    description="5 diverse queries modifying skills, experience, or filters for active search"
                )
                conversational_query: List[str] = Field(
                    description="5 diverse queries asking tech concepts, internet searches, or comparing candidates"
                )

            from agentic_profile_matching.tools import invoke_structured

            prompt = [
                SystemMessage(
                    content=(
                        "Generate 5 natural language query exemplars for each routing intent in an AI recruiter engine:\n"
                        "1. extract_requirements: sourcing candidates, posting JDs, hiring for roles.\n"
                        "2. adjust_requirements: refining constraints, tightening experience, adding must-have skills.\n"
                        "3. conversational_query: general tech/engineering concepts, internet web searches, or comparing candidates."
                    )
                ),
                HumanMessage(content="Generate diverse, realistic query exemplars."),
            ]
            result = invoke_structured(llm, prompt, DynamicAnchors)
            if result and all(
                k in result for k in ["extract_requirements", "adjust_requirements", "conversational_query"]
            ):
                _DYNAMIC_INTENT_ANCHORS = {
                    "extract_requirements": result["extract_requirements"],
                    "adjust_requirements": result["adjust_requirements"],
                    "conversational_query": result["conversational_query"],
                }
                logger.info("Successfully synthesized dynamic intent anchors via LLM.")
                _ANCHOR_EMBEDDINGS = None  # Force re-embedding
                return _DYNAMIC_INTENT_ANCHORS
    except Exception as e:
        logger.warning(f"Could not generate dynamic anchors via LLM: {e}")

    _DYNAMIC_INTENT_ANCHORS = DEFAULT_INTENT_DESCRIPTIONS
    return _DYNAMIC_INTENT_ANCHORS


def _get_embedder():
    global _EMBEDDER, _ANCHOR_EMBEDDINGS
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            from agentic_profile_matching import config as app_config

            _EMBEDDER = SentenceTransformer(app_config.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning(f"Could not initialize SentenceTransformer for semantic routing: {e}")
            _EMBEDDER = None
            _ANCHOR_EMBEDDINGS = None
            return None, None

    if _ANCHOR_EMBEDDINGS is None and _EMBEDDER is not None:
        try:
            anchors_to_use = _DYNAMIC_INTENT_ANCHORS or DEFAULT_INTENT_DESCRIPTIONS
            _ANCHOR_EMBEDDINGS = {}
            for intent, phrases in anchors_to_use.items():
                embeddings = _EMBEDDER.encode(phrases, normalize_embeddings=True)
                _ANCHOR_EMBEDDINGS[intent] = np.array(embeddings)
        except Exception as e:
            logger.warning(f"Could not compute anchor embeddings: {e}")
            _ANCHOR_EMBEDDINGS = None

    return _EMBEDDER, _ANCHOR_EMBEDDINGS


def _classify_via_semantic_similarity(query: str, confidence_threshold: float = 0.20) -> Optional[str]:
    """
    Tier 1 Local Semantic Router: Computes cosine similarity against dynamic intent prototypes.
    Returns the predicted intent if max similarity exceeds confidence_threshold, else None.
    """
    embedder, anchor_dict = _get_embedder()
    if embedder is None or not anchor_dict:
        return None

    try:
        query_emb = embedder.encode([query], normalize_embeddings=True)[0]
        best_intent = None
        best_score = -1.0

        for intent, anchor_embs in anchor_dict.items():
            sims = np.dot(anchor_embs, query_emb)
            max_sim = float(np.max(sims))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent

        logger.debug(f"Semantic routing evaluated top intent: {best_intent} (confidence: {best_score:.3f})")
        if best_score >= confidence_threshold:
            return best_intent
    except Exception as e:
        logger.warning(f"Error during semantic vector classification: {e}")

    return None


ROUTER_SYSTEM_PROMPT = """You are an expert intent classifier for Yojaka AI (an AI Recruiter & Technical Assistant).
Classify the user's latest message into one of three routing branches:

1. 'conversational_query':
   - CHOOSE THIS for general knowledge, technical concepts, engineering trends, architecture questions (e.g. 'tell me about graph engineering in 2026', 'what is LangGraph', 'explain vector databases').
   - CHOOSE THIS for internet/Google searches (e.g. 'search google for latest AI news', 'search the web for FastAPI vs Django').
   - CHOOSE THIS for questions about existing candidates (e.g. 'compare candidate 1 and 2', 'why is Alice ranked higher', 'what are Bob's strengths').
   - CHOOSE THIS for greetings or questions about the system (e.g. 'hello', 'who are you', 'what can you do').

2. 'extract_requirements':
   - CHOOSE THIS ONLY when the user explicitly provides a Job Description, or explicitly asks to search/source/screen/match candidates or resumes from the talent pool (e.g. 'find candidates with Python and AWS', 'search resumes for 5+ years React developers', 'looking for a Senior Backend Engineer').

3. 'adjust_requirements':
   - CHOOSE THIS when the user modifies active constraints (adding must-have skills, changing experience years, relaxing filters) for an existing candidate search.

Current Session Context:
- Has Active Requirements: {has_requirements}
- Active Shortlist Count: {shortlist_count}"""


def _classify_via_llm(state: AgentState) -> Optional[str]:
    """
    Primary Router: LLM Structured Intent Classifier for robust, zero-shot intent resolution
    across any natural language variation, typos, slang, domain terminology, and context.
    """
    try:
        import os
        from agentic_profile_matching import config as app_config
        from agentic_profile_matching.tools import invoke_structured

        llm_provider = (state.get("llm_provider") or app_config.DEFAULT_PROVIDER).lower().strip()
        llm_model = state.get("llm_model") or app_config.DEFAULT_MODEL
        api_url = state.get("api_url")

        provider_keys = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "sarvam": "SARVAM_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env_var = provider_keys.get(llm_provider, "GROQ_API_KEY")
        api_key = (
            state.get("api_key")
            or os.environ.get(env_var)
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )

        if not api_key:
            return None

        llm = app_config.get_llm_model(
            provider=llm_provider,
            model_name=llm_model,
            api_key=api_key,
            api_url=api_url,
        )

        has_requirements = bool(state.get("requirements"))
        shortlist_count = len(state.get("shortlist", []))
        messages = state.get("messages", [])
        last_msg = messages[-1].content if messages else ""

        system_content = ROUTER_SYSTEM_PROMPT.format(
            has_requirements=has_requirements,
            shortlist_count=shortlist_count,
        )

        prompt_messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"User Message to route: {last_msg}"),
        ]

        result = invoke_structured(llm, prompt_messages, RouteDecision)
        intent = result.get("intent")
        if intent in ["extract_requirements", "adjust_requirements", "conversational_query"]:
            logger.info(f"LLM routing decision: {intent} (Reason: {result.get('reasoning', 'N/A')})")
            return intent
    except Exception as e:
        logger.warning(f"LLM routing error: {e}")

    return None


def route_input(state: AgentState) -> str:
    """
    Production LLM-Driven Intent Router (2026 Standards):
    1. Zero-Token Structural Fast-Paths:
       - Empty message -> extract_requirements.
       - Multi-line raw Job Description paste -> extract_requirements.
    2. In-Memory Query Cache:
       - Sub-millisecond lookup for identical queries in the same context.
    3. Primary Intelligence:
       - LLM Structured Intent Classifier (with_structured_output).
    4. Resilient Local Fallbacks:
       - Dynamic semantic embedding similarity against intent prototypes.
       - State-aware heuristic fallback.
    """
    messages = state.get("messages", [])
    if not messages:
        return "extract_requirements"

    last_msg = messages[-1].content.strip()
    if not last_msg:
        return "extract_requirements"

    lower_msg = last_msg.lower()
    lines = [line.strip() for line in last_msg.split("\n") if line.strip()]

    # 1. Structural Fast-Path: Multi-line Job Description Paste (Zero token waste on raw pastes)
    if len(lines) > 3 and any(
        w in lower_msg
        for w in [
            "job description",
            "requirements:",
            "duties:",
            "responsibilities:",
            "qualifications:",
            "about the role:",
            "what you'll do:",
        ]
    ):
        return "extract_requirements"

    has_requirements = bool(state.get("requirements"))
    has_shortlist = bool(state.get("shortlist", []))

    # 2. In-Memory Cache Lookup (0ms sub-millisecond execution for repeated queries)
    cache_key = (lower_msg, has_requirements, has_shortlist)
    if cache_key in _ROUTING_CACHE:
        cached_intent = _ROUTING_CACHE[cache_key]
        logger.debug(f"Resolved intent from routing cache: {cached_intent}")
        return cached_intent

    # 3. Primary Tier: LLM Structured Intent Classifier
    llm_intent = _classify_via_llm(state)
    if llm_intent:
        _ROUTING_CACHE[cache_key] = llm_intent
        return llm_intent

    # 4. Secondary Tier: Local Semantic Embedding Router with Dynamic Intent Prototypes
    semantic_intent = _classify_via_semantic_similarity(last_msg, confidence_threshold=0.20)
    if semantic_intent:
        if semantic_intent == "adjust_requirements" and not has_requirements:
            semantic_intent = "extract_requirements"
        logger.info(f"Resolved intent via local semantic similarity: {semantic_intent}")
        _ROUTING_CACHE[cache_key] = semantic_intent
        return semantic_intent

    # 5. Tertiary Tier: State-Aware Resilient Fallback (Zero hardcoded arrays)
    if has_requirements and any(
        w in lower_msg for w in ["add", "make", "increase", "decrease", "remove", "change", "filter", "require"]
    ):
        return "adjust_requirements"

    if has_shortlist and any(w in lower_msg for w in ["why", "compare", "rank", "explain", "who", "versus", "vs"]):
        return "conversational_query"

    if any(
        lower_msg.startswith(prefix)
        for prefix in ["what", "how", "why", "who", "tell me", "can you", "search google", "search the web", "explain"]
    ):
        return "conversational_query"

    if not has_requirements:
        return "extract_requirements"

    return "adjust_requirements"
