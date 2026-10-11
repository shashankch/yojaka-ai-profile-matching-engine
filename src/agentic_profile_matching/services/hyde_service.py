"""
HyDE (Hypothetical Document Embeddings) Query Synthesis Service (Phase 19 / ADR-019).
Synthesizes a hypothetical ideal candidate profile summary from job requirements
before vector retrieval to bridge the semantic terminology gap between terse queries
and long-form resume accomplishments.
"""

import hashlib
import json
from typing import Any, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from agentic_profile_matching import config
from agentic_profile_matching.observability import get_logger

logger = get_logger("agentic_profile_matching.services.hyde")

HYDE_SYSTEM_PROMPT = """You are an expert technical recruiter and resume writer.
Given a job description or set of requirements, generate a concise, realistic hypothetical resume summary (80-130 words) for an ideal candidate who fits and exceeds these requirements.
Highlight relevant architecture accomplishments, production engineering responsibilities, frameworks, and key technologies using natural resume terminology and achievement-oriented prose.
Do NOT include greetings, section headers, bullet lists, or meta-commentary. Output ONLY the continuous hypothetical candidate profile paragraph."""


class HyDEService:
    """
    Hypothetical Document Embeddings service for query expansion and
    semantic vocabulary bridging.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        timeout: Optional[float] = None,
        llm: Optional[BaseChatModel] = None,
        enable_cache: bool = True,
    ):
        self.enabled = config.USE_HYDE if enabled is None else enabled
        self.timeout = config.HYDE_TIMEOUT if timeout is None else timeout
        self.llm = llm
        self.enable_cache = enable_cache
        self._cache: Dict[str, str] = {}

    def _hash_query(self, query: str, requirements: Optional[Dict[str, Any]] = None) -> str:
        payload = f"{query.strip()}_{json.dumps(requirements or {}, sort_keys=True)}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def generate_hypothetical_profile(
        self,
        query: str,
        requirements: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> str:
        """
        Synthesizes a hypothetical candidate profile summary matching requirements.
        Falls back to heuristic generation if LLM is unavailable or disabled.
        """
        if not self.enabled:
            return query

        cache_key = self._hash_query(query, requirements)
        if self.enable_cache and cache_key in self._cache:
            logger.debug("HyDE cache hit for query")
            return self._cache[cache_key]

        # Construct prompt content from query and structured requirements
        reqs = requirements or {}
        title = reqs.get("title", "")
        must_haves = reqs.get("must_have_skills", [])
        min_exp = reqs.get("min_experience_years", 0)
        education = reqs.get("education_level", "")

        user_content = f"Target Role / Query: {query}\n"
        if title:
            user_content += f"Job Title: {title}\n"
        if must_haves:
            user_content += f"Mandatory Skills: {', '.join(must_haves)}\n"
        if min_exp:
            user_content += f"Minimum Experience: {min_exp}+ years\n"
        if education:
            user_content += f"Target Education: {education}\n"

        # 1. Attempt LLM synthesis if provided
        active_llm = llm or self.llm
        if active_llm is not None:
            try:
                messages = [
                    SystemMessage(content=HYDE_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]
                response = active_llm.invoke(messages)
                profile_text = response.content if hasattr(response, "content") else str(response)
                profile_text = profile_text.strip().replace('"', "")
                if len(profile_text) > 40:
                    combined_profile = f"{query}. {profile_text}"
                    if self.enable_cache:
                        self._cache[cache_key] = combined_profile
                    logger.info(f"Synthesized HyDE profile ({len(combined_profile.split())} words)")
                    return combined_profile
            except Exception as e:
                logger.warning(f"HyDE LLM synthesis failed or timed out: {e}. Using heuristic fallback.")

        # 2. Deterministic Heuristic Synthesis Fallback
        heuristic_profile = self._build_heuristic_profile(query, reqs)
        if self.enable_cache:
            self._cache[cache_key] = heuristic_profile
        return heuristic_profile

    def _build_heuristic_profile(self, query: str, reqs: Dict[str, Any]) -> str:
        """Constructs a deterministic synthetic resume summary anchored on query and requirements."""
        title = reqs.get("title") or ""
        min_exp = reqs.get("min_experience_years") or 3
        skills = reqs.get("must_have_skills") or []
        education = reqs.get("education_level") or "B.S. in Computer Science or related degree"

        skills_phrase = (
            f"skilled in {', '.join(skills)}" if skills else "possessing relevant software engineering skills"
        )
        title_phrase = f"as {title}" if title else "in target engineering domain"

        return (
            f"Target Role / Profile Summary: {query}. "
            f"Experienced software engineer {title_phrase} with {min_exp}+ years of hands-on production experience, {skills_phrase}. "
            f"Demonstrated technical expertise delivering resilient applications, clean modular architectures, and modern engineering practices. "
            f"Holds {education} with solid problem-solving capabilities, Core Competencies, and achievement-oriented delivery."
        )

    def clear_cache(self) -> None:
        """Clears the in-memory HyDE profile cache."""
        self._cache.clear()
