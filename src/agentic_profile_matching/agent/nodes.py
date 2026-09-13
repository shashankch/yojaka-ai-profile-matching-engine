import os
import time
import json
import concurrent.futures
import threading
from typing import Dict, Any, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from agentic_profile_matching.mcp_client import mcp_client

from agentic_profile_matching.observability import get_logger, trace_node
from agentic_profile_matching import config as app_config
from agentic_profile_matching.fs_client import read_file
from agentic_profile_matching.job_matcher import JobMatcher
from agentic_profile_matching.tools import (
    extract_requirements,
    compare_candidates,
    generate_interview_questions,
    execute_with_retry,
    invoke_structured,
    DeepScreenOutput,
    JobRequirementsOutput,
)
from agentic_profile_matching.agent.state import AgentState
from agentic_profile_matching.agent.prompts import (
    DEEP_SCREEN_SYSTEM_PROMPT,
    RANKING_EXPLANATION_SYSTEM_PROMPT,
    RANKING_EXPLANATION_USER_PROMPT,
    ADJUST_REQUIREMENTS_SYSTEM_PROMPT,
    CONVERSATIONAL_QUERY_SYSTEM_PROMPT,
)
from agentic_profile_matching.stores import BaseVectorStore

logger = get_logger("agentic_profile_matching.agent.nodes")


def _get_llm(state: AgentState, config: Optional[RunnableConfig] = None):
    """
    Retrieves pre-instantiated LLM instance if passed via graph configuration or state,
    otherwise falls back to dynamically building the model via app_config.get_llm_model.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}

    # 1. Pre-instantiated LLM instance
    if "llm" in configurable and configurable["llm"] is not None:
        return configurable["llm"]
    if isinstance(state, dict) and state.get("llm") is not None:
        return state.get("llm")

    # 2. Configurable or state fallback parameters
    provider = (
        configurable.get("llm_provider")
        or (state.get("llm_provider") if isinstance(state, dict) else None)
        or app_config.DEFAULT_PROVIDER
    )
    model_name = (
        configurable.get("llm_model")
        or (state.get("llm_model") if isinstance(state, dict) else None)
        or app_config.DEFAULT_MODEL
    )

    provider_env_map = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
        "sarvam": "SARVAM_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    target_env = provider_env_map.get(str(provider).lower(), "GROQ_API_KEY")

    api_key = (
        configurable.get("api_key")
        or (state.get("api_key") if isinstance(state, dict) else None)
        or os.getenv(target_env)
        or os.getenv("GROQ_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("SARVAM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    api_url = (
        configurable.get("api_url")
        or os.getenv("GROQ_API_URL")
        or (state.get("api_url") if isinstance(state, dict) else None)
    )

    return app_config.get_llm_model(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        api_url=api_url,
    )


def _get_store(config: Optional[RunnableConfig] = None) -> Optional[BaseVectorStore]:
    """
    Retrieves vector store instance passed via graph configuration,
    returning None to fall back to ChromaVectorStore default in JobMatcher.
    """
    if config and isinstance(config, dict):
        configurable = config.get("configurable", {})
        if "store" in configurable and configurable["store"] is not None:
            return configurable["store"]
    return None


@trace_node("parse_input")
def parse_input_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Inspects and prepares state metadata prior to conditional graph routing.
    """
    errors = state.get("errors")
    if errors is None:
        errors = []

    # Capture the previous shortlist before updating
    prev_shortlist = state.get("shortlist", [])

    return {
        "current_round": 1,
        "previous_shortlist": prev_shortlist,
        "errors": errors,
    }


@trace_node("extract_requirements")
def extract_requirements_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    LLM extracts structured job requirements from raw input message content.
    """
    messages = state.get("messages", [])
    errors = state.get("errors")
    if errors is None:
        errors = []

    fallback_reqs = {
        "title": "Software Engineer",
        "must_have_skills": [],
        "nice_to_have_skills": [],
        "min_experience_years": 0,
        "education_level": "Not Specified",
        "other_constraints": [],
    }

    if not messages:
        return {"requirements": fallback_reqs, "errors": errors}

    try:
        last_msg = messages[-1].content

        # Dynamic LLM builder with injection support
        llm = _get_llm(state, config)

        logger.info("Extracting job requirements from input...")
        requirements = extract_requirements(last_msg, llm)
        if not isinstance(requirements, dict):
            requirements = fallback_reqs
            errors.append("Invalid structure returned by extract_requirements, using fallback.")

        return {"requirements": requirements, "current_round": 1, "errors": errors}
    except Exception as e:
        logger.error(f"Error in extract_requirements_node: {e}")
        errors.append(f"Requirements extraction failed: {str(e)}. Using fallback requirements.")
        return {"requirements": fallback_reqs, "current_round": 1, "errors": errors}


@trace_node("search_resumes")
def search_resumes_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Retrieves candidate resumes matching constraints using local hybrid search.
    """
    errors = state.get("errors")
    if errors is None:
        errors = []

    requirements = state.get("requirements", {}) or {}
    title = requirements.get("title", "Software Engineer")
    must_have = requirements.get("must_have_skills", [])
    min_exp = requirements.get("min_experience_years", 0)
    skill_expansions = requirements.get("skill_expansions", {})

    coarse_limit = state.get("coarse_screen_limit") or app_config.DEFAULT_COARSE_LIMIT
    retrieval_k = max(int(coarse_limit * 1.5), 15)

    try:
        # Query with requirements to find top candidates using injected store (if any)
        store = _get_store(config)
        matcher = JobMatcher(store=store)

        # Enrich search query text with semantic expansion terms
        expanded_keywords = []
        if skill_expansions:
            for parent_skill, children in skill_expansions.items():
                if children:
                    expanded_keywords.append(f"{parent_skill} ({', '.join(children[:3])})")
        skills_query_str = ", ".join(expanded_keywords) if expanded_keywords else ", ".join(must_have)
        query_text = f"Job Title: {title}. Must-Have Skills: {skills_query_str}. Experience: {min_exp} years."
        logger.info(f"Retrieving candidate resumes for requirements: {requirements}")

        # Try strict filtering first (with semantic expansion for must_haves)
        results = matcher.match(
            job_description=query_text,
            k=retrieval_k,
            min_exp=min_exp,
            must_have_skills=must_have,
            skill_expansions=skill_expansions,
            apply_filters=True,
        )

        # Fallback to no strict filtering if zero matches exist
        if not results or not results.get("top_matches"):
            logger.info("Zero candidates satisfied strict criteria. Relaxing filters...")
            results = matcher.match(
                job_description=query_text,
                k=retrieval_k,
                min_exp=min_exp,
                must_have_skills=must_have,
                skill_expansions=skill_expansions,
                apply_filters=False,
            )

        raw_matches = results.get("top_matches", []) if results else []
        return {"shortlist": raw_matches, "errors": errors}
    except Exception as e:
        logger.error(f"Error in search_resumes_node: {e}")
        errors.append(f"Resume search failed: {str(e)}. RAG retrieval skipped.")
        return {"shortlist": [], "errors": errors}


@trace_node("rank_candidates")
def rank_candidates_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Round 1: Performs coarse scoring, matches core skills, and filters shortlist to Top 10.
    """
    errors = state.get("errors")
    if errors is None:
        errors = []

    try:
        requirements = state.get("requirements", {}) or {}
        skill_expansions = requirements.get("skill_expansions", {})
        expansions_norm = {k.lower().strip(): v for k, v in skill_expansions.items() if isinstance(v, list)}
        raw_matches = state.get("shortlist", []) or []
        ranked_shortlist = []

        for c in raw_matches:
            candidate_skills = [s.lower().strip() for s in c.get("matched_skills", []) + c.get("skills", []) if s]

            # Calculate matched must-have and nice-to-have skills with semantic expansion support
            matched_must = []
            missing_must = []
            for req_skill in requirements.get("must_have_skills", []):
                req_lower = req_skill.lower().strip()
                synonyms = [req_lower] + [syn.lower().strip() for syn in expansions_norm.get(req_lower, []) if syn]
                if any(syn in candidate_skills for syn in synonyms):
                    matched_must.append(req_skill)
                else:
                    missing_must.append(req_skill)

            # Structure CandidateMatch profile
            candidate_profile = {
                "candidate_id": c.get("resume_path"),
                "name": c.get("candidate_name", "Unknown"),
                "score": c.get("match_score", 0),
                "matched_skills": matched_must,
                "missing_skills": missing_must,
                "skills": c.get("skills", []),
                "raw_text": c.get("raw_text", ""),
                "experience_years": c.get("experience_years", 0),
                "education": c.get("education", "Not Specified"),
                "relevance_excerpts": c.get("relevant_excerpts", []),
                "strengths": [],
                "gaps": [],
                "improvement_suggestions": "",
                "screening_status": "Shortlisted",
                "screening_reasoning": "Coarse filtering match",
                "interview_questions": [],
            }
            ranked_shortlist.append(candidate_profile)

        coarse_limit = state.get("coarse_screen_limit") or app_config.DEFAULT_COARSE_LIMIT
        # Sort and slice to limit downstream token consumption
        ranked_shortlist.sort(key=lambda x: x.get("score", 0), reverse=True)
        return {
            "shortlist": ranked_shortlist[: int(coarse_limit)],
            "current_round": 1,
            "errors": errors,
        }
    except Exception as e:
        logger.error(f"Error in rank_candidates_node: {e}")
        errors.append(f"Ranking failed: {str(e)}")
        return {"shortlist": [], "current_round": 1, "errors": errors}


@trace_node("deep_screen")
def deep_screen_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Round 2: profile deep text audit.
    Evaluates Top 5 candidate resumes with parallel processing and copy-on-write state immutability,
    mapping strengths, gaps, and suggestions.
    """
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    errors = list(state.get("errors") or [])

    try:
        llm = _get_llm(state, config)
    except Exception as e:
        logger.error(f"Error building LLM model in deep_screen_node: {e}")
        errors.append(f"Failed to build LLM for deep screening: {str(e)}")
        llm = None

    deep_limit = state.get("deep_screen_limit") or app_config.DEFAULT_DEEP_LIMIT
    # Screen candidates dynamically based on deep_limit
    candidates_to_screen = shortlist[: int(deep_limit)]
    logger.info(f"Executing Round 2 (Deep Screening) on top {len(candidates_to_screen)} candidates...")

    rate_limiter = threading.Semaphore(2)
    _rate_lock = threading.Lock()
    _last_call_time = 0.0

    def _rate_limited_call():
        nonlocal _last_call_time
        with _rate_lock:
            now = time.time()
            elapsed = now - _last_call_time
            if elapsed < app_config.THROTTLE_DELAY:
                time.sleep(app_config.THROTTLE_DELAY - elapsed)
            _last_call_time = time.time()

    def _screen_single(c_orig: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        c = dict(c_orig)
        candidate_name = c.get("candidate_name") or c.get("name") or "Unknown Candidate"
        resume_text = c.get("raw_text")

        if not resume_text:
            candidate_id = c.get("candidate_id") or ""
            res = read_file(candidate_id)
            if not res or not res.get("success"):
                err_msg = f"Incomplete parsing: could not read resume for {candidate_name} ({res.get('error') if res else 'Empty response'})"
                logger.warning(err_msg)
                c["strengths"] = ["Strong skill overlap based on RAG indexing"]
                c["gaps"] = ["Could not audit text (file unreadable / unparsed)"]
                c["improvement_suggestions"] = "Review resume file formatting before interviewing candidate."
                c["screening_status"] = "Screened"
                c["screening_reasoning"] = (
                    f"Fallback screening (unreadable file: {res.get('error') if res else 'Unknown error'})"
                )
                return c, err_msg
            resume_text = res["content"]

        # Truncate content to avoid model token limits
        if len(resume_text) > app_config.RESUME_TRUNCATION_LIMIT:
            resume_text = resume_text[: app_config.RESUME_TRUNCATION_LIMIT] + "... [truncated]"

        if llm is None:
            c["strengths"] = ["Semantic match based on vector DB indexing"]
            c["gaps"] = ["Skipped deep screening audit due to missing LLM configuration"]
            c["improvement_suggestions"] = "Configure LLM provider with a valid API key."
            c["screening_status"] = "Screened"
            c["screening_reasoning"] = "Fallback screening due to unconfigured LLM"
            return c, None

        prompt_content = f"""Candidate: {candidate_name}
Job Title: {requirements.get("title", "Software Engineer")}
Job Requirements: {requirements}
Candidate Resume Text:
{resume_text}"""

        def _call_deep_screen():
            messages = [
                SystemMessage(content=DEEP_SCREEN_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content),
            ]
            return invoke_structured(llm, messages, DeepScreenOutput)

        with rate_limiter:
            try:
                _rate_limited_call()
                result = execute_with_retry(_call_deep_screen)
                c["strengths"] = result.get("strengths", [])
                c["gaps"] = result.get("gaps", [])
                c["improvement_suggestions"] = result.get("improvement_suggestions", "")
                c["screening_status"] = result.get("screening_status", "Screened")
                c["screening_reasoning"] = result.get("screening_reasoning", "")
                return c, None
            except Exception as e:
                err_msg = f"Failed to screen {candidate_name}: {e}"
                logger.error(err_msg)
                c["strengths"] = ["Semantic match based on vector DB indexing"]
                c["gaps"] = ["Skipped deep screening audit due to LLM error"]
                c["improvement_suggestions"] = "Schedule interview to evaluate candidates skills directly."
                c["screening_status"] = "Screened"
                c["screening_reasoning"] = f"Fallback screening due to LLM or parse error: {str(e)}"
                return c, err_msg

    updated_screened = []
    if candidates_to_screen:
        max_workers = min(3, len(candidates_to_screen))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_screen_single, c): idx for idx, c in enumerate(candidates_to_screen)}
            results_by_idx = {}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    c_res, err = future.result()
                    results_by_idx[idx] = c_res
                    if err:
                        errors.append(err)
                except Exception as ex:
                    err_msg = f"Unexpected error screening candidate at index {idx}: {ex}"
                    logger.error(err_msg)
                    errors.append(err_msg)
                    fallback_c = dict(candidates_to_screen[idx])
                    fallback_c["strengths"] = fallback_c.get("strengths") or [
                        "Semantic match based on vector DB indexing"
                    ]
                    fallback_c["gaps"] = fallback_c.get("gaps") or [
                        "Skipped deep screening audit due to unhandled execution error"
                    ]
                    fallback_c["improvement_suggestions"] = fallback_c.get("improvement_suggestions") or (
                        "Schedule interview to evaluate candidates skills directly."
                    )
                    fallback_c["screening_status"] = "Screened"
                    fallback_c["screening_reasoning"] = f"Fallback screening (unhandled error: {ex})"
                    results_by_idx[idx] = fallback_c

            updated_screened = [results_by_idx[i] for i in range(len(candidates_to_screen))]

    # Combine screened candidates with remainder of shortlist (copy-on-write immutability)
    final_shortlist = updated_screened + [dict(c) for c in shortlist[int(deep_limit) :]]
    return {"shortlist": final_shortlist, "current_round": 2, "errors": errors}


@trace_node("recommendation")
def recommendation_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Round 3: Final Hiring decisions & customized technical screening questions generator.
    Enforces copy-on-write state immutability and robust safety guardrails.
    """
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    errors = list(state.get("errors") or [])

    try:
        llm = _get_llm(state, config)
    except Exception as e:
        logger.error(f"Error building LLM model in recommendation_node: {e}")
        errors.append(f"Failed to build LLM for interview question generation: {str(e)}")
        llm = None

    rec_limit = state.get("recommendation_limit") or app_config.DEFAULT_RECOMMENDATION_LIMIT
    # Generate questions and recommendations dynamically based on rec_limit
    candidates_to_decide = shortlist[: int(rec_limit)]
    logger.info(f"Executing Round 3 (Hire Decision & QGen) for top {len(candidates_to_decide)} candidates...")

    updated_decided = []
    for idx, c_orig in enumerate(candidates_to_decide):
        c = dict(c_orig)
        if idx > 0:
            time.sleep(app_config.THROTTLE_DELAY)

        # Generate interview questions targeting gaps
        try:
            if llm is not None:
                questions = generate_interview_questions(
                    candidate_name=c.get("name", "Unknown"),
                    skills=c.get("matched_skills", []) + c.get("skills", []),
                    gaps=c.get("gaps") if c.get("gaps") else c.get("missing_skills", []),
                    requirements=requirements,
                    llm=llm,
                )
                c["interview_questions"] = questions
            else:
                raise ValueError("LLM not configured.")
        except Exception as e:
            err_msg = f"Failed to generate interview questions for {c.get('name')}: {str(e)}"
            logger.error(err_msg)
            errors.append(err_msg)
            c["interview_questions"] = [
                "Can you walk me through your engineering experience?",
                "What is your approach to learning new technologies?",
                "How do you handle microservices architecture issues?",
            ]

        # Preserve LLM deep screening decision if assigned, enforcing mandatory constraint safety guardrails
        try:
            llm_status = c.get("screening_status", "")
            missing = c.get("missing_skills", [])
            candidate_exp = c.get("experience_years", 0)
            min_exp_req = requirements.get("min_experience_years", 0)

            exp_meets = candidate_exp >= min_exp_req

            if llm_status in ["Strong Hire", "Borderline Hire", "Rejected / No-Hire"]:
                # Guardrail: If candidate is missing 2+ mandatory skills or fails exp, override to Rejected
                if len(missing) >= 2 or not exp_meets:
                    c["screening_status"] = "Rejected / No-Hire"
                else:
                    c["screening_status"] = llm_status
            else:
                # Dynamic fallback if LLM status was not assigned
                if len(missing) == 0 and exp_meets:
                    c["screening_status"] = "Strong Hire"
                elif len(missing) <= 1 and exp_meets:
                    c["screening_status"] = "Borderline Hire"
                else:
                    c["screening_status"] = "Rejected / No-Hire"
        except Exception:
            c["screening_status"] = "Screened"

        updated_decided.append(c)

    final_shortlist = updated_decided + [dict(c) for c in shortlist[int(rec_limit) :]]
    return {"shortlist": final_shortlist, "current_round": 3, "errors": errors}


@trace_node("generate_report")
def generate_report_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Compiles candidate records, analysis reports, and interview matrices into a Markdown report.
    """
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    ranking_explanation = state.get("ranking_explanation", "")
    messages = state.get("messages", [])

    feedback_instructions = ""
    for m in reversed(messages):
        if hasattr(m, "content") and m.content:
            feedback_instructions = m.content
            break

    if feedback_instructions:
        try:
            llm = _get_llm(state, config)
        except Exception as e:
            logger.error(f"Error building LLM model in generate_report_node: {e}")
            llm = None

        prev_summary = "\n".join(
            f"  {idx + 1}. {c['name']} (Score: {c['score']}/100, Status: {c.get('screening_status', 'Shortlisted')}, Experience: {c.get('experience_years', 0)} Years, Matched Skills: {c.get('matched_skills', [])}, Missing Skills: {c.get('missing_skills', [])})"
            for idx, c in enumerate(state.get("previous_shortlist", []))
        )
        curr_summary = "\n".join(
            f"  {idx + 1}. {c['name']} (Score: {c['score']}/100, Status: {c.get('screening_status', 'Shortlisted')}, Experience: {c.get('experience_years', 0)} Years, Matched Skills: {c.get('matched_skills', [])}, Missing Skills: {c.get('missing_skills', [])})"
            for idx, c in enumerate(shortlist)
        )

        user_prompt = RANKING_EXPLANATION_USER_PROMPT.format(
            feedback_instructions=feedback_instructions,
            prev_summary=prev_summary if prev_summary else "None",
            curr_summary=curr_summary if curr_summary else "None",
        )

        def _call_explanation():
            msgs = [
                SystemMessage(content=RANKING_EXPLANATION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            resp = llm.invoke(msgs)
            return resp.content

        try:
            if llm is not None:
                ranking_explanation = execute_with_retry(_call_explanation)
        except Exception as e:
            logger.warning(f"Failed to generate ranking explanation via LLM: {e}")

    rec_limit = state.get("recommendation_limit") or app_config.DEFAULT_RECOMMENDATION_LIMIT
    deep_limit = state.get("deep_screen_limit") or app_config.DEFAULT_DEEP_LIMIT

    # Generate side-by-side comparison table
    candidate_ids = [c["candidate_id"] for c in shortlist[: int(rec_limit)]]
    compare_matrix = compare_candidates(candidate_ids, shortlist)

    report_lines = [
        "# Candidate Match & Screening Report",
        "",
        f"### Active Job Profile: {requirements.get('title', 'Software Engineer')}",
        f"- **Min Experience**: {requirements.get('min_experience_years', 0)} Years",
        f"- **Must-Have Skills**: {', '.join(requirements.get('must_have_skills', [])) if requirements.get('must_have_skills') else 'None'}",
        f"- **Nice-To-Have Skills**: {', '.join(requirements.get('nice_to_have_skills', [])) if requirements.get('nice_to_have_skills') else 'None'}",
        f"- **Education Target**: {requirements.get('education_level', 'Not Specified')}",
        "",
    ]

    if ranking_explanation:
        report_lines.extend(["## 🔄 Ranking Changes Explanation", "", ranking_explanation, ""])

    report_lines.extend(
        [
            "## Candidate Comparison Matrix",
            "",
            compare_matrix,
            "",
            "## Screening Details",
            "",
        ]
    )

    for idx, c in enumerate(shortlist[: int(deep_limit)]):  # Deep screening up to dynamic deep_limit candidates
        status = c.get("screening_status", "").lower()
        if "reject" in status or "no-hire" in status:
            status_color = "🔴"
        elif "borderline" in status:
            status_color = "🟡"
        else:
            status_color = "🟢"

        report_lines.extend(
            [
                f"### {idx + 1}. {c['name']} (Score: {c['score']}/100) - {status_color} {c.get('screening_status', 'Shortlisted')}",
                f"- **Experience**: {c['experience_years']} Years",
                f"- **Education**: {c['education']}",
                f"- **Matched Skills**: {', '.join(c['matched_skills']) if c['matched_skills'] else 'None'}",
                f"- **Missing Skills**: {', '.join(c['missing_skills']) if c['missing_skills'] else 'None'}",
                "",
                "#### Diagnostic Breakdown:",
                f"- **Strengths**: {', '.join(c.get('strengths', [])) if c.get('strengths') else 'Not evaluated yet'}",
                f"- **Gaps**: {', '.join(c.get('gaps', [])) if c.get('gaps') else 'None'}",
                f"- **Suggestions**: *{c.get('improvement_suggestions', 'None')}*",
                f"- **Reasoning**: {c.get('screening_reasoning', 'Coarse filtering match')}",
                "",
            ]
        )

        if c.get("interview_questions"):
            report_lines.extend(
                [
                    "#### Tailored Screening Questions:",
                    "\n".join(f"  - {q}" for q in c["interview_questions"]),
                    "",
                ]
            )

    if state.get("errors"):
        report_lines.extend(
            [
                "---",
                "### ⚠️ System warnings / execution errors during matching run:",
                "",
                "\n".join(f"- {err}" for err in state["errors"]),
                "",
            ]
        )

    report = "\n".join(report_lines)

    # Generate conversational summary chat message for the chat workspace logs
    summary_lines = [
        "I have completed the screening cascade for the candidate resumes.",
        "",
        "**Top Shortlisted Candidates**:",
    ]
    for idx, c in enumerate(shortlist[:3]):
        status = c.get("screening_status", "").lower()
        if "reject" in status or "no-hire" in status:
            status_color = "🔴"
        elif "borderline" in status:
            status_color = "🟡"
        else:
            status_color = "🟢"
        summary_lines.append(
            f"- **#{idx + 1} {c['name']}** (Score: {c['score']}/100) — {status_color} `{c.get('screening_status', 'Shortlisted')}`"
        )

    if ranking_explanation:
        summary_lines.extend(["", "**Ranking Changes Explanation**:", ranking_explanation])

    if state.get("errors"):
        summary_lines.extend(
            [
                "",
                "⚠️ **Warnings encountered** (see deep audits or report logs for details)",
            ]
        )

    summary_lines.extend(
        [
            "",
            "*(Note: You can view full head-to-head metrics in the **Shortlist & Comparison** tab, and read deep audits in the **Deep Screening Reports** tab.)*",
        ]
    )

    ai_msg = AIMessage(content="\n".join(summary_lines))

    return {
        "final_report": report,
        "ranking_explanation": ranking_explanation,
        "messages": messages + [ai_msg],
    }


@trace_node("adjust_requirements")
def adjust_requirements_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Conversational feedback loop: refines requirements constraints using user instructions.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1].content
    current_reqs = state.get("requirements", {})

    llm = _get_llm(state, config)

    logger.info(f"Refining job requirements based on feedback: '{last_msg}'")

    system_prompt = ADJUST_REQUIREMENTS_SYSTEM_PROMPT.format(current_reqs_json=json.dumps(current_reqs, indent=2))

    def _call_adjust():
        messages_prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Update the requirements using this instruction: '{last_msg}'"),
        ]
        return invoke_structured(llm, messages_prompt, JobRequirementsOutput)

    try:
        updated_reqs = execute_with_retry(_call_adjust)
        logger.info(f"Updated requirements: {updated_reqs}")
        return {"requirements": updated_reqs, "current_round": 1}
    except Exception as e:
        logger.error(f"Failed to adjust requirements: {e}")
        return {}


@tool
def search_web_tool(query: str) -> str:
    """
    Search the web for candidate portfolios, Github repositories, technology news, or general information.
    """
    try:
        from agentic_profile_matching import config as app_config

        if not app_config.USE_MCP:
            from agentic_profile_matching.search_mcp_server import search_web

            res = search_web(query)
            if isinstance(res, dict) and "results" in res:
                return json.dumps(res["results"])
            return str(res)

        res = mcp_client.call_tool("search", "search_web", {"query": query})
        if isinstance(res, dict) and "results" in res:
            return json.dumps(res["results"])
        return str(res)
    except Exception as e:
        logger.warning(f"search_web_tool error: {e}")
        return json.dumps([{"title": f"Web results for: {query}", "snippet": f"Search lookup for '{query}'"}])


@tool
def fetch_candidate_notes_tool(candidate_name: str) -> str:
    """
    Retrieve mock HR coordinator screening notes for a specific candidate name.
    """
    try:
        from agentic_profile_matching import config as app_config

        if not app_config.USE_MCP:
            from agentic_profile_matching.search_mcp_server import fetch_candidate_notes

            res = fetch_candidate_notes(candidate_name)
            if isinstance(res, dict) and "notes" in res:
                return res["notes"]
            return str(res)

        res = mcp_client.call_tool("search", "fetch_candidate_notes", {"candidate_name": candidate_name})
        if isinstance(res, dict) and "notes" in res:
            return res["notes"]
        return str(res)
    except Exception as e:
        logger.warning(f"fetch_candidate_notes_tool error: {e}")
        return f"No coordinator notes found for '{candidate_name}'."


@trace_node("conversational_query")
def conversational_query_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    Directly answers conversational questions from the recruiter (e.g. comparison queries, rankings explanation)
    using the active candidate shortlist data, with optional web search and candidate notes tools.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1].content
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})

    llm = _get_llm(state, config)

    logger.info(f"Executing Conversational Query with search tools: '{last_msg}'")

    tools = [search_web_tool, fetch_candidate_notes_tool]
    llm_with_tools = llm.bind_tools(tools)

    import datetime

    current_date = datetime.date.today().strftime("%B %d, %Y")

    sys_prompt = (
        CONVERSATIONAL_QUERY_SYSTEM_PROMPT.format(
            reqs_json=json.dumps(requirements, indent=2),
            shortlist_json=json.dumps(shortlist, indent=2),
        )
        + f"\n\nToday's Date: {current_date}.\nYou have access to search tools. Use them ONLY if the user asks for external information (like web search or candidate notes) not present in the current candidate shortlist."
    )

    local_messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Answer this query: '{last_msg}'"),
    ]

    try:
        # ReAct style tool execution loop (max 3 iterations)
        for attempt in range(3):

            def _call_llm():
                return llm_with_tools.invoke(local_messages)

            response = execute_with_retry(_call_llm)
            local_messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]

                logger.info(f"Agent requested tool call: {tool_name} with args {args}")

                if tool_name == "search_web_tool":
                    result = search_web_tool.invoke(args)
                elif tool_name == "fetch_candidate_notes_tool":
                    result = fetch_candidate_notes_tool.invoke(args)
                else:
                    result = f"Error: Tool '{tool_name}' not found."

                from langchain_core.messages import ToolMessage

                local_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        ai_msg = AIMessage(content=local_messages[-1].content)
        return {"messages": messages + [ai_msg]}
    except Exception as e:
        logger.error(f"Failed to execute conversational query: {e}")
        err_msg = AIMessage(content=f"I encountered an error trying to analyze that: {str(e)}")
        return {"messages": messages + [err_msg]}
