from typing import Dict, Any, Optional
from agentic_profile_matching import config
from agentic_profile_matching.celery_app import celery_app
from agentic_profile_matching.services.ingestion_service import IngestionService
from agentic_profile_matching.fs_client import read_file
from agentic_profile_matching.tools import invoke_structured, DeepScreenOutput, execute_with_retry
from agentic_profile_matching.agent.prompts import DEEP_SCREEN_SYSTEM_PROMPT
from agentic_profile_matching.agent.nodes import _get_llm
from langchain_core.messages import SystemMessage, HumanMessage


@celery_app.task(name="tasks.async_ingest_directory")
def async_ingest_directory(directory_path: str) -> Dict[str, Any]:
    """
    Celery background worker task for ingesting a directory of candidate resumes.
    """
    service = IngestionService()
    return service.ingest_directory(directory_path)


@celery_app.task(name="tasks.async_deep_screen_candidate")
def async_deep_screen_candidate(
    candidate_id: str,
    candidate_name: str,
    requirements: Dict[str, Any],
    config_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Celery background worker task for asynchronous deep screening audit on a candidate resume.
    """
    res = read_file(candidate_id)
    if not res or not res.get("success"):
        return {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "strengths": ["Semantic match based on vector DB indexing"],
            "gaps": ["Could not audit text (file unreadable / unparsed)"],
            "improvement_suggestions": "Review resume file formatting before interviewing candidate.",
            "screening_status": "Screened",
            "screening_reasoning": f"Fallback screening (unreadable file: {res.get('error') if res else 'Unknown error'})",
        }

    resume_text = res["content"]
    # Truncate content to avoid token exhaustion
    if len(resume_text) > config.RESUME_TRUNCATION_LIMIT:
        resume_text = resume_text[: config.RESUME_TRUNCATION_LIMIT] + "... [truncated]"

    prompt_content = f"""Candidate: {candidate_name}
Job Title: {requirements.get("title", "Software Engineer")}
Must-Have Skills: {requirements.get("must_have_skills", [])}
Nice-to-Have Skills: {requirements.get("nice_to_have_skills", [])}

Resume Content:
{resume_text}"""

    try:
        llm = _get_llm({}, config_dict)
        if llm is None:
            raise ValueError("LLM provider unconfigured")

        def _call():
            messages = [
                SystemMessage(content=DEEP_SCREEN_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content),
            ]
            return invoke_structured(llm, messages, DeepScreenOutput)

        result = execute_with_retry(_call)
        result["candidate_id"] = candidate_id
        result["candidate_name"] = candidate_name
        return result
    except Exception as e:
        return {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "strengths": ["Semantic match based on vector DB indexing"],
            "gaps": ["Skipped deep screening audit due to LLM error"],
            "improvement_suggestions": "Schedule interview to evaluate candidate skills directly.",
            "screening_status": "Screened",
            "screening_reasoning": f"Fallback screening due to LLM error: {str(e)}",
        }
