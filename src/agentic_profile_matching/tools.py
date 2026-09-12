import re
import time
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from agentic_profile_matching.exceptions import LLMParseError

logger = logging.getLogger("agentic_profile_matching.tools")


def execute_with_retry(func, *args, **kwargs):
    """
    Executes an LLM function call with exponential backoff on 429 (rate limit) or transient errors.
    """
    max_retries = 4
    delay = 1.5
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ["429", "rate_limit", "too many requests", "resource_exhausted", "quota"]):
                logger.warning(f"Rate limit hit. Retrying in {delay:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2.0
            else:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(0.5)
    return func(*args, **kwargs)


class JobRequirementsOutput(BaseModel):
    title: str = Field(default="Software Engineer", description="Extracted job title")
    must_have_skills: List[str] = Field(default_factory=list, description="Mandatory technical skills and tools")
    nice_to_have_skills: List[str] = Field(default_factory=list, description="Preferred/optional technical skills")
    skill_expansions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Dynamic semantic equivalents, cloud providers, frameworks, or child technologies for extracted skills",
    )
    min_experience_years: int = Field(default=0, description="Minimum years of professional experience required")
    education_level: str = Field(default="Not Specified", description="Required education level (e.g. B.Tech, MS, PhD)")
    other_constraints: List[str] = Field(
        default_factory=list, description="Location, visa, or certification constraints"
    )


class DeepScreenOutput(BaseModel):
    strengths: List[str] = Field(default_factory=list, description="Core technical and architectural strengths")
    gaps: List[str] = Field(default_factory=list, description="Missing required skills or experience deficits")
    improvement_suggestions: str = Field(default="", description="Actionable advice for candidate skill growth")
    screening_status: str = Field(
        default="Screened",
        description="One of: 'Strong Hire', 'Borderline Hire', 'Rejected / No-Hire'",
    )
    screening_reasoning: str = Field(
        default="", description="Short reasoning summary justifying the screening decision"
    )


class InterviewQuestionsOutput(BaseModel):
    questions: List[str] = Field(default_factory=list, description="3 to 5 targeted technical screening questions")


def _extract_json_chunk(text: str) -> Optional[str]:
    """
    Finds the first balanced outermost {...} or [...] block in raw text,
    correctly skipping brackets inside double-quoted strings.
    """
    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace == -1 and start_bracket == -1:
        return None

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_idx = start_brace
        open_char, close_char = "{", "}"
    else:
        start_idx = start_bracket
        open_char, close_char = "[", "]"

    depth = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
    return None


def parse_json_output(content: Any, model_cls: Optional[Any] = None) -> Any:
    """
    Production-grade multi-tier JSON parser for LLM outputs:
    1. Fast-path if already dict/model instance.
    2. Strips <think> reasoning tokens and markdown code fences.
    3. Direct Pydantic model_validate_json.
    4. Balanced bracket JSON chunk extraction.
    5. Cleans trailing commas and common syntax defects.
    6. Safe fallback to default model_cls instance if unparseable.
    """
    if content is None:
        return model_cls().model_dump() if model_cls is not None else {}

    if isinstance(content, dict):
        if model_cls is not None:
            return model_cls.model_validate(content).model_dump()
        return content

    if isinstance(content, BaseModel):
        return content.model_dump()

    text = str(content).strip()

    # 1. Remove <think>...</think> reasoning blocks (DeepSeek, reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Remove markdown code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
        else:
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()

    # 3. Direct model validation if clean
    if model_cls is not None:
        try:
            return model_cls.model_validate_json(text).model_dump()
        except (ValidationError, Exception):
            pass

    # 4. Extract first balanced JSON chunk
    extracted = _extract_json_chunk(text)
    if extracted:
        text = extracted

    # 5. Clean common LLM formatting defects: trailing commas, unquoted single quotes
    cleaned = re.sub(r",\s*([\]}])", r"\1", text)
    cleaned = re.sub(r"\'([a-zA-Z0-9_]+)\'\s*:", r'"\1":', cleaned)

    try:
        parsed = json.loads(cleaned)
        if model_cls is not None:
            return model_cls.model_validate(parsed).model_dump()
        return parsed
    except Exception:
        try:
            parsed = json.loads(text)
            if model_cls is not None:
                return model_cls.model_validate(parsed).model_dump()
            return parsed
        except Exception as e:
            if model_cls is not None:
                logger.warning(f"Falling back to default model schema due to JSON parse error: {e}")
                return model_cls().model_dump()
            raise LLMParseError(f"Failed to parse LLM output JSON: {e}") from e


def invoke_structured(llm, messages: List[Any], schema_cls: Any) -> Dict[str, Any]:
    """
    Invokes LLM using native with_structured_output (Gold Standard) if supported,
    with graceful fallback to standard prompt-based invocation + robust parsing.
    """
    # 1. Attempt native structured output first
    try:
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(schema_cls)
            result = structured_llm.invoke(messages)
            if isinstance(result, BaseModel):
                return result.model_dump()
            if isinstance(result, dict):
                return schema_cls.model_validate(result).model_dump()
    except Exception as e:
        logger.debug(f"Native structured output fallback triggered: {e}")

    # 2. Resilient fallback via standard invoke + parse_json_output
    response = llm.invoke(messages)
    content = getattr(response, "content", str(response))
    return parse_json_output(content, model_cls=schema_cls)


def extract_requirements(jd: str, llm) -> Dict[str, Any]:
    """
    Extracts structured job requirements from an unstructured JD string using structured output.
    """
    system_prompt = """You are a professional recruiting assistant. Analyze the provided Job Description (JD) and extract the job requirements.
For any broad skill or technology (e.g. "Cloud", "Backend", "Frontend", "Machine Learning", "DevOps", "Database"), provide a list of concrete semantic equivalents, major providers, or common frameworks in `skill_expansions` (e.g. {"Cloud": ["AWS", "GCP", "Azure", "Kubernetes", "EKS"], "Java": ["Spring Boot", "JVM", "Core Java"], "Python": ["Django", "FastAPI", "Flask"]}).
You MUST return a valid JSON object matching the requested schema."""

    def _call():
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract requirements for this Job Description:\n\n{jd}"),
        ]
        return invoke_structured(llm, messages, JobRequirementsOutput)

    try:
        return execute_with_retry(_call)
    except Exception as e:
        logger.error(f"Error in extract_requirements tool: {e}")
        return JobRequirementsOutput().model_dump()


def compare_candidates(candidate_ids: List[str], shortlist: List[Dict]) -> str:
    """
    Generates a side-by-side Markdown grid comparing selected candidates.
    """
    candidates = []
    for c_id in candidate_ids:
        matched = None
        for c in shortlist:
            if c["candidate_id"] == c_id or c["name"] == c_id or Path(c["candidate_id"]).stem == c_id:
                matched = c
                break
        if matched and matched not in candidates:
            candidates.append(matched)

    if not candidates and len(shortlist) > 0:
        candidates = shortlist[:3]

    if not candidates:
        return "No matching candidates found in the shortlist to compare."

    header_row = "| Match Category | " + " | ".join(f"**{c['name']}**" for c in candidates) + " |"
    separator_row = "|---| " + " | ".join("---" for _ in candidates) + " |"

    rows = [
        ("Match Score", lambda c: f"**{c.get('score', 0)}/100**"),
        ("Experience", lambda c: f"{c.get('experience_years', 0)} Years"),
        ("Education", lambda c: f"{c.get('education', 'Not Specified')}"),
        (
            "Matched Skills",
            lambda c: ", ".join(c.get("matched_skills", [])) if c.get("matched_skills") else "None",
        ),
        (
            "Missing Core Skills",
            lambda c: ", ".join(c.get("missing_skills", [])) if c.get("missing_skills") else "None",
        ),
        (
            "Decision Status",
            lambda c: (
                ":green[**Strong Hire**]"
                if "strong" in c.get("screening_status", "").lower()
                else ":orange[**Borderline Hire**]"
                if "borderline" in c.get("screening_status", "").lower()
                else ":red[**Rejected / No-Hire**]"
                if "reject" in c.get("screening_status", "").lower()
                or "no-hire" in c.get("screening_status", "").lower()
                else f"`{c.get('screening_status', 'Screened')}`"
            ),
        ),
        (
            "Core Strengths",
            lambda c: "; ".join(c.get("strengths", [])) if c.get("strengths") else "Not Specified",
        ),
        (
            "Identified Gaps",
            lambda c: "; ".join(c.get("gaps", [])) if c.get("gaps") else "None",
        ),
    ]

    table_lines = [header_row, separator_row]
    for label, extractor_fn in rows:
        line = f"| **{label}** | " + " | ".join(extractor_fn(c) for c in candidates) + " |"
        table_lines.append(line)

    return "\n".join(table_lines)


def generate_interview_questions(
    candidate_name: str,
    skills: List[str],
    gaps: List[str],
    requirements: Dict[str, Any],
    llm,
) -> List[str]:
    """
    Generates 3-5 technical questions tailored to probe the candidate's gaps.
    Supports both native structured object {"questions": [...]} and JSON list [...] formats.
    """
    system_prompt = """You are a technical interviewer. Design 3 to 5 customized screening questions for a candidate.
Focus on probing their skills gaps, validating their experience, and assessing alignment with the job requirements.
Return a JSON object: {"questions": ["Question 1...", "Question 2..."]} or a JSON list of strings."""

    prompt_content = f"""Candidate: {candidate_name}
Job Title: {requirements.get("title", "Software Engineer")}
Must-Have Skills: {requirements.get("must_have_skills", [])}
Candidate Skills: {skills}
Identified Gaps: {gaps}

Generate 3-5 targeted interview questions."""

    def _call():
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt_content),
        ]
        # Try native structured output with InterviewQuestionsOutput
        try:
            if hasattr(llm, "with_structured_output"):
                structured_llm = llm.with_structured_output(InterviewQuestionsOutput)
                res = structured_llm.invoke(messages)
                if isinstance(res, InterviewQuestionsOutput) and res.questions:
                    return res.questions
                if isinstance(res, dict) and res.get("questions"):
                    return res["questions"]
        except Exception:
            pass

        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
        parsed = parse_json_output(content)
        if isinstance(parsed, list) and parsed:
            return parsed
        if isinstance(parsed, dict) and parsed.get("questions"):
            return parsed["questions"]

        return [
            f"Can you walk me through your hands-on experience with {', '.join(skills[:3]) if skills else 'software engineering'}?",
            f"How have you tackled technical challenges regarding {', '.join(gaps[:2]) if gaps else 'system design'}?",
            "What is your approach to learning and adopting new technology stacks quickly?",
        ]

    try:
        return execute_with_retry(_call)
    except Exception as e:
        logger.error(f"Error in generate_interview_questions tool: {e}")
        return [
            f"Can you tell me about your experience working with {', '.join(gaps) if gaps else 'software engineering architectures'}?",
            "Can you walk me through a complex technical project you engineered and the design tradeoffs you made?",
            "What strategies do you use when adapting to new tech stacks or programming paradigms on the job?",
        ]
