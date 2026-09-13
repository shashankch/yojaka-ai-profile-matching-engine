from typing import TypedDict, List, Dict, Optional
from langchain_core.messages import BaseMessage


class JobRequirements(TypedDict, total=False):
    title: str
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    skill_expansions: Dict[str, List[str]]
    min_experience_years: int
    education_level: str
    other_constraints: List[str]


class CandidateMatch(TypedDict, total=False):
    candidate_id: str
    name: str
    score: float
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
