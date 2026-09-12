from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from agentic_profile_matching.agent.routers import (
    route_input,
    _classify_via_semantic_similarity,
    _classify_via_llm,
)
from agentic_profile_matching.agent.state import AgentState


def test_route_input_empty_messages():
    state: AgentState = {"messages": [], "requirements": {}}
    assert route_input(state) == "extract_requirements"


def test_route_input_raw_job_description():
    jd_text = """
    Software Engineer - Backend
    Requirements:
    - 5+ years Python experience
    - Microservices and Docker/Kubernetes
    - SQL database design
    """
    state: AgentState = {
        "messages": [HumanMessage(content=jd_text)],
        "requirements": {},
    }
    assert route_input(state) == "extract_requirements"


def test_route_input_semantic_search():
    state: AgentState = {
        "messages": [HumanMessage(content="Search for candidates with Java and Python Experience")],
        "requirements": {"title": "Old Search"},
        "shortlist": [],
    }
    assert route_input(state) == "extract_requirements"


def test_route_input_semantic_adjust():
    state: AgentState = {
        "messages": [
            HumanMessage(content="Make Python a mandatory must-have skill and increase experience to 7 years")
        ],
        "requirements": {"title": "Python Developer", "min_experience_years": 3},
        "shortlist": [{"candidate_id": "1", "name": "Alice Smith", "score": 90}],
    }
    assert route_input(state) == "adjust_requirements"


def test_route_input_semantic_conversational():
    state: AgentState = {
        "messages": [HumanMessage(content="Why is candidate Alice Smith ranked higher than Bob?")],
        "requirements": {"title": "Python Developer"},
        "shortlist": [
            {"candidate_id": "1", "name": "Alice Smith", "score": 90},
            {"candidate_id": "2", "name": "Bob Jones", "score": 75},
        ],
    }
    assert route_input(state) == "conversational_query"


def test_classify_via_semantic_similarity_direct():
    intent = _classify_via_semantic_similarity("Looking for a full stack developer with React and Node.js")
    assert intent == "extract_requirements"

    intent_adjust = _classify_via_semantic_similarity("Exclude candidates without Docker experience")
    assert intent_adjust == "adjust_requirements"


@patch("agentic_profile_matching.tools.invoke_structured")
@patch("agentic_profile_matching.config.get_llm_model")
def test_classify_via_llm(mock_get_llm, mock_invoke_structured):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    mock_invoke_structured.return_value = {
        "intent": "conversational_query",
        "reasoning": "User is asking for comparison between candidates.",
    }

    state: AgentState = {
        "messages": [HumanMessage(content="Break down candidate strengths and differences")],
        "requirements": {"title": "Lead Dev"},
        "shortlist": [{"candidate_id": "1", "name": "Alice", "score": 90}],
        "api_key": "dummy-key",
    }

    decision = _classify_via_llm(state)
    assert decision == "conversational_query"


def test_route_input_general_tech_query_with_empty_requirements():
    state: AgentState = {
        "messages": [HumanMessage(content="can you tell me about graph enginnering in 2026")],
        "requirements": {},
        "shortlist": [],
    }
    assert route_input(state) == "conversational_query"


def test_route_input_web_search_query():
    state: AgentState = {
        "messages": [HumanMessage(content="search google for python 3.14 features")],
        "requirements": {},
        "shortlist": [],
    }
    assert route_input(state) == "conversational_query"


def test_route_input_explicit_candidate_sourcing():
    state: AgentState = {
        "messages": [HumanMessage(content="Search resumes for Python cloud architects with 5+ years experience")],
        "requirements": {},
        "shortlist": [],
    }
    assert route_input(state) == "extract_requirements"


@patch("agentic_profile_matching.tools.invoke_structured")
def test_generate_dynamic_intent_anchors(mock_invoke_structured):
    mock_invoke_structured.return_value = {
        "extract_requirements": ["Find candidates with Go", "Source Java architects"],
        "adjust_requirements": ["Add Rust to skills", "Require 5+ years"],
        "conversational_query": ["What is LangGraph?", "Compare top profiles"],
    }
    from agentic_profile_matching.agent.routers import generate_dynamic_intent_anchors

    mock_llm = MagicMock()
    anchors = generate_dynamic_intent_anchors(mock_llm)
    assert "extract_requirements" in anchors
    assert len(anchors["extract_requirements"]) == 2
    assert "conversational_query" in anchors
