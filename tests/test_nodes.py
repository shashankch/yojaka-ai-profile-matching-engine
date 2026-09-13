from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from agentic_profile_matching.agent.nodes import (
    parse_input_node,
    extract_requirements_node,
    search_resumes_node,
    rank_candidates_node,
    deep_screen_node,
    recommendation_node,
    generate_report_node,
    adjust_requirements_node,
    conversational_query_node,
)
from agentic_profile_matching.agent.state import AgentState


def test_parse_input_node():
    state: AgentState = {
        "messages": [HumanMessage(content="Looking for Python Engineer")],
        "shortlist": [{"candidate_id": "1", "name": "Alice", "score": 90}],
        "errors": [],
    }
    res = parse_input_node(state)
    assert res["current_round"] == 1
    assert len(res["previous_shortlist"]) == 1


def test_extract_requirements_node():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"title": "Senior Python Dev", "must_have_skills": ["Python"]}'

    state: AgentState = {
        "messages": [HumanMessage(content="Job Description:\nPython Developer with 5+ years experience")],
        "requirements": {},
    }
    config_dict = {"configurable": {"llm": mock_llm}}

    res = extract_requirements_node(state, config_dict)
    assert res["current_round"] == 1
    assert "requirements" in res
    assert res["requirements"]["title"] == "Senior Python Dev"


def test_search_resumes_node():
    mock_store = MagicMock()
    mock_store.get_all.return_value = {
        "documents": ["Alice Smith resume text with Python experience"],
        "metadatas": [
            {
                "filename": "resume_alice.txt",
                "candidate_name": "Alice Smith",
                "skills": "Python",
                "experience_years": 5,
                "education": "BS",
                "resume_path": "/path/resume_alice.txt",
            }
        ],
        "ids": ["resume_alice.txt_general_0"],
    }
    mock_store.query.return_value = {
        "documents": [["Alice Smith resume text with Python experience"]],
        "metadatas": [
            [
                {
                    "filename": "resume_alice.txt",
                    "candidate_name": "Alice Smith",
                    "skills": "Python",
                    "experience_years": 5,
                    "education": "BS",
                    "resume_path": "/path/resume_alice.txt",
                }
            ]
        ],
        "ids": [["resume_alice.txt_general_0"]],
        "distances": [[0.1]],
    }

    state: AgentState = {
        "messages": [HumanMessage(content="Python Developer")],
        "requirements": {"title": "Python Dev", "must_have_skills": ["Python"]},
        "coarse_screen_limit": 5,
    }
    config_dict = {"configurable": {"store": mock_store}}

    res = search_resumes_node(state, config_dict)
    assert "shortlist" in res
    assert len(res["shortlist"]) > 0


def test_rank_candidates_node():
    state: AgentState = {
        "shortlist": [
            {
                "resume_path": "/path/alice.txt",
                "candidate_name": "Alice",
                "match_score": 90,
                "matched_skills": ["Python"],
            },
            {
                "resume_path": "/path/bob.txt",
                "candidate_name": "Bob",
                "match_score": 70,
                "matched_skills": ["Java"],
            },
        ],
        "requirements": {"title": "Python Dev", "must_have_skills": ["Python"]},
    }
    res = rank_candidates_node(state)
    assert res["shortlist"][0]["name"] == "Alice"


def test_deep_screen_node_fallback():
    state: AgentState = {
        "shortlist": [{"candidate_id": "nonexistent.txt", "name": "Unknown", "score": 80}],
        "deep_screen_limit": 1,
    }
    res = deep_screen_node(state)
    assert res["current_round"] == 2
    assert res["shortlist"][0]["screening_status"] == "Screened"


def test_recommendation_node():
    state: AgentState = {
        "shortlist": [
            {
                "candidate_id": "1",
                "name": "Alice",
                "score": 90,
                "matched_skills": ["Python"],
                "missing_skills": [],
                "strengths": ["Strong coding"],
                "gaps": [],
                "experience_years": 5,
                "education": "BS CS",
            }
        ],
        "requirements": {"title": "Python Dev", "must_have_skills": ["Python"]},
        "recommendation_limit": 1,
    }
    res = recommendation_node(state)
    assert res["current_round"] == 3
    assert len(res["shortlist"]) == 1


def test_generate_report_node():
    state: AgentState = {
        "shortlist": [
            {
                "candidate_id": "1",
                "name": "Alice",
                "score": 90,
                "matched_skills": ["Python"],
                "missing_skills": [],
                "screening_status": "Strong Hire",
                "experience_years": 5,
                "education": "BS",
            }
        ],
        "requirements": {"title": "Python Dev"},
        "ranking_explanation": "Alice scored highest due to Python experience.",
    }
    res = generate_report_node(state)
    assert "final_report" in res
    assert "Candidate Match & Screening Report" in res["final_report"]


def test_adjust_requirements_node():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = '{"title": "Lead Python Engineer", "must_have_skills": ["Python", "Docker"]}'

    state: AgentState = {
        "messages": [HumanMessage(content="Focus more on candidates with Docker experience")],
        "requirements": {"title": "Python Engineer", "must_have_skills": ["Python"]},
    }
    config_dict = {"configurable": {"llm": mock_llm}}

    res = adjust_requirements_node(state, config_dict)
    assert res["current_round"] == 1
    assert "requirements" in res
    assert "Docker" in res["requirements"]["must_have_skills"]


def test_conversational_query_node():
    mock_llm = MagicMock()
    mock_bound_llm = MagicMock()
    mock_response = AIMessage(
        content="Alice is a stronger candidate because she has 5 years of Python experience compared to Bob's 2 years."
    )
    mock_response.tool_calls = []
    mock_bound_llm.invoke.return_value = mock_response
    mock_llm.bind_tools.return_value = mock_bound_llm

    state: AgentState = {
        "messages": [HumanMessage(content="Why is Alice ranked higher than Bob?")],
        "shortlist": [
            {"candidate_id": "1", "name": "Alice", "score": 90},
            {"candidate_id": "2", "name": "Bob", "score": 70},
        ],
    }
    config_dict = {"configurable": {"llm": mock_llm}}

    res = conversational_query_node(state, config_dict)
    assert len(res["messages"]) == 2
    assert "Alice" in res["messages"][-1].content


def test_rank_candidates_case_insensitive_skill_expansions():
    state: AgentState = {
        "shortlist": [
            {
                "resume_path": "stream://candidate1.pdf",
                "candidate_name": "Cloud Engineer",
                "match_score": 85,
                "skills": ["AWS", "Docker"],
                "raw_text": "Experienced AWS cloud engineer",
            }
        ],
        "requirements": {
            "title": "Cloud Architect",
            "must_have_skills": ["cloud"],
            # TitleCase key in expansions
            "skill_expansions": {"Cloud": ["AWS", "GCP", "Azure"]},
        },
    }
    res = rank_candidates_node(state)
    cand = res["shortlist"][0]
    assert "cloud" in cand["matched_skills"]
    assert "cloud" not in cand["missing_skills"]
    assert cand["raw_text"] == "Experienced AWS cloud engineer"


def test_deep_screen_node_uses_in_memory_raw_text():
    # When candidate has raw_text (from stream upload), read_file is NOT called
    state: AgentState = {
        "shortlist": [
            {
                "candidate_id": "stream://uploaded_resume.pdf",
                "name": "Stream Candidate",
                "score": 90,
                "raw_text": "Jane Doe. Senior Python developer with 6 years experience.",
            }
        ],
        "deep_screen_limit": 1,
    }
    with (
        patch("agentic_profile_matching.agent.nodes.read_file") as mock_read_file,
        patch("agentic_profile_matching.agent.nodes._get_llm", return_value=None),
    ):
        res = deep_screen_node(state)
        mock_read_file.assert_not_called()
        assert res["current_round"] == 2
        assert res["shortlist"][0]["screening_status"] == "Screened"
        # Reasoning was fallback due to unconfigured LLM, NOT unreadable file
        assert "unreadable" not in res["shortlist"][0]["screening_reasoning"].lower()
