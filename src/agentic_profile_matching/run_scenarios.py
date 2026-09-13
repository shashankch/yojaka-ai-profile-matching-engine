import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()
from agentic_profile_matching.matching_agent import matching_agent_workflow  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

# Scenario JD
TEST_JD = """
Job Description: Senior DevOps and Cloud Infrastructure Architect
Requirements:
- Must have 5+ years of experience
- Expert knowledge of AWS cloud provider
- Strong experience with Terraform (Infrastructure as Code)
- Experience with container orchestration using Docker and Kubernetes
"""


def run_scenarios():
    print("=" * 65)
    print("RUNNING AGENTIC PROFILE MATCHING SCENARIOS")
    print("=" * 65)

    # 1. Initialize State
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("Error: GROQ_API_KEY not found in env.")
        return

    common_config = {
        "configurable": {
            "thread_id": "scenario-test-thread",
            "api_key": api_key,
            "llm_provider": "Groq",
            "llm_model": "openai/gpt-oss-120b",
        }
    }

    state = {
        "messages": [HumanMessage(content=TEST_JD)],
        "requirements": {},
        "shortlist": [],
        "coarse_screen_limit": 10,
        "deep_screen_limit": 10,
        "recommendation_limit": 5,
        "current_round": 1,
        "final_report": "",
        "feedback_pending": False,
        "user_feedback": "",
    }

    # Run Scenario 1, 2, 3, 4: Initial JD parsing and full cascading screening
    print("\n--- SCENARIOS 1-4: Raw JD Ingestion & Cascading Screening (Rounds 1, 2 & 3) ---")
    result = matching_agent_workflow.invoke(state, config=common_config)

    print("\n[Scenario 1] Extracted Job Requirements:")
    print(json_format(result["requirements"]))

    print(f"\n[Scenarios 2-4] Top Shortlisted Candidates (Total: {len(result['shortlist'])}):")
    for idx, c in enumerate(result["shortlist"][:5]):  # Print top 5
        print(f"  {idx + 1}. {c['name']} (Score: {c['score']}/100) - Status: {c.get('screening_status')}")
        print(f"     Experience: {c['experience_years']} yrs | Education: {c['education']}")
        if c.get("strengths"):
            print(f"     Strengths: {c['strengths']}")
        if c.get("gaps"):
            print(f"     Gaps: {c['gaps']}")
        if c.get("interview_questions"):
            print(f"     Questions: {c['interview_questions'][:2]}")

    # Run Scenario 5: User Feedback Refinement Loop
    print("\n--- SCENARIO 5: User Feedback & Dynamic Requirements Refinement ---")
    feedback_instruction = "Focus more on AWS cloud deployment skills and raise minimum required experience to 4 years."
    print(f"Adding user feedback message: '{feedback_instruction}'")

    state_refinement = {
        "messages": result["messages"] + [HumanMessage(content=feedback_instruction)],
        "requirements": result["requirements"],
        "shortlist": result["shortlist"],
        "coarse_screen_limit": 10,
        "deep_screen_limit": 10,
        "recommendation_limit": 5,
        "current_round": 1,
        "final_report": result["final_report"],
        "feedback_pending": False,
        "user_feedback": "",
    }

    result_ref = matching_agent_workflow.invoke(state_refinement, config=common_config)

    print("\n[Scenario 5] Updated Job Requirements:")
    print(json_format(result_ref["requirements"]))

    print("\n[Scenario 5] Ranking Changes Explanation:")
    print(result_ref.get("ranking_explanation", "No explanation generated."))

    print(f"\n[Scenario 5] Candidates Shortlisted after Refinement (Total: {len(result_ref['shortlist'])}):")
    for idx, c in enumerate(result_ref["shortlist"][:5]):  # Print top 5
        print(f"  {idx + 1}. {c['name']} (Score: {c['score']}/100) - Status: {c.get('screening_status')}")
        print(f"     Matched Skills: {c.get('matched_skills')}")
        print(f"     Missing Skills: {c.get('missing_skills')}")

    # Run Scenario 6: Conversational Comparison Query
    print("\n--- SCENARIO 6: Conversational Comparison/Explanation Query ---")
    query_msg = "Why did Michael Lee rank higher than Bruce Wayne? Contrast their skills and experience."
    print(f"Adding user query: '{query_msg}'")

    state_query = {
        "messages": result_ref["messages"] + [HumanMessage(content=query_msg)],
        "requirements": result_ref["requirements"],
        "shortlist": result_ref["shortlist"],
        "coarse_screen_limit": 10,
        "deep_screen_limit": 10,
        "recommendation_limit": 5,
        "current_round": 1,
        "final_report": result_ref["final_report"],
        "feedback_pending": False,
        "user_feedback": "",
    }

    result_query = matching_agent_workflow.invoke(state_query, config=common_config)
    print("\n[Scenario 6] Agent Response:")
    print(result_query["messages"][-1].content)


def json_format(d):
    return json.dumps(d, indent=2)


if __name__ == "__main__":
    import json

    run_scenarios()
