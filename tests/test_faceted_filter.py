"""Unit tests for FacetedFilter (pre-retrieval metadata constraint evaluation)."""

from agentic_profile_matching.services.faceted_filter import FacetedFilter


def test_faceted_filter_experience():
    ff = FacetedFilter(min_experience_years=5.0)

    cand_pass = {"candidate_id": "c1", "metadata": {"experience_years": 7.0}}
    cand_fail = {"candidate_id": "c2", "metadata": {"experience_years": 3.0}}
    cand_unknown = {"candidate_id": "c3", "metadata": {}}

    assert ff.evaluate_candidate(cand_pass) is True
    assert ff.evaluate_candidate(cand_fail) is False
    assert ff.evaluate_candidate(cand_unknown) is True  # Unknown preserved for downstream scoring


def test_faceted_filter_education():
    ff = FacetedFilter(education_levels=["bachelor", "master", "phd"])

    cand_bachelor = {"candidate_id": "c1", "metadata": {"education": "B.S. in Computer Science"}}
    cand_highschool = {"candidate_id": "c2", "metadata": {"education": "High School Diploma"}}
    cand_none = {"candidate_id": "c3", "metadata": {}}

    assert ff.evaluate_candidate(cand_bachelor) is True
    assert ff.evaluate_candidate(cand_highschool) is False
    assert ff.evaluate_candidate(cand_none) is True


def test_faceted_filter_must_have_skills_with_expansions():
    ff = FacetedFilter(
        must_have_skills=["Kubernetes", "Python"],
        skill_expansions={"kubernetes": ["k8s", "helm", "eks"]},
    )

    # Has Python and k8s (synonym of Kubernetes)
    cand_1 = {
        "candidate_id": "c1",
        "metadata": {
            "skills": ["python", "k8s", "fastapi"],
            "experience_years": 4.0,
        },
    }
    # Has only Python, missing Kubernetes/k8s
    cand_2 = {
        "candidate_id": "c2",
        "metadata": {
            "skills": ["python", "django"],
            "experience_years": 4.0,
        },
    }

    assert ff.evaluate_candidate(cand_1) is True
    assert ff.evaluate_candidate(cand_2) is False

    filtered = ff.filter_candidates([cand_1, cand_2])
    assert len(filtered) == 1
    assert filtered[0]["candidate_id"] == "c1"
