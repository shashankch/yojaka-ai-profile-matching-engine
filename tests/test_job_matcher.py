import time
from unittest.mock import MagicMock, patch
from agentic_profile_matching.job_matcher import JobMatcher
from agentic_profile_matching.stores import QdrantVectorStore, BaseVectorStore


@patch("agentic_profile_matching.job_matcher.SentenceTransformer")
@patch("agentic_profile_matching.job_matcher.ChromaVectorStore")
def test_job_matcher_match(mock_chroma_store_cls, mock_sentence_transformer):
    # Mock Chroma store setup
    mock_store = MagicMock()
    mock_chroma_store_cls.return_value = mock_store

    # Mock SentenceTransformer setup
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_sentence_transformer.return_value = mock_embedder

    # Define mock DB documents, metadatas, ids
    mock_store.get_all.return_value = {
        "documents": [
            "Experienced Python developer working with Docker and microservices.",
            "Java engineer with experience in Spring Boot.",
        ],
        "metadatas": [
            {
                "candidate_name": "Alice Smith",
                "resume_path": "alice.pdf",
                "experience_years": 5,
                "skills": "Python, Docker, Microservices",
                "education": "Master",
                "section": "WORK EXPERIENCE",
            },
            {
                "candidate_name": "Bob Jones",
                "resume_path": "bob.pdf",
                "experience_years": 2,
                "skills": "Java, Spring Boot",
                "education": "Bachelor",
                "section": "WORK EXPERIENCE",
            },
        ],
        "ids": ["id1", "id2"],
    }

    # Mock query result (Alice is closer to query, lower distance)
    mock_store.query.return_value = {
        "ids": [["id1", "id2"]],
        "distances": [[0.2, 0.8]],
    }

    # Instantiate JobMatcher
    matcher = JobMatcher()

    # Run match with min_exp=3 and Python filter
    res = matcher.match(
        job_description="Python developer with 3+ years experience.",
        k=10,
        min_exp=3,
        must_have_skills=["Python"],
        apply_filters=True,
    )

    assert len(res["top_matches"]) == 1  # Bob has 2 years exp and no Python, so filtered out
    match = res["top_matches"][0]
    assert match["candidate_name"] == "Alice Smith"
    assert match["match_score"] > 50
    assert "Python" in match["matched_skills"]


@patch("agentic_profile_matching.job_matcher.SentenceTransformer")
def test_job_matcher_bm25_caching(mock_sentence_transformer):
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2]
    mock_sentence_transformer.return_value = mock_embedder

    mock_store = MagicMock()
    mock_store.get_all.return_value = {
        "documents": ["Python developer", "Java developer"],
        "metadatas": [
            {"candidate_name": "Alice", "experience_years": 5, "skills": "Python", "resume_path": "a.pdf"},
            {"candidate_name": "Bob", "experience_years": 3, "skills": "Java", "resume_path": "b.pdf"},
        ],
        "ids": ["1", "2"],
    }
    mock_store.query.return_value = {"ids": [["1", "2"]], "distances": [[0.1, 0.9]]}

    matcher = JobMatcher(store=mock_store)

    start1 = time.perf_counter()
    matcher.match("Python developer")
    dur1 = time.perf_counter() - start1

    bm25_instance1 = matcher._cached_bm25
    assert bm25_instance1 is not None

    start2 = time.perf_counter()
    matcher.match("Python developer")
    dur2 = time.perf_counter() - start2

    # Second call uses cached BM25 index instance
    assert matcher._cached_bm25 is bm25_instance1
    assert dur2 <= dur1 + 0.05


def test_job_matcher_custom_store_injection():
    qdrant_stub = QdrantVectorStore()
    assert isinstance(qdrant_stub, BaseVectorStore)

    qdrant_stub.upsert(
        ids=["chunk_1"],
        documents=["Senior Backend Engineer with Golang experience."],
        embeddings=[[0.1] * 384],
        metadatas=[
            {
                "candidate_name": "Charlie",
                "experience_years": 7,
                "skills": "Go, Golang",
                "education": "BS CS",
                "resume_path": "charlie.pdf",
                "section": "EXPERIENCE",
            }
        ],
    )

    with patch("agentic_profile_matching.job_matcher.SentenceTransformer") as mock_st:
        mock_st.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        matcher = JobMatcher(store=qdrant_stub)
        res = matcher.match("Golang engineer with 5+ years", min_exp=5)
        assert len(res["top_matches"]) == 1
        assert res["top_matches"][0]["candidate_name"] == "Charlie"


def test_job_matcher_skill_expansions():
    mock_store = MagicMock()
    mock_store.get_all.return_value = {
        "documents": [
            "Senior Infrastructure Architect with AWS and Kubernetes experience.",
            "Junior Frontend Developer with HTML and CSS.",
        ],
        "metadatas": [
            {
                "candidate_name": "DevOps Expert",
                "resume_path": "devops.pdf",
                "experience_years": 6,
                "skills": "AWS, Kubernetes, Docker",
                "education": "BS",
                "section": "WORK EXPERIENCE",
            },
            {
                "candidate_name": "Junior Web",
                "resume_path": "web.pdf",
                "experience_years": 1,
                "skills": "HTML, CSS",
                "education": "BA",
                "section": "WORK EXPERIENCE",
            },
        ],
        "ids": ["d1", "d2"],
    }
    mock_store.query.return_value = {"ids": [["d1", "d2"]], "distances": [[0.2, 0.9]]}

    with patch("agentic_profile_matching.job_matcher.SentenceTransformer") as mock_st:
        mock_st.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        matcher = JobMatcher(store=mock_store)

        # Candidate has AWS, query requires Cloud. With expansion, should match!
        res = matcher.match(
            job_description="Seeking candidate with Cloud experience",
            must_have_skills=["Cloud"],
            skill_expansions={"Cloud": ["AWS", "GCP", "Azure"]},
            apply_filters=True,
        )

        assert len(res["top_matches"]) == 1
        assert res["top_matches"][0]["candidate_name"] == "DevOps Expert"
        assert "Cloud" in res["top_matches"][0]["matched_skills"]


def test_job_matcher_word_boundary_skill_matching():
    mock_store = MagicMock()
    mock_store.get_all.return_value = {
        "documents": ["Frontend developer with JavaScript and TypeScript experience."],
        "metadatas": [
            {
                "candidate_name": "JS Dev",
                "resume_path": "js.pdf",
                "experience_years": 4,
                "skills": "JavaScript, TypeScript",
                "education": "BS",
                "section": "SKILLS",
            }
        ],
        "ids": ["id_js"],
    }
    mock_store.query.return_value = {"ids": [["id_js"]], "distances": [[0.5]]}

    with patch("agentic_profile_matching.job_matcher.SentenceTransformer") as mock_st:
        mock_st.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        matcher = JobMatcher(store=mock_store)

        # Requirement is "Java", candidate only has "JavaScript". Must NOT match!
        res = matcher.match(
            job_description="Seeking Java engineer",
            must_have_skills=["Java"],
            apply_filters=True,
        )

        assert len(res["top_matches"]) == 0


def test_job_matcher_cache_invalidated_on_document_text_update():
    mock_store = MagicMock()
    mock_store.get_all.return_value = {
        "documents": ["Original resume text version 1"],
        "metadatas": [{"candidate_name": "Alice", "experience_years": 3, "skills": "Python", "resume_path": "a.pdf"}],
        "ids": ["id_1"],
    }
    mock_store.query.return_value = {"ids": [["id_1"]], "distances": [[0.1]]}

    with patch("agentic_profile_matching.job_matcher.SentenceTransformer") as mock_st:
        mock_st.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        matcher = JobMatcher(store=mock_store)

        matcher.match("Python")
        initial_bm25 = matcher._cached_bm25

        # Now simulate document content update with the SAME id
        mock_store.get_all.return_value = {
            "documents": ["Completely revised resume text version 2 with Go and Rust"],
            "metadatas": [
                {"candidate_name": "Alice", "experience_years": 4, "skills": "Go, Rust", "resume_path": "a.pdf"}
            ],
            "ids": ["id_1"],
        }

        matcher.match("Go")
        updated_bm25 = matcher._cached_bm25

        # The cache MUST be rebuilt with new corpus
        assert initial_bm25 is not updated_bm25


def test_job_matcher_score_clamping():
    mock_store = MagicMock()
    mock_store.get_all.return_value = {
        "documents": ["Python specialist"],
        "metadatas": [{"candidate_name": "Dev", "experience_years": 10, "skills": "Python", "resume_path": "dev.pdf"}],
        "ids": ["d1"],
    }
    mock_store.query.return_value = {"ids": [["d1"]], "distances": [[0.0]]}

    with patch("agentic_profile_matching.job_matcher.SentenceTransformer") as mock_st:
        mock_st.return_value.encode.return_value.tolist.return_value = [0.1] * 384
        matcher = JobMatcher(store=mock_store)

        res = matcher.match("Python developer with 5+ years experience")
        assert len(res["top_matches"]) == 1
        score = res["top_matches"][0]["match_score"]
        assert 0.0 <= score <= 100.0
