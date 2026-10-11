"""Unit tests for ParentDocumentService and ParentDocumentStore."""

from agentic_profile_matching.services.parent_document_service import (
    ParentDocumentService,
    ParentDocumentStore,
)


def test_parent_document_store_crud():
    store = ParentDocumentStore()
    store.clear()

    store.save_parent_chunk(
        "parent_001", "Full resume content for candidate Alice with 10 years experience", candidate_id="cand_1"
    )
    assert store.get_parent_chunk("parent_001") is not None
    assert "Alice" in store.get_parent_chunk("parent_001")
    assert store.get_parent_chunk("non_existent") is None
    assert len(store.get_all()) == 1

    store.clear()
    assert len(store.get_all()) == 0


def test_parent_document_service_hierarchical_chunking():
    service = ParentDocumentService(
        parent_chunk_size=300,
        parent_chunk_overlap=50,
        child_chunk_size_tokens=40,
        child_chunk_overlap_tokens=10,
    )

    sample_doc = """
    EXPERIENCE
    Senior Backend Architect at CloudCorp (2020 - Present)
    Designed distributed streaming architectures using Kafka, Go, and Python.
    Orchestrated multi-region Kubernetes deployments handling 50k requests per second.
    Led a team of 12 platform engineers and maintained 99.99% system availability.

    SKILLS & TECHNOLOGIES
    Languages: Python, Go, Rust, TypeScript
    Cloud & Infrastructure: AWS, GCP, Docker, Kubernetes, Terraform
    Databases: PostgreSQL, Redis, ScyllaDB, ChromaDB

    EDUCATION
    B.S. in Computer Science, University of Technology, 2018
    """

    parent_chunks, child_chunks = service.decompose_document(sample_doc, candidate_id="cand_alice")

    assert len(parent_chunks) > 0
    assert len(child_chunks) >= len(parent_chunks)

    # Every child chunk must possess a valid parent_id referencing an existing parent chunk
    parent_ids = {p.parent_id for p in parent_chunks}
    for child in child_chunks:
        assert child.metadata.get("parent_id") in parent_ids
        assert len(child.content.strip()) > 0


def test_enrich_matches_with_parent_context():
    service = ParentDocumentService()
    store = service.parent_store
    store.clear()

    # Pre-populate store
    store.save_parent_chunk(
        "p_101",
        "Deep Section: Architecture & Performance Tuning. Spearheaded 10x database read speedup.",
        candidate_id="cand_bob",
    )

    matches = [
        {
            "candidate_id": "cand_bob",
            "metadata": {"parent_id": "p_101"},
            "raw_text": "read speedup",
        },
        {
            "candidate_id": "cand_carol",
            "metadata": {},
            "raw_text": "Frontend React dev",
        },
    ]

    enriched = service.enrich_candidate_matches(matches)

    assert (
        enriched[0]["full_parent_context"]
        == "Deep Section: Architecture & Performance Tuning. Spearheaded 10x database read speedup."
    )
    assert "full_parent_context" not in enriched[1] or enriched[1]["full_parent_context"] == ""
