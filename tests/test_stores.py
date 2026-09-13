from agentic_profile_matching.stores import (
    BaseVectorStore,
    ChromaVectorStore,
    QdrantVectorStore,
    CompositeVectorStore,
)
from agentic_profile_matching.resume_rag import ResumeRAGPipeline
from agentic_profile_matching.services.ingestion_service import IngestionService


def test_chroma_store_implements_protocol(tmp_path):
    store = ChromaVectorStore(collection_name="test_resumes", db_path=str(tmp_path / "chroma_db"))
    assert isinstance(store, BaseVectorStore)


def test_qdrant_store_implements_protocol():
    store = QdrantVectorStore(collection_name="test_resumes")
    assert isinstance(store, BaseVectorStore)


def test_composite_store_implements_protocol(tmp_path):
    base_store = ChromaVectorStore(collection_name="base_resumes", db_path=str(tmp_path / "chroma_db"))
    eph_store = ChromaVectorStore(collection_name="eph_resumes", ephemeral=True)
    comp_store = CompositeVectorStore(base_store=base_store, ephemeral_store=eph_store)
    assert isinstance(comp_store, BaseVectorStore)


def test_chroma_store_operations_and_idempotency(tmp_path):
    store = ChromaVectorStore(collection_name="test_resumes", db_path=str(tmp_path / "chroma_db"))

    # Initial state
    assert store.count() == 0

    ids = ["res_1_exp_0", "res_1_skills_1"]
    docs = ["5 years Python experience", "Skills: Python, Docker, Kubernetes"]
    embeddings = [[0.1] * 384, [0.2] * 384]
    metadatas = [
        {"candidate_name": "John Doe", "section": "EXPERIENCE", "filename": "john.pdf"},
        {"candidate_name": "John Doe", "section": "SKILLS", "filename": "john.pdf"},
    ]

    # First upsert
    store.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    assert store.count() == 2

    # Query
    res = store.query(query_embedding=[0.1] * 384, n_results=1)
    assert "documents" in res
    assert len(res["documents"][0]) == 1

    # Re-upsert identical IDs (idempotency check)
    store.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    assert store.count() == 2

    # Delete by ID
    store.delete(ids=["res_1_exp_0"])
    assert store.count() == 1

    # Delete by metadata filter
    store.delete(where={"filename": "john.pdf"})
    assert store.count() == 0


def test_chroma_ephemeral_store_zero_disk():
    eph_store = ChromaVectorStore(collection_name="ephemeral_test", ephemeral=True)
    assert eph_store.ephemeral is True
    assert eph_store.count() == 0

    ids = ["eph_1"]
    docs = ["Ephemeral Cloud Architect"]
    embeddings = [[0.15] * 384]
    metadatas = [{"candidate_name": "Ephemeral Candidate", "filename": "eph.pdf"}]

    eph_store.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    assert eph_store.count() == 1

    eph_store.delete(ids=ids)
    assert eph_store.count() == 0


def test_qdrant_store_operations_and_idempotency():
    store = QdrantVectorStore(collection_name="test_resumes")
    assert store.count() == 0

    ids = ["res_1_exp_0", "res_2_exp_0"]
    docs = ["5 years Go experience", "3 years Rust experience"]
    embeddings = [[0.1] * 384, [0.2] * 384]
    metadatas = [
        {"candidate_name": "Jane Smith", "section": "EXPERIENCE", "filename": "jane.pdf"},
        {"candidate_name": "Bob Jones", "section": "EXPERIENCE", "filename": "bob.pdf"},
    ]

    store.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    assert store.count() == 2

    # Delete by where
    store.delete(where={"filename": "jane.pdf"})
    assert store.count() == 1

    # Delete by ids
    store.delete(ids=["res_2_exp_0"])
    assert store.count() == 0


def test_composite_vector_store_layering(tmp_path):
    base = ChromaVectorStore(collection_name="base_col", db_path=str(tmp_path / "base_db"))
    eph = ChromaVectorStore(collection_name="eph_col", ephemeral=True)
    comp = CompositeVectorStore(base_store=base, ephemeral_store=eph)

    # Base has 1 document
    base.upsert(
        ids=["base_1"],
        documents=["Base candidate text"],
        embeddings=[[0.05] * 384],
        metadatas=[{"candidate_name": "Base Cand"}],
    )

    # Ingest into composite (should write strictly to ephemeral)
    comp.upsert(
        ids=["eph_1"],
        documents=["Uploaded candidate text"],
        embeddings=[[0.10] * 384],
        metadatas=[{"candidate_name": "Uploaded Cand"}],
    )

    assert base.count() == 1
    assert eph.count() == 1
    assert comp.count() == 2

    # get_all combines records
    all_recs = comp.get_all()
    assert len(all_recs["ids"]) == 2

    # Query merges results
    res = comp.query(query_embedding=[0.05] * 384, n_results=2)
    assert len(res["ids"][0]) == 2

    # Delete from composite prunes from ephemeral
    comp.delete(ids=["eph_1"])
    assert comp.count() == 1
    assert eph.count() == 0
    assert base.count() == 1


def test_store_injection_into_pipeline_and_service():
    qdrant_stub = QdrantVectorStore()
    pipeline = ResumeRAGPipeline(store=qdrant_stub)
    assert pipeline.store is qdrant_stub

    service = IngestionService(store=qdrant_stub)
    assert service.pipeline.store is qdrant_stub
