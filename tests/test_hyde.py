"""Unit tests for HyDEService (Hypothetical Document Embeddings query synthesis)."""

from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from agentic_profile_matching.services.hyde_service import HyDEService


def test_hyde_heuristic_fallback_when_no_llm():
    hyde = HyDEService(llm=None)
    query = "Looking for a Senior Python Backend Engineer with Kubernetes and FastAPI experience"
    result = hyde.generate_hypothetical_profile(query)

    assert "Senior Python Backend Engineer" in result
    assert "FastAPI" in result or "Kubernetes" in result
    assert "Target Role / Profile Summary" in result


def test_hyde_llm_synthesis():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Experienced Senior Python Engineer with 6+ years designing scalable microservices using FastAPI, Redis, and Kubernetes on AWS."
    )

    hyde = HyDEService(llm=mock_llm, enable_cache=True)
    query = "Senior Python Engineer"
    result = hyde.generate_hypothetical_profile(query)

    assert "Experienced Senior Python Engineer" in result
    assert mock_llm.invoke.call_count == 1

    # Second call should hit in-memory cache and not invoke LLM again
    cached_result = hyde.generate_hypothetical_profile(query)
    assert cached_result == result
    assert mock_llm.invoke.call_count == 1


def test_hyde_llm_failure_falls_back_gracefully():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Groq API Timeout")

    hyde = HyDEService(llm=mock_llm, enable_cache=False)
    query = "Staff Data Engineer with Spark and Kafka"
    result = hyde.generate_hypothetical_profile(query)

    # Should not raise exception, but return heuristic synthetic profile
    assert "Staff Data Engineer" in result
    assert "Core Competencies" in result
