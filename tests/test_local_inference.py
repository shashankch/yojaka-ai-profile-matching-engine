"""Unit tests for LocalInferenceService (air-gapped local daemon health probes and client factory)."""

import json
from unittest.mock import patch, MagicMock
from agentic_profile_matching.services.local_inference import LocalInferenceService


@patch("urllib.request.urlopen")
def test_ollama_health_probe_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(
        {"models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:latest"}]}
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    health = LocalInferenceService.check_local_llm_health(
        base_url="http://localhost:11434",
        provider_type="ollama",
    )

    assert health["available"] is True
    assert health["provider"] == "ollama"
    assert "llama3.2:latest" in health["models"]
    assert health["model_count"] == 2


@patch("urllib.request.urlopen")
def test_ollama_health_probe_failure(mock_urlopen):
    mock_urlopen.side_effect = ConnectionRefusedError("Connection refused")

    health = LocalInferenceService.check_local_llm_health(
        base_url="http://localhost:11434",
        provider_type="ollama",
    )

    assert health["available"] is False
    assert health["models"] == []
    assert "unreachable" in health["error"].lower()


@patch("urllib.request.urlopen")
def test_vllm_health_probe_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    health = LocalInferenceService.check_local_llm_health(
        base_url="http://localhost:8000",
        provider_type="vllm",
    )

    assert health["available"] is True
    assert health["provider"] == "vllm"
    assert "meta-llama/Llama-3.1-8B-Instruct" in health["models"]


def test_get_local_chat_model():
    model = LocalInferenceService.get_local_chat_model(
        model_name="llama3.2:latest",
        base_url="http://localhost:11434",
        provider_type="ollama",
    )
    assert model is not None
    assert model.model_name == "llama3.2:latest"
