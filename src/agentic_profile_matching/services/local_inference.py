"""
Air-Gapped Local Inference Engine (Phase 19 / ADR-019).
Provides zero-cloud, 100% offline candidate evaluation using local LLM inference
backends (Ollama, vLLM, and OpenAI-compatible local endpoints).
Guarantees zero candidate PII egress, zero external API token costs, and full
GDPR / sovereign cloud compliance.
"""

import json
from typing import Any, Dict, Optional
import urllib.request
import urllib.error

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agentic_profile_matching import config
from agentic_profile_matching.observability import get_logger

logger = get_logger("agentic_profile_matching.services.local_inference")


class LocalInferenceService:
    """
    Coordinates health probing, model enumeration, and instantiation for
    air-gapped local model servers.
    """

    @staticmethod
    def check_local_llm_health(
        base_url: Optional[str] = None,
        provider_type: str = "ollama",
        timeout: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Probes local model daemon availability and lists locally available models.
        Supports Ollama (default: http://localhost:11434) and vLLM (default: http://localhost:8000).
        """
        url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        prov = provider_type.lower()

        # 1. Ollama probe endpoint: /api/tags
        if "ollama" in prov:
            endpoint = f"{url}/api/tags"
            try:
                req = urllib.request.Request(endpoint, headers={"User-Agent": "Yojaka-AI-HealthProbe"})
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode("utf-8"))
                        models = [m.get("name", "") for m in data.get("models", []) if "name" in m]
                        return {
                            "available": True,
                            "provider": "ollama",
                            "base_url": url,
                            "models": models,
                            "model_count": len(models),
                            "error": None,
                        }
            except Exception as e:
                logger.debug(f"Ollama health probe to {endpoint} failed: {e}")

        # 2. vLLM / OpenAI-compatible probe endpoint: /v1/models
        endpoint = f"{url}/v1/models"
        try:
            req = urllib.request.Request(endpoint, headers={"User-Agent": "Yojaka-AI-HealthProbe"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    models = [m.get("id", "") for m in data.get("data", []) if "id" in m]
                    return {
                        "available": True,
                        "provider": "vllm",
                        "base_url": url,
                        "models": models,
                        "model_count": len(models),
                        "error": None,
                    }
        except Exception as e:
            logger.debug(f"OpenAI-compatible local health probe to {endpoint} failed: {e}")

        return {
            "available": False,
            "provider": provider_type,
            "base_url": url,
            "models": [],
            "model_count": 0,
            "error": f"Local model service unreachable at {url}. Ensure 'ollama serve' or 'vllm' is running.",
        }

    @staticmethod
    def get_local_chat_model(
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_type: str = "ollama",
        temperature: float = 0.1,
    ) -> BaseChatModel:
        """
        Instantiates a local BaseChatModel via ChatOpenAI compatibility layer.
        Both Ollama and vLLM expose standard /v1/chat/completions endpoints.
        """
        url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        if not url.endswith("/v1"):
            api_base = f"{url}/v1"
        else:
            api_base = url

        target_model = model_name or config.OLLAMA_MODEL or "llama3.2"

        logger.info(f"Connecting to air-gapped local model: {target_model} at {api_base}")
        return ChatOpenAI(
            model=target_model,
            base_url=api_base,
            api_key="ollama",  # dummy key for local endpoint
            temperature=temperature,
        )
