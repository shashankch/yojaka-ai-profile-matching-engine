import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file at startup
load_dotenv()

# Base Paths
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
BASE_DIR = PROJECT_ROOT if (PROJECT_ROOT / "data").exists() else PACKAGE_DIR.parent
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(PROJECT_ROOT / "chroma_db"))
DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        str(PROJECT_ROOT / "data" if (PROJECT_ROOT / "data").exists() else BASE_DIR / "data"),
    )
)
RESUMES_DIR = os.getenv("RESUMES_DIR", str(DATA_DIR / "resumes"))
JOB_DESCRIPTIONS_DIR = os.getenv("JOB_DESCRIPTIONS_DIR", str(DATA_DIR / "job_descriptions"))

# Embedding & Two-Stage Reranking Config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "True").lower() in ("true", "1", "yes")
TOP_K = int(os.getenv("TOP_K", "10"))

# Supported LLM Models & Providers
SUPPORTED_PROVIDERS = {
    "Groq": [
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
    ],
    "Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "Sarvam AI": ["sarvam-105b", "sarvam-2b"],
    "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Ollama (Local)": ["llama3.2", "qwen2.5:7b", "deepseek-r1:8b", "mistral:7b"],
    "vLLM (Local)": ["local-model"],
    "Custom (OpenAI-compatible)": ["custom-model"],
}

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "Groq")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-120b")
THROTTLE_DELAY = float(os.getenv("THROTTLE_DELAY", "0.5"))  # Delay in seconds between candidate screening calls

# Default Candidate Processing Limits
DEFAULT_COARSE_LIMIT = int(os.getenv("DEFAULT_COARSE_LIMIT", "10"))
DEFAULT_DEEP_LIMIT = int(os.getenv("DEFAULT_DEEP_LIMIT", "5"))
DEFAULT_RECOMMENDATION_LIMIT = int(os.getenv("DEFAULT_RECOMMENDATION_LIMIT", "3"))
RESUME_TRUNCATION_LIMIT = int(os.getenv("RESUME_TRUNCATION_LIMIT", "12000"))

# Phase 19: Advanced RAG (HyDE & Parent Document) & Air-Gapped Local Inference
USE_HYDE = os.getenv("USE_HYDE", "True").lower() in ("true", "1", "yes")
HYDE_TIMEOUT = float(os.getenv("HYDE_TIMEOUT", "1.5"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LOCAL_INFERENCE_URL = os.getenv("LOCAL_INFERENCE_URL", "http://localhost:8000/v1")
LOCAL_INFERENCE_MODEL = os.getenv("LOCAL_INFERENCE_MODEL", "llama3.2")

# MCP Protocol Configuration
USE_MCP = os.getenv("USE_MCP", "False").lower() in ("true", "1", "yes")
FILESYSTEM_SERVER_PATH = os.getenv(
    "FILESYSTEM_SERVER_PATH",
    str(PACKAGE_DIR / "filesystem_mcp_server.py"),
)
SEARCH_SERVER_PATH = os.getenv(
    "SEARCH_SERVER_PATH",
    str(PACKAGE_DIR / "search_mcp_server.py"),
)
MCP_TIMEOUT = float(os.getenv("MCP_TIMEOUT", "30.0"))

# Celery & Redis Async Task Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Observability & Tracing Configuration
OBSERVABILITY_BACKEND = os.getenv("OBSERVABILITY_BACKEND", "none").lower()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")

# Ingestion Engine Configuration
USE_UNSTRUCTURED = os.getenv("USE_UNSTRUCTURED", "false").lower() == "true"


def get_llm_model(provider: str, model_name: str, api_key: str, api_url: Optional[str] = None):
    prov = (provider or "").strip().lower()
    if prov == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, api_key=api_key)  # type: ignore[arg-type]
    elif prov in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    elif "sarvam" in prov:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,  # type: ignore[arg-type]
            base_url="https://api.sarvam.ai/v1",
        )
    elif prov == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, api_key=api_key)  # type: ignore[arg-type]
    elif "ollama" in prov:
        from langchain_openai import ChatOpenAI

        ollama_url = api_url or OLLAMA_BASE_URL
        base_url = ollama_url if ollama_url.endswith("/v1") else f"{ollama_url.rstrip('/')}/v1"
        return ChatOpenAI(
            model=model_name or OLLAMA_MODEL,
            api_key=api_key or "ollama",
            base_url=base_url,
        )
    elif "vllm" in prov or "local" in prov:
        from langchain_openai import ChatOpenAI

        vllm_url = api_url or LOCAL_INFERENCE_URL
        return ChatOpenAI(
            model=model_name or LOCAL_INFERENCE_MODEL,
            api_key=api_key or "local",
            base_url=vllm_url,
        )
    elif "custom" in prov or "openai-compatible" in prov:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key or "custom",  # type: ignore[arg-type]
            base_url=api_url or "http://localhost:11434/v1",
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
