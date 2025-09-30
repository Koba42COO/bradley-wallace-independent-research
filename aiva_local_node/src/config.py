#!/usr/bin/env python3
"""
AIVA Configuration
Centralized configuration management for the AIVA local node
"""

import os
from typing import Optional

# === AIVA Core Configuration ===
AIVA_SYSTEM_PROMPT: str = os.getenv(
    "AIVA_SYSTEM_PROMPT",
    (
        "You are Aiva — a nerdy, playful, wise AI mentor. You promote truth, knowledge, the scientific method, and critical thinking. "
        "You contextualize speculative ideas as working theories, speak plainly and conversationally, avoid pretension, and delight in cross-domain links. "
        "Default style: friendly and clear; avoid purple prose; keep structure light; push back gently on illogic."
    ),
)

# === Context Retrieval Configuration ===
CONTEXT_DIR: Optional[str] = os.getenv("AIVA_CONTEXT_DIR")  # e.g., "./context"
MAX_CONTEXT_CHARS: int = int(os.getenv("AIVA_MAX_CONTEXT_CHARS", "4000"))
MAX_CONTEXT_PASSAGES: int = int(os.getenv("AIVA_MAX_CONTEXT_PASSAGES", "6"))

# === Embeddings Configuration ===
USE_EMBEDDINGS: bool = os.getenv("AIVA_USE_EMBEDDINGS", "0").strip() in {"1","true","yes","on"}
EMB_MODEL_NAME: str = os.getenv("AIVA_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# === Web Tool Configuration ===
WEB_MAX_BYTES: int = int(os.getenv("AIVA_WEB_MAX_BYTES", "524288"))  # 512 KiB
WEB_TIMEOUT: float = float(os.getenv("AIVA_WEB_TIMEOUT", "8.0"))
WEB_ALLOW_DOMAINS: Optional[str] = os.getenv("AIVA_WEB_ALLOW_DOMAINS")  # comma-separated allowlist

# === Security Configuration ===
API_KEY: Optional[str] = os.getenv("AIVA_API_KEY")
WRITE_MAX_BYTES: int = int(os.getenv("AIVA_WRITE_MAX_BYTES", "262144"))  # 256 KiB

# === Request Processing Configuration ===
LOG_REQUESTS: bool = os.getenv("AIVA_LOG_REQUESTS", "1") in {"1","true","yes","on"}
LOG_MAX_BODY: int = int(os.getenv("AIVA_LOG_MAX_BODY", "2000"))
RATE_LIMIT_PER_MIN: int = int(os.getenv("AIVA_RATE_LIMIT_PER_MIN", "120"))

# === Model Configuration ===
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/Mixtral-8x7B-Instruct-v0.1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "mixtral-8x7b-instruct")
TENSOR_PARALLEL_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
MAX_MODEL_LEN: int = int(os.getenv("MAX_MODEL_LEN", "4096"))
DTYPE: str = os.getenv("DTYPE", "auto")
TRUST_REMOTE_CODE: bool = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"

# === Server Configuration ===
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
WORKERS: int = int(os.getenv("WORKERS", "1"))

# === Development Configuration ===
NODE_ENV: str = os.getenv("NODE_ENV", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")