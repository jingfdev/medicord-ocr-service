"""Application configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Central configuration loaded from .env / environment variables."""

    # ── Application ──
    APP_NAME: str = "medicord-ocr-service"
    APP_ENV: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Server ──
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # ── JWT Auth ──
    JWT_SECRET_KEY: str = "change-me-to-a-secure-random-string-min-32-chars"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── Redis ──
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Celery ──
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── OCR ──
    TESSERACT_CMD: str = "/usr/bin/tesseract"
    TESSDATA_PREFIX: str = "/usr/share/tesseract-ocr/5/tessdata"
    DEFAULT_LANGUAGES: str = "khm+eng"
    OCR_FALLBACK_ENGINE: str = "easyocr"  # easyocr | paddleocr

    # ── File Upload ──
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: str = "jpg,jpeg,png,pdf,tiff,bmp"
    UPLOAD_DIR: str = "/tmp/medicord-uploads"

    # ── Security ──
    RATE_LIMIT_PER_MINUTE: int = 30
    CORS_ORIGINS: str = "*"
    DELETE_AFTER_PROCESSING: bool = True

    # ── Optional LLM Post-processing ──
    LLM_ENABLED: bool = False
    LLM_PROVIDER: str = "gemini"  # gemini | openai
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
