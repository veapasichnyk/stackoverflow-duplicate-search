from __future__ import annotations

import os
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # ========================
    # Service URLs
    # ========================
    EMBEDDER_URL: str = Field(
        default="http://embedder:8000",
        description="Base URL of embedding service"
    )

    CLASSIFIER_URL: str = Field(
        default="http://classifier:8000",
        description="Base URL of duplicate classifier service"
    )

    # ========================
    # Qdrant
    # ========================
    QDRANT_URL: str = Field(
        default="http://qdrant:6333",
        description="Qdrant service URL"
    )

    QDRANT_COLLECTION: str = Field(
        default="stack_overflow_titles",
        description="Qdrant collection name"
    )

    # ========================
    # API behavior
    # ========================
    HTTP_TIMEOUT: float = Field(
        default=60.0,
        description="Timeout for inter-service HTTP calls (seconds)"
    )

    # ========================
    # Runtime
    # ========================
    ENV: str = Field(
        default="dev",
        description="Environment name (dev/staging/prod)"
    )

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton-style settings object
settings = Settings()