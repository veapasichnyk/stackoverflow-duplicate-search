from __future__ import annotations

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Which provider is default if request doesn't specify (we still allow override)
    DEFAULT_PROVIDER: str = Field(default="sbert")

    # SBERT config
    SBERT_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    DEVICE: str = Field(default="cpu")  # "cpu" or "cuda"
    SBERT_BATCH_SIZE: int = Field(default=64, ge=1, le=512)

    # OpenAI config
    OPENAI_MODEL: str = Field(default="text-embedding-3-large")
    OPENAI_API_KEY: str | None = Field(default=None)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()