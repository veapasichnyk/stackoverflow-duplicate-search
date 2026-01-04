from __future__ import annotations

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Paths inside container
    VECTORIZER_PATH: str = Field(
        default="artifacts/tfidf_vectorizer.joblib"
    )
    MODEL_PATH: str = Field(
        default="artifacts/classifier.joblib"
    )

    # Prediction
    THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()