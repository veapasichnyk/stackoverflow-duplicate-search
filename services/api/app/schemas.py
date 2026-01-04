from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


EmbeddingProvider = Literal["sbert", "openai"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query text")
    top_k: int = Field(5, ge=1, le=50)
    provider: EmbeddingProvider = Field("sbert", description="Embedding provider to use")
    with_scores: bool = Field(True, description="Return similarity scores")


class SearchHit(BaseModel):
    id: str
    title: str
    score: float


class SearchResponse(BaseModel):
    query: str
    provider: EmbeddingProvider
    top_k: int
    hits: List[SearchHit]


class DuplicatePredictRequest(BaseModel):
    title1: str = Field(..., min_length=1)
    title2: str = Field(..., min_length=1)


class DuplicatePredictResponse(BaseModel):
    label: int
    proba: float


class HealthResponse(BaseModel):
    api: str
    embedder: Optional[str] = None
    classifier: Optional[str] = None
    qdrant: Optional[str] = None
    collection: Optional[str] = None