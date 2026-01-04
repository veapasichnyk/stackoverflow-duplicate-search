from __future__ import annotations

from typing import List, Literal
from pydantic import BaseModel, Field


EmbeddingProvider = Literal["sbert", "openai"]


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to embed")
    provider: EmbeddingProvider = Field("sbert", description="Embedding provider: sbert or openai")


class EmbedResponse(BaseModel):
    provider: EmbeddingProvider
    vectors: List[List[float]]