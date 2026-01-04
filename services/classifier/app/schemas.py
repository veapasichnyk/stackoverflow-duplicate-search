from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    title1: str = Field(..., min_length=1)
    title2: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    label: int
    proba: float