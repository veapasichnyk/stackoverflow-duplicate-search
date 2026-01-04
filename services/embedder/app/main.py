from __future__ import annotations

from fastapi import FastAPI, HTTPException
from .schemas import EmbedRequest, EmbedResponse
from .settings import settings
from .embedding.sbert import SBERTEmbedder
from .embedding.openai_embed import OpenAIEmbedder


app = FastAPI(title="Embedder Service", version="0.1.0")

# Lazy init (so import is fast)
_sbert = None
_openai = None


def get_sbert() -> SBERTEmbedder:
    global _sbert
    if _sbert is None:
        _sbert = SBERTEmbedder(
            model_name=settings.SBERT_MODEL,
            device=settings.DEVICE,
            batch_size=settings.SBERT_BATCH_SIZE,
        )
    return _sbert


def get_openai() -> OpenAIEmbedder:
    global _openai
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set for provider=openai")
    if _openai is None:
        _openai = OpenAIEmbedder(api_key=api_key, model_name=settings.OPENAI_MODEL)
    return _openai


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_provider": settings.DEFAULT_PROVIDER,
        "sbert_model": settings.SBERT_MODEL,
        "device": settings.DEVICE,
        "openai_model": settings.OPENAI_MODEL,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    provider = req.provider or settings.DEFAULT_PROVIDER

    # Light sanitization
    texts = [str(t).strip() for t in req.texts]
    if any(len(t) == 0 for t in texts):
        raise HTTPException(status_code=400, detail="Empty text found in request.")

    if provider == "sbert":
        vectors = get_sbert().embed(texts)
    elif provider == "openai":
        vectors = get_openai().embed(texts)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    return EmbedResponse(provider=provider, vectors=vectors)