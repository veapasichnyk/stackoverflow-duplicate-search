from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import requests
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient

from .schemas import (
    SearchRequest,
    SearchResponse,
    SearchHit,
    DuplicatePredictRequest,
    DuplicatePredictResponse,
    HealthResponse,
)

# -------------------------
# Settings (env)
# -------------------------
EMBEDDER_URL = os.getenv("EMBEDDER_URL", "http://embedder:8000")
CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "stack_overflow_titles")

TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

app = FastAPI(
    title="StackOverflow Duplicate + Semantic Search API",
    version="0.1.0",
)

qdrant = QdrantClient(url=QDRANT_URL)


# -------------------------
# Helpers
# -------------------------
def embed_text(query: str, provider: str) -> List[float]:
    """
    Calls embedder service to get a single embedding vector for query.
    Expected embedder contract:
      POST /embed  { "texts": [..], "provider": "sbert"|"openai" }
      -> { "vectors": [[..], ...] }
    """
    try:
        r = requests.post(
            f"{EMBEDDER_URL.rstrip('/')}/embed",
            json={"texts": [query], "provider": provider},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        vec = data["vectors"][0]
        return vec
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Embedder error: {e}")
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(status_code=502, detail=f"Embedder invalid response: {e}")


def qdrant_search(vector: List[float], top_k: int) -> List[SearchHit]:
    try:
        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Qdrant search error: {e}")

    out: List[SearchHit] = []
    for h in hits:
        payload = h.payload or {}
        title = payload.get("title", "")
        out.append(SearchHit(id=str(h.id), title=title, score=float(h.score)))
    return out


def classifier_predict(title1: str, title2: str) -> Tuple[int, float]:
    """
    Calls classifier service.
    Expected contract:
      POST /predict { "title1": "...", "title2": "..." }
      -> { "label": 0|1, "proba": float }
    """
    try:
        r = requests.post(
            f"{CLASSIFIER_URL.rstrip('/')}/predict",
            json={"title1": title1, "title2": title2},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return int(data["label"]), float(data["proba"])
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Classifier error: {e}")
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Classifier invalid response: {e}")


# -------------------------
# Routes
# -------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Best-effort checks
    embedder_status = None
    classifier_status = None
    qdrant_status = None
    collection_status = None

    try:
        r = requests.get(f"{EMBEDDER_URL.rstrip('/')}/health", timeout=10)
        embedder_status = "ok" if r.status_code == 200 else f"bad({r.status_code})"
    except Exception:
        embedder_status = "unreachable"

    try:
        r = requests.get(f"{CLASSIFIER_URL.rstrip('/')}/health", timeout=10)
        classifier_status = "ok" if r.status_code == 200 else f"bad({r.status_code})"
    except Exception:
        classifier_status = "unreachable"

    try:
        cols = [c.name for c in qdrant.get_collections().collections]
        qdrant_status = "ok"
        collection_status = "ok" if QDRANT_COLLECTION in cols else "missing"
    except Exception:
        qdrant_status = "unreachable"
        collection_status = None

    return HealthResponse(
        api="ok",
        embedder=embedder_status,
        classifier=classifier_status,
        qdrant=qdrant_status,
        collection=collection_status,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    vec = embed_text(req.query, req.provider)
    hits = qdrant_search(vec, req.top_k)
    return SearchResponse(query=req.query, provider=req.provider, top_k=req.top_k, hits=hits)


@app.post("/duplicate/predict", response_model=DuplicatePredictResponse)
def predict_duplicate(req: DuplicatePredictRequest) -> DuplicatePredictResponse:
    label, proba = classifier_predict(req.title1, req.title2)
    return DuplicatePredictResponse(label=label, proba=proba)