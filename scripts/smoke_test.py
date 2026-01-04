#!/usr/bin/env python3
"""
smoke_test.py

Quick checks for Qdrant + embedding flow.

Modes:
  - direct: embed with SBERT locally, then search Qdrant
  - service: call embedder service (/embed), then search Qdrant
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def embed_direct_sbert(text: str, model_name: str, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True, device=device)
    return vec.astype("float32")[0]


def embed_via_service(texts: List[str], embedder_url: str, provider: str = "sbert") -> np.ndarray:
    # expected embedder API contract:
    # POST {embedder_url}/embed  { "texts": [...], "provider": "sbert"|"openai" }
    r = requests.post(
        f"{embedder_url.rstrip('/')}/embed",
        json={"texts": texts, "provider": provider},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    vecs = np.array(data["vectors"], dtype="float32")
    return vecs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qdrant-url", required=True, help="http://localhost:6333")
    p.add_argument("--collection", required=True, help="collection name")
    p.add_argument("--query", default="How to center a div in CSS?", help="query text")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--mode", choices=["direct", "service"], default="direct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedder-url", default="http://localhost:8001", help="embedder service base url")
    p.add_argument("--provider", choices=["sbert", "openai"], default="sbert")
    args = p.parse_args()

    client = QdrantClient(url=args.qdrant_url)

    # 1) Qdrant health
    cols = client.get_collections().collections
    col_names = [c.name for c in cols]
    print("[smoke] qdrant ok. collections:", col_names[:10], ("..." if len(col_names) > 10 else ""))

    if args.collection not in col_names:
        raise RuntimeError(f"[smoke] collection not found: {args.collection}")

    info = client.get_collection(args.collection)
    print(f"[smoke] collection '{args.collection}' points:", info.points_count)

    if not info.points_count or info.points_count == 0:
        raise RuntimeError("[smoke] collection is empty (no points). Run embed_and_upsert.py first.")

    # 2) Embedding
    if args.mode == "direct":
        qvec = embed_direct_sbert(args.query, args.sbert_model, args.device)
    else:
        vecs = embed_via_service([args.query], args.embedder_url, provider=args.provider)
        # normalize for cosine search (safe even if already normalized)
        norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norm
        qvec = vecs[0]

    # 3) Search
    hits = client.search(
        collection_name=args.collection,
        query_vector=qvec.tolist(),
        limit=args.topk,
        with_payload=True,
    )

    print("\n[smoke] query:", args.query)
    for i, h in enumerate(hits, 1):
        title = (h.payload or {}).get("title", "")
        print(f"  {i:>2}. score={h.score:.4f}  title={title}")

    print("\n[smoke] OK ✅")


if __name__ == "__main__":
    main()