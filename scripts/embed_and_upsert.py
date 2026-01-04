#!/usr/bin/env python3
"""
embed_and_upsert.py

Batch-embed corpus titles and upsert into Qdrant.

Inputs:
  - corpus file (.csv/.parquet) with columns: id, title

Providers:
  - sbert: sentence-transformers model (default all-MiniLM-L6-v2)
  - openai: OpenAI embeddings (default text-embedding-3-large)

Qdrant:
  - creates collection if missing
  - upserts points (int/uuid id, vector, payload)

Note on IDs:
  Qdrant point IDs must be either:
    - unsigned integer, or
    - UUID string
  String IDs like "t_abc123" are NOT valid.
  This script uses sequential integer IDs and stores the original "id" as payload["source_id"].

Example:
  python scripts/embed_and_upsert.py ^
    --corpus data/processed/unique_titles.parquet ^
    --qdrant-url http://localhost:6333 ^
    --collection stack_overflow_titles ^
    --provider sbert ^
    --device cpu
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported corpus format: {ext}. Use .csv or .parquet")


def chunk_iter(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        yield start, end


def ensure_collection(client: QdrantClient, collection: str, vector_size: int, distance: str = "Cosine") -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        return

    dist_map = {"Cosine": qm.Distance.COSINE, "Dot": qm.Distance.DOT, "Euclid": qm.Distance.EUCLID}
    if distance not in dist_map:
        raise ValueError(f"Unsupported distance: {distance}. Choose from {list(dist_map.keys())}")

    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=vector_size, distance=dist_map[distance]),
    )


def embed_sbert(texts: List[str], model_name: str, device: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,  # good for cosine
    )
    return vecs.astype("float32")


def embed_openai(texts: List[str], model_name: str) -> np.ndarray:
    from openai import OpenAI

    # expects OPENAI_API_KEY in env
    client = OpenAI()
    resp = client.embeddings.create(model=model_name, input=[t.replace("\n", " ") for t in texts])
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")

    # Normalize for cosine consistency
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    return vecs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="Path to corpus (.csv/.parquet) with columns id,title")
    p.add_argument("--qdrant-url", required=True, help="Qdrant URL, e.g. http://localhost:6333")
    p.add_argument("--collection", required=True, help="Qdrant collection name")
    p.add_argument("--provider", choices=["sbert", "openai"], default="sbert")
    p.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--openai-model", default="text-embedding-3-large")
    p.add_argument("--device", default="cpu", help="cpu or cuda (for sbert)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding + upsert")
    p.add_argument("--embed-batch-size", type=int, default=64, help="Internal batch size for SBERT encode")
    p.add_argument("--limit", type=int, default=0, help="Optional limit number of rows (0=all)")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds for Qdrant client")
    args = p.parse_args()

    df = read_table(args.corpus)
    if "id" not in df.columns or "title" not in df.columns:
        raise ValueError("Corpus must contain columns: id, title")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    n = len(df)
    if n == 0:
        raise ValueError("Corpus is empty")

    # Original string IDs (kept in payload only)
    source_ids = df["id"].astype(str).tolist()
    titles = df["title"].astype(str).tolist()

    # Qdrant IDs: sequential ints (valid)
    qdrant_ids = list(range(n))

    print(f"[embed_and_upsert] corpus rows: {n:,}")

    client = QdrantClient(url=args.qdrant_url, timeout=args.timeout)

    # Determine vector size using a tiny sample
    sample_text = [titles[0]]
    if args.provider == "sbert":
        sample_vec = embed_sbert(sample_text, args.sbert_model, args.device, batch_size=1)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        sample_vec = embed_openai(sample_text, args.openai_model)

    vector_size = int(sample_vec.shape[1])
    ensure_collection(client, args.collection, vector_size, distance="Cosine")
    print(f"[embed_and_upsert] collection ready: {args.collection} (dim={vector_size})")

    total_upserted = 0
    t0 = time.time()

    for start, end in chunk_iter(n, args.batch_size):
        batch_titles = titles[start:end]
        batch_qids = qdrant_ids[start:end]
        batch_sids = source_ids[start:end]

        if args.provider == "sbert":
            vecs = embed_sbert(batch_titles, args.sbert_model, args.device, batch_size=args.embed_batch_size)
        else:
            vecs = embed_openai(batch_titles, args.openai_model)

        # payload: keep original id + title
        payloads = [{"title": t, "source_id": sid} for t, sid in zip(batch_titles, batch_sids)]

        points = [
            qm.PointStruct(
                id=int(batch_qids[i]),            # VALID: unsigned int
                vector=vecs[i].tolist(),
                payload=payloads[i],
            )
            for i in range(len(batch_qids))
        ]

        client.upsert(collection_name=args.collection, points=points)
        total_upserted += len(points)

        if total_upserted % (args.batch_size * 10) == 0 or end == n:
            elapsed = time.time() - t0
            rate = total_upserted / max(elapsed, 1e-9)
            print(f"[embed_and_upsert] upserted {total_upserted:,}/{n:,}  ({rate:,.0f} pts/s, elapsed {elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"[embed_and_upsert] done. total upserted: {total_upserted:,} (elapsed {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
