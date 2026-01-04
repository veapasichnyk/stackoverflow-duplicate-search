#!/usr/bin/env python3
"""
scripts/search_titles.py

Client-facing demo:
- takes a query text
- embeds with SBERT
- searches in Qdrant
- prints score, source_id, title

Example:
  python scripts/search_titles.py --query "How to create friendly URL in php?" --limit 5
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sentence_transformers import SentenceTransformer


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    p.add_argument("--collection", default="stack_overflow_titles", help="Qdrant collection name")
    p.add_argument("--query", required=True, help="Query text")
    p.add_argument("--limit", type=int, default=5, help="Top-k results")
    p.add_argument("--min-score", type=float, default=None, help="Optional minimum similarity score filter")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SBERT model name")
    p.add_argument("--device", default="cpu", help="cpu or cuda (if available)")
    p.add_argument("--timeout", type=int, default=120, help="Qdrant client timeout (seconds)")
    args = p.parse_args()

    query = _safe_str(args.query).strip()
    if not query:
        print("ERROR: --query is empty.", file=sys.stderr)
        return 2
    
    # Load encoder

    try:
        model = SentenceTransformer(args.model, device=args.device)
    except Exception as e:
        print(f"ERROR: Failed to load SBERT model: {e}", file=sys.stderr)
        return 2

    # Embed query

    try:
        qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    except Exception as e:
        print(f"ERROR: Failed to encode query: {e}", file=sys.stderr)
        return 2

    # Connect to Qdrant

    try:
        client = QdrantClient(url=args.qdrant_url, timeout=args.timeout)
    except Exception as e:
        print(f"ERROR: Failed to create Qdrant client: {e}", file=sys.stderr)
        return 2

    # Ensure collection exists

    try:
        if not client.collection_exists(args.collection):
            print(f"ERROR: Collection '{args.collection}' does not exist on {args.qdrant_url}.", file=sys.stderr)
            return 2
    except Exception as e:
        print(f"ERROR: Could not check collection existence: {e}", file=sys.stderr)
        return 2

    # Search (use query_points - works across newer qdrant-client versions)

    try:
        res = client.query_points(
            collection_name=args.collection,
            query=qvec,
            limit=args.limit,
            with_payload=True,
            with_vectors=False,
        )
        hits = res.points  # List[ScoredPoint]
    except (ResponseHandlingException, UnexpectedResponse) as e:
        print(f"ERROR: Qdrant query failed: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: Unexpected error during search: {e}", file=sys.stderr)
        return 2

    # Print results
    print(f'Query: "{query}"')
    print(f"Qdrant: {args.qdrant_url} | collection: {args.collection}")
    print("-" * 92)

    shown = 0
    for i, h in enumerate(hits, 1):
        score = float(h.score) if h.score is not None else 0.0
        if args.min_score is not None and score < args.min_score:
            continue

        payload: Dict[str, Any] = h.payload or {}
        title = _safe_str(payload.get("title", "")).replace("\n", " ").strip()
        source_id = _safe_str(payload.get("source_id", ""))

        print(f"{i:02d}. score={score:.4f} | source_id={source_id} | point_id={h.id}")
        print(f"    title: {title}")
        shown += 1

    if shown == 0:
        print("No results (or all filtered by --min-score).")

    print("-" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
