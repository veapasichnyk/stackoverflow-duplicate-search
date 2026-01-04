from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from qdrant_client import QdrantClient


@dataclass
class QdrantHit:
    id: str
    title: str
    score: float


class QdrantSearchClient:
    def __init__(self, url: str, collection: str):
        self.url = url
        self.collection = collection
        self.client = QdrantClient(url=url)

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        cols = [c.name for c in self.client.get_collections().collections]
        return self.collection in cols

    def points_count(self) -> Optional[int]:
        try:
            info = self.client.get_collection(self.collection)
            return info.points_count
        except Exception:
            return None

    def search(self, vector: List[float], top_k: int = 5) -> List[QdrantHit]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )

        out: List[QdrantHit] = []
        for h in hits:
            payload = h.payload or {}
            out.append(
                QdrantHit(
                    id=str(h.id),
                    title=str(payload.get("title", "")),
                    score=float(h.score),
                )
            )
        return out