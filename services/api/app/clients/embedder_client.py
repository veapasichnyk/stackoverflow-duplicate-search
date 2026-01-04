from __future__ import annotations

from typing import List, Literal, Sequence

import requests


EmbeddingProvider = Literal["sbert", "openai"]


class EmbedderClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, texts: Sequence[str], provider: EmbeddingProvider = "sbert") -> List[List[float]]:
        """
        Returns list of vectors aligned to input texts.
        Contract:
          POST /embed {"texts":[...], "provider":"sbert"|"openai"}
          -> {"vectors":[[...], ...]}
        """
        payload = {"texts": list(texts), "provider": provider}
        r = requests.post(f"{self.base_url}/embed", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        if "vectors" not in data or not isinstance(data["vectors"], list):
            raise ValueError("Embedder response missing 'vectors' list.")
        return data["vectors"]