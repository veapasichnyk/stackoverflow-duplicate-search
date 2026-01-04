from __future__ import annotations

from typing import List
import numpy as np
from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-large"):
        # Uses env automatically too, but explicit key is fine for container clarity
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI recommends stripping newlines
        inp = [t.replace("\n", " ") for t in texts]

        resp = self.client.embeddings.create(
            model=self.model_name,
            input=inp
        )

        vecs = np.array([d.embedding for d in resp.data], dtype="float32")

        # Normalize for cosine consistency
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms

        return vecs.tolist()