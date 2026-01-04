from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class SBERTEmbedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
            normalize_embeddings=True,   # good for cosine
        ).astype("float32")

        return vecs.tolist()