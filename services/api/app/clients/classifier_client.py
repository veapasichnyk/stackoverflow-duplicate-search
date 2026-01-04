from __future__ import annotations

from typing import Tuple

import requests


class ClassifierClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def predict(self, title1: str, title2: str) -> Tuple[int, float]:
        """
        Contract:
          POST /predict {"title1":"...", "title2":"..."}
          -> {"label":0|1, "proba": float}
        """
        payload = {"title1": title1, "title2": title2}
        r = requests.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        if "label" not in data or "proba" not in data:
            raise ValueError("Classifier response must contain 'label' and 'proba'.")
        return int(data["label"]), float(data["proba"])