from __future__ import annotations

import joblib
from typing import Tuple


def load_vectorizer_and_model(
    vectorizer_path: str,
    model_path: str,
):
    """
    Loads TF-IDF vectorizer and classifier from disk.
    """
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model