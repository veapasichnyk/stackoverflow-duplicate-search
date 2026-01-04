from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity


def build_features(texts1, texts2, vectorizer):
    """
    Builds feature matrix for a pair of texts:
    - TF-IDF(title1)
    - TF-IDF(title2)
    - cosine similarity between TF-IDF vectors
    - absolute difference of text lengths
    """

    tfidf_1 = vectorizer.transform(texts1)
    tfidf_2 = vectorizer.transform(texts2)

    # Cosine similarity (as a single feature)
    cos_sim = cosine_similarity(tfidf_1, tfidf_2).reshape(-1, 1)

    # Length difference feature
    len_diff = np.abs(
        np.array([len(t.split()) for t in texts1]) -
        np.array([len(t.split()) for t in texts2])
    ).reshape(-1, 1)

    # Stack everything together
    X = hstack([tfidf_1, tfidf_2, cos_sim, len_diff])

    return X