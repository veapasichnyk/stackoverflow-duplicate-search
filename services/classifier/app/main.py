from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .schemas import PredictRequest, PredictResponse
from .settings import settings
from .model.load_model import load_vectorizer_and_model
from .model.features import build_features


app = FastAPI(title="Duplicate Classifier Service", version="0.1.0")

vectorizer = None
model = None


@app.on_event("startup")
def load_artifacts():
    global vectorizer, model
    try:
        vectorizer, model = load_vectorizer_and_model(
            settings.VECTORIZER_PATH,
            settings.MODEL_PATH,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = build_features(
        [req.title1],
        [req.title2],
        vectorizer
    )

    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= settings.THRESHOLD)

    return PredictResponse(label=label, proba=proba)