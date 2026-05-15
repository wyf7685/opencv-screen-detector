from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.scoring.rules import FEATURE_NAMES

# Label mapping for 3-class classification
LABEL_NAMES = ["screenshot", "normal_photo", "screen_photo"]


def predict(
    model: object,
    features: dict[str, float],
    threshold: float = 0.5,  # noqa: ARG001
) -> tuple[str, float]:
    x = np.array([[features.get(name, 0.0) for name in FEATURE_NAMES]])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        pred_class = int(np.argmax(probs))
        prob = float(probs[pred_class])
    else:
        probs = model.predict(x)[0]
        pred_class = int(np.argmax(probs))
        prob = float(probs[pred_class])

    label = LABEL_NAMES[pred_class]
    return label, prob


def load_model(model_path: str | Path) -> object | None:
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301
    except Exception:
        return None


def batch_predict(
    model: object,
    features_list: list[dict[str, float]],
    threshold: float = 0.5,
) -> list[tuple[str, float]]:
    return [predict(model, f, threshold) for f in features_list]
