from __future__ import annotations

from pathlib import Path

from src.ml.predict import load_model, predict

_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "model"
    / "screen_detector.pkl"
)


class MLModel:
    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else _MODEL_PATH
        self._model = load_model(str(path))

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self, features: dict[str, float], threshold: float = 0.5
    ) -> tuple[str, float]:
        if self._model is None:
            return "unknown", 0.5
        return predict(self._model, features, threshold)
