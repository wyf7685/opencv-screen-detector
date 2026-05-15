from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

from src.scoring.rules import FEATURE_NAMES

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "model"
_CACHE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "features_cache.json"
)

# Label mapping for 3-class classification
LABEL_NAMES = ["screenshot", "normal_photo", "screen_photo"]
NUM_CLASSES = len(LABEL_NAMES)


def load_training_data(
    data_dir: Path, use_cache: bool = True
) -> tuple[list[dict[str, float]], list[int]]:
    if use_cache and _CACHE_PATH.exists():
        return _load_from_cache(_CACHE_PATH)

    from src.detector import ScreenDetector

    detector = ScreenDetector()
    features_list: list[dict[str, float]] = []
    labels: list[int] = []

    # img/ → screenshot (0)
    for image_path in sorted((data_dir / "img").glob("*")):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            result = detector.detect(image_path)
            features_list.append(result["features"])
            labels.append(0)

    # no_screen/ → normal_photo (1)
    for image_path in sorted((data_dir / "no_screen").glob("*")):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            result = detector.detect(image_path)
            features_list.append(result["features"])
            labels.append(1)

    # photo/ → screen_photo (2)
    for image_path in sorted((data_dir / "photo").glob("*")):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            result = detector.detect(image_path)
            features_list.append(result["features"])
            labels.append(2)

    if use_cache:
        _save_to_cache(_CACHE_PATH, features_list, labels)

    return features_list, labels


def _load_from_cache(
    cache_path: Path,
) -> tuple[list[dict[str, float]], list[int]]:
    with cache_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["features"], data["labels"]


def _save_to_cache(
    cache_path: Path,
    features_list: list[dict[str, float]],
    labels: list[int],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({"features": features_list, "labels": labels}, f)


def prepare_training_data(
    features_list: list[dict[str, float]], labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[f.get(name, 0.0) for name in FEATURE_NAMES] for f in features_list]
    )
    y = np.array(labels)
    return x, y


def train_model(x: np.ndarray, y: np.ndarray) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": NUM_CLASSES,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    dataset = lgb.Dataset(x, label=y, feature_name=FEATURE_NAMES)
    return lgb.train(params, dataset, num_boost_round=1000)


def save_model(model: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> object | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "input"
    model_path = _MODEL_DIR / "screen_detector.pkl"

    logger.info("Loading training data...")
    features_list, labels = load_training_data(data_dir)

    # Count samples per class
    class_counts = dict.fromkeys(LABEL_NAMES, 0)
    for label in labels:
        class_counts[LABEL_NAMES[label]] += 1
    logger.info("Loaded %d samples: %s", len(labels), class_counts)

    x, y = prepare_training_data(features_list, labels)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Training LightGBM model...")
    model = train_model(x_train, y_train)

    # Predict on validation set
    y_pred_prob = model.predict(x_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    accuracy = float(np.mean(y_pred == y_val))
    logger.info("Validation accuracy: %.4f", accuracy)

    # Per-class accuracy
    for i, name in enumerate(LABEL_NAMES):
        mask = y_val == i
        if mask.sum() > 0:
            class_acc = float(np.mean(y_pred[mask] == y_val[mask]))
            logger.info("  %s accuracy: %.4f (%d samples)", name, class_acc, mask.sum())

    logger.info("Training on full dataset...")
    full_model = train_model(x, y)

    save_model(full_model, model_path)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
