"""ONNX predictor for screen detector V3 inference.

Orchestrates two-stage CNN inference with TTA, OOD detection, and
confidence tiering. Delegates model loading and FFT caching to
dedicated modules.
"""

import functools
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .config import settings
from .fft_service import FFTService
from .model_loader import ModelLoader
from .preprocess import normalize_rgb


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _run_stage(
    session: ort.InferenceSession,
    rgb_input: np.ndarray,
    fft_input: np.ndarray,
    class_names: list[str],
) -> dict:
    """Run inference on a single stage and return structured result."""
    rgb_name = session.get_inputs()[0].name
    fft_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name

    outputs: Any = session.run(
        [output_name],
        {rgb_name: rgb_input, fft_name: fft_input},
    )

    logits = outputs[0][0]
    probabilities = _softmax(logits)

    class_idx = np.argmax(probabilities)
    probs_dict = {
        name: float(prob)
        for name, prob in zip(class_names, probabilities, strict=False)
    }

    return {
        "class": class_names[class_idx],
        "confidence": float(probabilities[class_idx]),
        "probabilities": probs_dict,
    }


def _get_confidence_tier(confidence: float) -> dict:
    """Get confidence tier and recommended action."""
    if confidence >= settings.confidence_high:
        return {"confidence_tier": "high", "action": "accept"}
    if confidence >= settings.confidence_medium:
        return {"confidence_tier": "medium", "action": "review"}
    return {"confidence_tier": "low", "action": "ignore"}


def _check_ood(probabilities: dict[str, float]) -> bool:
    """Return True if max probability is below the OOD threshold."""
    return max(probabilities.values()) < settings.ood_threshold


class PredictTask:
    def __init__(self, models: ModelLoader, fft: FFTService, image_path: Path) -> None:
        self.models = models
        self.fft = fft
        self.image_path = image_path

    @functools.cached_property
    def original_image(self) -> np.ndarray:
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        return image

    @functools.cached_property
    def rgb_input(self) -> np.ndarray:
        return normalize_rgb(self.original_image)

    @functools.cached_property
    def fft_input(self) -> np.ndarray:
        return self.fft.get_fft_input(self.image_path)

    def run_stage1(self) -> dict:
        if not self.models.stage1_available:
            raise RuntimeError("Stage 1 model not loaded")

        stage1_names = ["natural", "screenshot"]
        flipped = cv2.flip(self.original_image, 1)
        flipped_rgb = normalize_rgb(flipped)
        flipped_fft = self.fft.get_fft_input_from_array(flipped)

        with self.models.get_stage1_session() as session:
            # Original
            result = _run_stage(session, self.rgb_input, self.fft_input, stage1_names)
            # Horizontal flip
            result_flip = _run_stage(session, flipped_rgb, flipped_fft, stage1_names)

        all_probs = [result["probabilities"], result_flip["probabilities"]]

        # Average probabilities
        avg_probs: dict[str, float] = {}
        for key in all_probs[0]:
            avg_probs[key] = float(np.mean([p[key] for p in all_probs]))

        class_idx = np.argmax(list(avg_probs.values()))
        class_name = list(avg_probs.keys())[class_idx]

        return {
            "class": class_name,
            "confidence": float(avg_probs[class_name]),
            "probabilities": avg_probs,
        }

    def run_stage2(self) -> dict:
        if not self.models.stage2_available:
            raise RuntimeError("Stage 2 model not loaded")

        stage2_names = ["screenshot", "screen_photo"]
        with self.models.get_stage2_session() as session:
            result = _run_stage(
                session,
                self.rgb_input,
                self.fft_input,
                stage2_names,
            )

        # Map Stage 2 result for external output
        is_photo = result["class"] == "screen_photo"
        final_class = "screen_photo" if is_photo else "screenshot"

        return {
            "class": final_class,
            "confidence": result["confidence"],
            "probabilities": {
                "screenshot": result["probabilities"]["screenshot"],
                "screen_photo": result["probabilities"]["screen_photo"],
            },
        }

    def run(self) -> dict:
        # Stage 1: natural vs screenshot (with TTA)
        s1 = self.run_stage1()

        # OOD Detection
        if _check_ood(s1["probabilities"]):
            max_prob = max(s1["probabilities"].values())
            return {
                "class": "unknown",
                "confidence": float(max_prob),
                "probabilities": s1["probabilities"],
                "stage": 1,
                "confidence_tier": "ood",
                "action": "ignore",
            }

        # If natural, return directly
        if s1["class"] == "natural":
            return {
                "class": "natural",
                "confidence": s1["confidence"],
                "probabilities": s1["probabilities"],
                "stage": 1,
                **_get_confidence_tier(s1["confidence"]),
            }

        # Stage 2: screenshot vs screen_photo
        s2 = self.run_stage2()

        return {
            **s2,
            "stage": 2,
            **_get_confidence_tier(s2["confidence"]),
        }


class ScreenDetectorPredictor:
    """Two-stage ONNX-based screen detector predictor."""

    def __init__(
        self,
        stage1_path: Path | None = None,
        stage2_path: Path | None = None,
    ) -> None:
        s1 = stage1_path or settings.stage1_model_path
        s2 = stage2_path or settings.stage2_model_path

        self._models = ModelLoader(s1, s2)
        self._fft = FFTService()

    # -- Properties --

    @property
    def stage1_available(self) -> bool:
        return self._models.stage1_available

    @property
    def stage2_available(self) -> bool:
        return self._models.stage2_available

    @property
    def model_available(self) -> bool:
        return self._models.model_available

    # -- Prediction --

    def predict(self, image_path: Path) -> dict:
        """Two-stage prediction with OOD detection.

        Returns:
            dict with keys: class, confidence, probabilities,
            stage, confidence_tier, action
        """
        return PredictTask(self._models, self._fft, image_path).run()

    def predict_batch(self, image_paths: list[Path]) -> list[dict]:
        """Predict on multiple images."""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result["filename"] = Path(image_path).name
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "filename": Path(image_path).name,
                        "error": str(e),
                    }
                )
        return results
