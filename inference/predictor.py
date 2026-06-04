"""ONNX predictor for screen detector V3 inference.

Orchestrates two-stage CNN inference with TTA, OOD detection, and
confidence tiering. Delegates model loading and FFT caching to
dedicated modules.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .config import settings
from .fft_service import FFTService
from .model_loader import ModelLoader
from .preprocess import normalize_rgb, preprocess_image


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


class ScreenDetectorPredictor:
    """Two-stage ONNX-based screen detector predictor."""

    def __init__(
        self,
        stage1_path: str | None = None,
        stage2_path: str | None = None,
    ) -> None:
        s1 = stage1_path or str(settings.stage1_model_path)
        s2 = stage2_path or str(settings.stage2_model_path)

        self._models = ModelLoader(s1, s2)
        self._fft = FFTService()

    # -- Properties for backward compatibility --

    @property
    def stage1_session(self) -> ort.InferenceSession | None:
        return self._models.stage1_session

    @property
    def stage2_session(self) -> ort.InferenceSession | None:
        return self._models.stage2_session

    @property
    def stage1_loaded(self) -> bool:
        return self._models.stage1_loaded

    @property
    def stage2_loaded(self) -> bool:
        return self._models.stage2_loaded

    @property
    def model_loaded(self) -> bool:
        return self._models.model_loaded

    # -- Prediction --

    def predict_with_tta(self, image_path: Path) -> dict:
        """Test Time Augmentation: original + horizontal flip, averaged."""
        if not self._models.stage1_loaded:
            raise RuntimeError("Stage 1 model not loaded")

        rgb_input = preprocess_image(image_path)
        fft_input = self._fft.get_fft_input(image_path)

        # Original
        stage1_names = ["natural", "screenshot"]
        result = _run_stage(
            self._models.stage1_session, rgb_input, fft_input, stage1_names
        )
        all_probs = [result["probabilities"]]

        # Horizontal flip
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        flipped = cv2.flip(image, 1)
        flipped_rgb = normalize_rgb(flipped)
        flipped_fft = self._fft.get_fft_input_from_array(flipped)
        result_flip = _run_stage(
            self._models.stage1_session,
            flipped_rgb,
            flipped_fft,
            stage1_names,
        )
        all_probs.append(result_flip["probabilities"])

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

    def predict(self, image_path: Path) -> dict:
        """Two-stage prediction with OOD detection.

        Returns:
            dict with keys: class, confidence, probabilities,
            stage, confidence_tier, action
        """
        if not self._models.stage1_loaded:
            raise RuntimeError("Stage 1 model not loaded")

        # Stage 1: natural vs screenshot (with TTA)
        s1_result = self.predict_with_tta(image_path)

        # OOD Detection
        if _check_ood(s1_result["probabilities"]):
            max_prob = max(s1_result["probabilities"].values())
            return {
                "class": "unknown",
                "confidence": float(max_prob),
                "probabilities": s1_result["probabilities"],
                "stage": 1,
                "confidence_tier": "ood",
                "action": "ignore",
            }

        # If natural, return directly
        if s1_result["class"] == "natural":
            return {
                "class": "natural",
                "confidence": s1_result["confidence"],
                "probabilities": s1_result["probabilities"],
                "stage": 1,
                **_get_confidence_tier(s1_result["confidence"]),
            }

        # Stage 2: screenshot vs screen_photo
        if not self._models.stage2_loaded:
            raise RuntimeError("Stage 2 model not loaded")

        rgb_input = preprocess_image(image_path)
        fft_input = self._fft.get_fft_input(image_path)

        stage2_names = ["screenshot", "screen_photo"]
        s2_result = _run_stage(
            self._models.stage2_session,
            rgb_input,
            fft_input,
            stage2_names,
        )

        # Map Stage 2 result for external output
        is_photo = s2_result["class"] == "screen_photo"
        final_class = "screen_photo" if is_photo else "screenshot"

        return {
            "class": final_class,
            "confidence": s2_result["confidence"],
            "probabilities": {
                "screenshot": s2_result["probabilities"]["screenshot"],
                "screen_photo": s2_result["probabilities"]["screen_photo"],
            },
            "stage": 2,
            **_get_confidence_tier(s2_result["confidence"]),
        }

    def predict_batch(self, image_paths: list) -> list:
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
