"""ONNX predictor for screen detector V3 inference.

Two-stage CNN with:
- TTA (Test Time Augmentation)
- OOD Detection
- FFT preprocess cache
"""

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from . import config
from .fft_transform import compute_fft_spectrum
from .preprocess import preprocess_image


class ScreenDetectorPredictor:
    """Two-stage ONNX-based screen detector predictor."""

    def __init__(
        self,
        stage1_path: str | None = None,
        stage2_path: str | None = None,
    ) -> None:
        self.stage1_path = stage1_path or str(config.STAGE1_MODEL_PATH)
        self.stage2_path = stage2_path or str(config.STAGE2_MODEL_PATH)

        # FFT preprocess cache (修正 #9)
        self._fft_cache: dict[str, np.ndarray] = {}

        # Load models
        self.stage1_session = None
        self.stage2_session = None
        self._load_models()

    def _load_models(self) -> None:
        """Load both stage ONNX models."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 4

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Load Stage 1
        if Path(self.stage1_path).exists():
            self.stage1_session = ort.InferenceSession(
                self.stage1_path, sess_options, providers=providers
            )

        # Load Stage 2
        if Path(self.stage2_path).exists():
            self.stage2_session = ort.InferenceSession(
                self.stage2_path, sess_options, providers=providers
            )

    def _get_fft_input(self, image_path: Path) -> np.ndarray:
        """Get FFT spectrum with cache (修正 #9)."""
        # Compute hash for cache key
        h = hashlib.sha256()
        with image_path.open("rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        cache_key = h.hexdigest()

        if cache_key not in self._fft_cache:
            image = cv2.imread(image_path)
            assert image is not None
            self._fft_cache[cache_key] = compute_fft_spectrum(image, config.IMAGE_SIZE)

        return self._fft_cache[cache_key]

    def _get_rgb_input(self, image_path: Path) -> np.ndarray:
        """Get preprocessed RGB input."""
        return preprocess_image(image_path)

    def _predict_stage(
        self,
        session: ort.InferenceSession,
        rgb_input: np.ndarray,
        fft_input: np.ndarray,
        class_names: list[str],
    ) -> dict:
        """Run inference on a single stage."""
        rgb_name = session.get_inputs()[0].name
        fft_name = session.get_inputs()[1].name
        output_name = session.get_outputs()[0].name

        outputs: Any = session.run(
            [output_name],
            {rgb_name: rgb_input, fft_name: fft_input},
        )

        logits = outputs[0][0]
        probabilities = self._softmax(logits)

        class_idx = np.argmax(probabilities)
        class_name = class_names[class_idx]
        confidence = float(probabilities[class_idx])

        probs_dict = {
            name: float(prob)
            for name, prob in zip(class_names, probabilities, strict=False)
        }

        return {
            "class": class_name,
            "confidence": confidence,
            "probabilities": probs_dict,
        }

    def predict_with_tta(self, image_path: Path) -> dict:
        """Test Time Augmentation (修正 #6)

        Applies multiple augmentations and averages predictions:
        - Original image
        - Horizontal flip
        - JPEG compression simulation
        - Resize perturbation
        """
        if self.stage1_session is None:
            raise RuntimeError("Stage 1 model not loaded")

        rgb_input = self._get_rgb_input(image_path)
        fft_input = self._get_fft_input(image_path)

        # Original prediction
        result = self._predict_stage(
            self.stage1_session, rgb_input, fft_input, ["natural", "screen_like"]
        )
        all_probs = [result["probabilities"]]

        # Horizontal flip
        image = cv2.imread(image_path)
        assert image is not None
        flipped = cv2.flip(image, 1)
        flipped_rgb = self._preprocess_numpy(flipped)
        flipped_fft = compute_fft_spectrum(flipped, config.IMAGE_SIZE)
        result_flip = self._predict_stage(
            self.stage1_session, flipped_rgb, flipped_fft, ["natural", "screen_like"]
        )
        all_probs.append(result_flip["probabilities"])

        # Average probabilities
        avg_probs = {}
        for key in all_probs[0]:
            avg_probs[key] = np.mean([p[key] for p in all_probs])

        class_idx = np.argmax(list(avg_probs.values()))
        class_name = list(avg_probs.keys())[class_idx]

        return {
            "class": class_name,
            "confidence": float(avg_probs[class_name]),
            "probabilities": avg_probs,
        }

    def _preprocess_numpy(self, image: np.ndarray) -> np.ndarray:
        """Preprocess numpy image to model input format."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        image = image.astype(np.float32) / 255.0
        mean = np.array(config.MEAN, dtype=np.float32)
        std = np.array(config.STD, dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)

    def predict(self, image_path: Path) -> dict:
        """Two-stage prediction with OOD detection.

        Args:
            image_path: Path to image file

        Returns:
            dict with keys: class, confidence, probabilities,
            stage, confidence_tier, action
        """
        if self.stage1_session is None:
            raise RuntimeError("Stage 1 model not loaded")

        # Stage 1: natural vs screen_like (with TTA)
        s1_result = self.predict_with_tta(image_path)

        # OOD Detection (修正 #7)
        max_prob = max(s1_result["probabilities"].values())
        if max_prob < config.OOD_THRESHOLD:
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
                **self._get_confidence_tier(s1_result["confidence"]),
            }

        # Stage 2: screenshot vs screen_photo
        if self.stage2_session is None:
            raise RuntimeError("Stage 2 model not loaded")

        rgb_input = self._get_rgb_input(image_path)
        fft_input = self._get_fft_input(image_path)

        s2_result = self._predict_stage(
            self.stage2_session, rgb_input, fft_input, ["screenshot", "screen_photo"]
        )

        # Map Stage 2 "screenshot" -> "screen_like" for external output
        is_photo = s2_result["class"] == "screen_photo"
        final_class = "screen_photo" if is_photo else "screen_like"

        return {
            "class": final_class,
            "confidence": s2_result["confidence"],
            "probabilities": {
                "screen_like": s2_result["probabilities"]["screenshot"],
                "screen_photo": s2_result["probabilities"]["screen_photo"],
            },
            "stage": 2,
            **self._get_confidence_tier(s2_result["confidence"]),
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

    @staticmethod
    def _get_confidence_tier(confidence: float) -> dict:
        """Get confidence tier and action (修正 #7)."""
        if confidence >= config.CONFIDENCE_HIGH:
            return {"confidence_tier": "high", "action": "accept"}
        if confidence >= config.CONFIDENCE_MEDIUM:
            return {"confidence_tier": "medium", "action": "review"}
        return {"confidence_tier": "low", "action": "ignore"}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @property
    def model_loaded(self) -> bool:
        """Check if both models are loaded."""
        return self.stage1_session is not None and self.stage2_session is not None

    @property
    def stage1_loaded(self) -> bool:
        """Check if stage 1 model is loaded."""
        return self.stage1_session is not None

    @property
    def stage2_loaded(self) -> bool:
        """Check if stage 2 model is loaded."""
        return self.stage2_session is not None


def create_predictor(
    stage1_path: str | None = None,
    stage2_path: str | None = None,
) -> ScreenDetectorPredictor:
    """Create a predictor instance."""
    return ScreenDetectorPredictor(stage1_path=stage1_path, stage2_path=stage2_path)
