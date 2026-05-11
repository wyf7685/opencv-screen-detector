from __future__ import annotations

import unittest

import numpy as np

from src.feature.artifact import analyze_artifact
from src.feature.banding import analyze_banding
from src.feature.chroma import analyze_chroma
from src.feature.moire import analyze_moire
from src.feature.overexposed import analyze_overexposed
from src.feature.perspective import analyze_perspective
from src.feature.reflection import analyze_reflection
from src.feature.sensor_noise import analyze_sensor_noise
from src.feature.subpixel_fringing import analyze_subpixel_fringing
from src.feature.frequency import analyze_frequency
from src.feature.illumination import analyze_illumination
from src.feature.rectangle import analyze_rectangle
from src.feature.softness import analyze_softness
from src.preprocess import preprocess_image
from src.scoring.rules import THRESHOLD, WEIGHTS, classify_score, compute_score


class TestPreprocessAndFeatures(unittest.TestCase):
    def test_preprocess_image_converts_color_image_to_grayscale(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :, 0] = 10
        image[:, :, 1] = 20
        image[:, :, 2] = 30

        processed = preprocess_image(image)

        self.assertEqual((16, 16), processed.shape)
        self.assertEqual(2, processed.ndim)

    def test_preprocess_image_returns_none_for_none_input(self) -> None:
        self.assertIsNone(preprocess_image(None))

    def test_feature_scores_stay_in_range(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24, 0] = 255
        image[4:28, 4:28, 1] = 64
        image[10:18, 10:18, 2] = 128
        gray = preprocess_image(image)

        scores = [
            analyze_frequency(gray),
            analyze_banding(gray),
            analyze_chroma(image),
            analyze_softness(image),
            analyze_illumination(image),
            analyze_artifact(image),
            analyze_rectangle(image),
        ]

        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_rule_scoring_and_classification(self) -> None:
        features = {
            "frequency": 1.0,
            "banding": 1.0,
            "chroma": 1.0,
            "softness": 1.0,
            "illumination": 1.0,
            "artifact": 1.0,
            "rectangle": 1.0,
            "display_content": 1.0,
            "overexposed": 1.0,
            "perspective": 1.0,
            "moire": 1.0,
            "reflection": 1.0,
            "sensor_noise": 1.0,
            "subpixel_fringing": 1.0,
            "exif_camera": 1.0,
            "format_score": 1.0,
            "color_noise": 1.0,
        }

        score = compute_score(features)

        self.assertAlmostEqual(score, sum(WEIGHTS.values()))
        self.assertEqual("normal", classify_score(score))
        self.assertEqual("normal", classify_score(THRESHOLD - 0.01))
        self.assertEqual("screen_photo", classify_score(THRESHOLD + 0.01))