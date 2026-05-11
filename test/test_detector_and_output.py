from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.detector import ScreenDetector
from src.main import main as run_main
from src.utils.json_export import save_json


class TestDetectorAndOutput(unittest.TestCase):
    def test_detector_returns_expected_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "screen.png"
            self._write_image(image_path)

            result = ScreenDetector().detect(image_path)

            self.assertEqual("screen.png", result["filename"])
            self.assertIn("score", result)
            self.assertIn("result", result)
            self.assertIn("model_probability", result)
            self.assertIn("rule_score", result)
            self.assertIn("features", result)
            self.assertIsInstance(result["features"], dict)
            self.assertIn("display_content", result["features"])
            self.assertIn("overexposed", result["features"])
            self.assertIn("perspective", result["features"])
            self.assertIn("moire", result["features"])
            self.assertIn("reflection", result["features"])
            self.assertIn("sensor_noise", result["features"])
            self.assertIn("subpixel_fringing", result["features"])
            self.assertIn("exif_camera", result["features"])

    def test_save_json_writes_utf8_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "result.json"
            data = [{"filename": "a.png", "score": 0.5}]

            save_json(data, output_path)

            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data, written)

    def test_main_orchestrates_detector_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "input.png"
            self._write_image(image_path)
            captured = {}

            def fake_save_json(data, output_path):
                captured["data"] = data
                captured["output_path"] = Path(output_path)

            with patch("src.main.load_images", return_value=[str(image_path)]), patch(
                "src.main.save_json", side_effect=fake_save_json
            ):
                run_main()

            self.assertEqual(["input.png"], [item["filename"] for item in captured["data"]])
            self.assertEqual(Path("data/output/result.json"), captured["output_path"])

    @staticmethod
    def _write_image(path: Path) -> None:
        array = np.full((16, 16, 3), 180, dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), array)