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
            input_dir = Path(tmp_dir) / "input"
            input_dir.mkdir()
            image_path = input_dir / "input.png"
            self._write_image(image_path)
            captured = {}

            def fake_save_json(data, output_path):
                captured["data"] = data
                captured["output_path"] = Path(output_path)

            with (
                patch("src.main.Path") as mock_path_cls,
                patch("src.main.save_json", side_effect=fake_save_json),
            ):
                # Make Path("./data/input") return our temp input dir
                original_path = Path

                def path_side_effect(path_str):
                    if path_str == "./data/input":
                        return original_path(input_dir)
                    return original_path(path_str)

                mock_path_cls.side_effect = path_side_effect
                # Make Path() constructor work normally for other uses
                mock_path_cls.return_value = original_path(tmp_dir)
                run_main()

            self.assertEqual(
                ["input.png"], [item["filename"] for item in captured["data"]]
            )
            expected_path = original_path("data/output/result.json")
            self.assertEqual(expected_path, captured["output_path"])

    @staticmethod
    def _write_image(path: Path) -> None:
        array = np.full((16, 16, 3), 180, dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), array)
