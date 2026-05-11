from __future__ import annotations

import unittest
from statistics import mean
from pathlib import Path

from src.detector import ScreenDetector
from src.scoring.rules import THRESHOLD


class TestInputSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(r"e:\github\opencv-screen-detector\data\input")
        cls.detector = ScreenDetector()

    def test_img_and_no_screen_are_normal(self) -> None:
        img_results = self._detect_folder("img")
        no_screen_results = self._detect_folder("no_screen")

        self.assertGreater(len(img_results), 0)
        self.assertGreater(len(no_screen_results), 0)
        img_avg = mean(score for _, _, score in img_results)
        no_screen_avg = mean(score for _, _, score in no_screen_results)
        self.assertLess(img_avg, THRESHOLD)
        self.assertLess(no_screen_avg, THRESHOLD)

    def test_no_screen_folder_has_at_most_one_positive(self) -> None:
        results = self._detect_folder("no_screen")

        self.assertGreater(len(results), 0)
        self.assertLessEqual(sum(1 for _, label, _ in results if label == "screen_photo"), 1)

    def test_photo_folder_scores_above_img(self) -> None:
        photo_results = self._detect_folder("photo")
        img_results = self._detect_folder("img")

        self.assertGreater(len(photo_results), 0)
        self.assertGreater(mean(score for _, _, score in photo_results), mean(score for _, _, score in img_results))

    def _detect_folder(self, folder_name: str) -> list[tuple[str, str, float]]:
        folder = self.root / folder_name
        results = []

        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            result = self.detector.detect(path)
            results.append((path.name, result["result"], float(result["score"])))

        return results