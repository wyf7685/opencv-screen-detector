from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.utils.image_io import load_image, load_images


class TestImageIO(unittest.TestCase):
    def test_load_images_filters_and_sorts_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            self._write_image(directory / "b.jpg")
            self._write_image(directory / "a.png")
            (directory / "c.txt").write_text("ignore", encoding="utf-8")

            loaded = load_images(directory)

            self.assertEqual(
                [str(directory / "a.png"), str(directory / "b.jpg")], loaded
            )

    def test_load_image_reads_written_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            self._write_image(image_path)

            image = load_image(image_path)

            self.assertIsNotNone(image)
            self.assertEqual((8, 8, 3), image.shape)

    @staticmethod
    def _write_image(path: Path) -> None:
        array = np.full((8, 8, 3), 127, dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), array)
