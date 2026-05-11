from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(image_path):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to load images.") from exc

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    return image


def load_images(input_dir: str | Path) -> list[str]:
    directory = Path(input_dir)
    if not directory.exists():
        return []

    return sorted(
        str(path)
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )