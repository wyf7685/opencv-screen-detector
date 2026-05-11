from __future__ import annotations

from pathlib import Path


def analyze_format_score(image_path: str | Path) -> float:
    try:
        from PIL import Image
        with Image.open(str(image_path)) as img:
            fmt = img.format
        if fmt == "PNG":
            return 0.0
        elif fmt in ("JPEG", "JPG"):
            return 0.5
        else:
            return 0.25
    except Exception:
        suffix = Path(image_path).suffix.lower()
        if suffix == ".png":
            return 0.0
        elif suffix in (".jpg", ".jpeg"):
            return 0.5
        return 0.25
