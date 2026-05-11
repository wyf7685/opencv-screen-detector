from __future__ import annotations


def analyze_artifact(image) -> float:
    try:
        import numpy as np
    except Exception:
        return 0.0

    if image is None:
        return 0.0

    array = np.asarray(image, dtype=float)
    bright_ratio = float((array > 240).mean())
    return max(0.0, min(1.0, bright_ratio * 8.0))