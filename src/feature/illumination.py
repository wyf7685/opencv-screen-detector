from __future__ import annotations


def analyze_illumination(image) -> float:
    try:
        import numpy as np
    except Exception:
        return 0.0

    if image is None:
        return 0.0

    array = np.asarray(image, dtype=float)
    spread = float(array.std())
    return max(0.0, min(1.0, spread / 128.0))