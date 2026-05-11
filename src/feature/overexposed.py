from __future__ import annotations


def analyze_overexposed(image) -> float:
    try:
        import numpy as np
    except Exception:
        return 0.0

    if image is None:
        return 0.0

    array = np.asarray(image, dtype=float)
    return max(0.0, min(1.0, float((array > 245).mean()) * 10.0))