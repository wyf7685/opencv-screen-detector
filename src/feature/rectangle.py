from __future__ import annotations


def analyze_rectangle(image) -> float:
    try:
        import cv2
        import numpy as np
    except Exception:
        return 0.0

    if image is None:
        return 0.0

    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(array, 80, 160)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size or 1)
    return max(0.0, min(1.0, edge_ratio * 4.0))