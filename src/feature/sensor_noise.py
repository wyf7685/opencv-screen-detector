from __future__ import annotations


def analyze_sensor_noise(image) -> float:
    try:
        import cv2
        import numpy as np
    except Exception:
        return 0.0

    if image is None:
        return 0.0

    array = np.asarray(image)
    if array.ndim == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    else:
        gray = array

    gray = gray.astype(float)
    low_pass = cv2.GaussianBlur(gray, (9, 9), 0)
    residual = gray - low_pass
    edge_mask = cv2.Canny(gray.astype(np.uint8), 50, 150) == 0
    if not np.any(edge_mask):
        edge_mask = np.ones_like(gray, dtype=bool)

    noise = residual[edge_mask]
    if noise.size == 0:
        return 0.0

    score = float(np.std(noise)) / 18.0
    return max(0.0, min(1.0, score))
