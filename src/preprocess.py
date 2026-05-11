from __future__ import annotations


def preprocess_image(image):
    try:
        import cv2
    except Exception:
        return image

    if image is None:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)