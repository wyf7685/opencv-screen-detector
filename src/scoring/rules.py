from __future__ import annotations


WEIGHTS = {
    "frequency": 0.00,
    "banding": 0.00,
    "chroma": 0.00,
    "softness": 0.00,
    "illumination": 0.00,
    "artifact": -0.188,
    "rectangle": 0.00,
    "display_content": 0.106,
    "overexposed": 0.00,
    "perspective": -0.012,
    "moire": 0.00,
    "reflection": -0.068,
    "sensor_noise": 0.070,
    "subpixel_fringing": 0.00,
    "exif_camera": 0.00,
    "format_score": 0.40,
    "color_noise": 0.188,
}

THRESHOLD = 0.248


def compute_score(features: dict[str, float]) -> float:
    return sum(float(features.get(name, 0.0)) * weight for name, weight in WEIGHTS.items())


def classify_score(score: float) -> str:
    return "screen_photo" if score >= THRESHOLD else "normal"
