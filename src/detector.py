from dataclasses import dataclass
from pathlib import Path

from src.feature.artifact import analyze_artifact
from src.feature.banding import analyze_banding
from src.feature.chroma import analyze_chroma
from src.feature.color_noise import analyze_color_noise
from src.feature.display_content import analyze_display_content
from src.feature.format_score import analyze_format_score
from src.feature.frequency import analyze_frequency
from src.feature.illumination import analyze_illumination
from src.feature.moire import analyze_moire
from src.feature.overexposed import analyze_overexposed
from src.feature.perspective import analyze_perspective
from src.feature.reflection import analyze_reflection
from src.feature.sensor_noise import analyze_sensor_noise
from src.feature.rectangle import analyze_rectangle
from src.feature.softness import analyze_softness
from src.feature.subpixel_fringing import analyze_subpixel_fringing
from src.preprocess import preprocess_image
from src.scoring.ml_model import MLModel
from src.scoring.rules import classify_score, compute_score
from src.utils.image_io import load_image
from src.utils.image_metadata import camera_exif_score


@dataclass
class ScreenDetector:
    def detect(self, image_path: str | Path) -> dict:
        image = load_image(image_path)
        processed = preprocess_image(image)

        features = {
            "frequency": analyze_frequency(processed),
            "banding": analyze_banding(processed),
            "chroma": analyze_chroma(image),
            "softness": analyze_softness(image),
            "illumination": analyze_illumination(processed),
            "artifact": analyze_artifact(processed),
            "rectangle": analyze_rectangle(processed),
            "display_content": analyze_display_content(image),
            "overexposed": analyze_overexposed(processed),
            "perspective": analyze_perspective(image),
            "moire": analyze_moire(image),
            "reflection": analyze_reflection(image),
            "sensor_noise": analyze_sensor_noise(image),
            "subpixel_fringing": analyze_subpixel_fringing(image),
            "exif_camera": camera_exif_score(image_path),
            "format_score": analyze_format_score(image_path),
            "color_noise": analyze_color_noise(image),
        }

        rule_score = compute_score(features)
        model_probability = MLModel().predict(features)
        score = rule_score
        result = classify_score(score)

        return {
            "filename": Path(image_path).name,
            "score": round(score, 4),
            "result": result,
            "model_probability": round(model_probability, 4),
            "rule_score": round(rule_score, 4),
            "features": features,
        }