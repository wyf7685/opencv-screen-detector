from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.feature.artifact import analyze_artifact
from src.feature.banding import analyze_banding
from src.feature.blackscreen import analyze_blackscreen
from src.feature.chroma import analyze_chroma
from src.feature.color_noise import analyze_color_noise
from src.feature.display_content import analyze_display_content
from src.feature.format_score import analyze_format_score
from src.feature.frequency import analyze_frequency
from src.feature.illumination import analyze_illumination
from src.feature.moire import analyze_moire
from src.feature.overexposed import analyze_overexposed
from src.feature.perspective import analyze_perspective
from src.feature.rectangle import analyze_rectangle
from src.feature.reflection import analyze_reflection
from src.feature.sensor_noise import analyze_sensor_noise
from src.feature.softness import analyze_softness
from src.feature.subpixel_fringing import analyze_subpixel_fringing
from src.preprocess import preprocess_image
from src.scoring.ml_model import MLModel
from src.scoring.rules import RULE_ENGINE, THRESHOLD, compute_score
from src.utils.image_io import load_image
from src.utils.image_metadata import camera_exif_score


class ScreenDetector:
    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._ml_model = MLModel()

    def __del__(self) -> None:
        self._executor.shutdown(wait=False)

    def detect(self, image_path: str | Path) -> dict:
        image = load_image(image_path)
        processed = preprocess_image(image)

        feature_tasks = {
            "frequency": (analyze_frequency, processed),
            "banding": (analyze_banding, processed),
            "blackscreen": (analyze_blackscreen, image),
            "chroma": (analyze_chroma, image),
            "softness": (analyze_softness, image),
            "illumination": (analyze_illumination, processed),
            "artifact": (analyze_artifact, processed),
            "rectangle": (analyze_rectangle, processed),
            "display_content": (analyze_display_content, image),
            "overexposed": (analyze_overexposed, processed),
            "perspective": (analyze_perspective, image),
            "moire": (analyze_moire, image),
            "reflection": (analyze_reflection, image),
            "sensor_noise": (analyze_sensor_noise, image),
            "subpixel_fringing": (analyze_subpixel_fringing, image),
            "exif_camera": (camera_exif_score, image_path),
            "format_score": (analyze_format_score, image_path),
            "color_noise": (analyze_color_noise, image),
        }

        features = {}
        future_to_name = {
            self._executor.submit(func, arg): name
            for name, (func, arg) in feature_tasks.items()
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                features[name] = future.result()
            except Exception:
                features[name] = 0.0

        rule_score = compute_score(features)
        adjusted_score = RULE_ENGINE.apply(rule_score, features)

        if self._ml_model.is_loaded:
            ml_label, ml_probability = self._ml_model.predict(features)
            result = ml_label
            score = ml_probability
        else:
            ml_probability = 0.5
            score = adjusted_score
            result = "screen_photo" if score >= THRESHOLD else "normal"

        return {
            "filename": Path(image_path).name,
            "score": round(score, 4),
            "result": result,
            "model_probability": round(ml_probability, 4),
            "rule_score": round(rule_score, 4),
            "features": features,
        }
