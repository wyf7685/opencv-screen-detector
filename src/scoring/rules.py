from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

WEIGHTS = {
    "frequency": -0.15,
    "banding": 0.048,
    "blackscreen": 0.03,
    "chroma": -0.055,
    "softness": 0.181,
    "illumination": 0.131,
    "artifact": -0.081,
    "rectangle": 0.111,
    "display_content": -0.116,
    "overexposed": -0.002,
    "perspective": -0.013,
    "moire": -0.060,
    "reflection": -0.008,
    "sensor_noise": 0.15,
    "subpixel_fringing": -0.009,
    "exif_camera": 0.054,
    "format_score": 0.088,
    "color_noise": 0.020,
}

THRESHOLD = 0.23


def _feature_value(features: Mapping[str, float], name: str) -> float:
    return float(features.get(name, 0.0))


@dataclass(frozen=True)
class Rule:
    name: str
    condition: Callable[[Mapping[str, float]], bool]
    delta: float


class RuleEngine:
    def __init__(self, rules: list[Rule]) -> None:
        self._rules = rules

    def apply(self, score: float, features: Mapping[str, float]) -> float:
        adjusted_score = score

        for rule in self._rules:
            if rule.condition(features):
                adjusted_score += rule.delta

        return adjusted_score


def _build_rule_engine() -> RuleEngine:
    return RuleEngine(
        [
            Rule(
                name="ui_screenshot_with_high_sensor_noise",
                condition=lambda features: (
                    _feature_value(features, "sensor_noise") > 0.95
                    and _feature_value(features, "softness") < 0.74
                    and _feature_value(features, "artifact") < 0.10
                    and _feature_value(features, "moire") > 0.80
                ),
                delta=-0.12,
            ),
            Rule(
                name="black_screen_photo",
                condition=lambda features: (
                    _feature_value(features, "moire") > 0.95
                    and _feature_value(features, "softness") > 0.95
                    and _feature_value(features, "blackscreen") > 0.50
                ),
                delta=0.06,
            ),
            Rule(
                name="clean_screenshot_with_screen_like_features",
                condition=lambda features: (
                    _feature_value(features, "softness") > 0.90
                    and _feature_value(features, "moire") > 0.95
                    and _feature_value(features, "artifact") < 0.08
                    and _feature_value(features, "rectangle") > 0.10
                ),
                delta=-0.06,
            ),
            Rule(
                name="high_softness_high_moire_low_artifact",
                condition=lambda features: (
                    _feature_value(features, "softness") > 0.80
                    and _feature_value(features, "moire") > 0.90
                    and _feature_value(features, "artifact") < 0.10
                    and _feature_value(features, "rectangle") > 0.15
                ),
                delta=-0.08,
            ),
            Rule(
                name="normal_image_with_geometric_structure",
                condition=lambda features: (
                    _feature_value(features, "sensor_noise") < 0.85
                    and _feature_value(features, "softness") > 0.85
                    and _feature_value(features, "moire") > 0.95
                    and _feature_value(features, "artifact") >= 0.10
                    and _feature_value(features, "rectangle") > 0.13
                ),
                delta=-0.06,
            ),
        ]
    )


RULE_ENGINE = _build_rule_engine()


def compute_score(features: dict[str, float]) -> float:
    return sum(
        float(features.get(name, 0.0)) * weight for name, weight in WEIGHTS.items()
    )


def classify_score(score: float, features: dict[str, float] | None = None) -> str:
    adjusted_score = score

    if features:
        adjusted_score = RULE_ENGINE.apply(score, features)

    return "screen_photo" if adjusted_score >= THRESHOLD else "normal"
