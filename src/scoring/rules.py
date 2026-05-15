from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "rules.json"


@dataclass(frozen=True)
class Rule:
    name: str
    condition: dict
    delta: float


def _evaluate_condition(condition: dict, features: Mapping[str, float]) -> bool:
    if "all" in condition:
        return all(_evaluate_condition(sub, features) for sub in condition["all"])

    feature_name = condition["feature"]
    value = float(features.get(feature_name, 0.0))
    op = condition["op"]

    if op == ">":
        return value > condition["value"]
    if op == "<":
        return value < condition["value"]
    if op == ">=":
        return value >= condition["value"]
    if op == "<=":
        return value <= condition["value"]
    if op == "==":
        return value == condition["value"]
    if op == "between":
        return condition["low"] <= value <= condition["high"]
    raise ValueError(f"Unknown operator: {op}")


class RuleEngine:
    def __init__(self, rules: list[Rule]) -> None:
        self._rules = rules

    def apply(self, score: float, features: Mapping[str, float]) -> float:
        adjusted_score = score

        for rule in self._rules:
            if _evaluate_condition(rule.condition, features):
                adjusted_score += rule.delta

        return adjusted_score


def _load_config(
    config_path: Path,
) -> tuple[dict[str, float], float, RuleEngine]:
    with config_path.open(encoding="utf-8") as f:
        data = json.load(f)

    weights: dict[str, float] = data["weights"]
    threshold = float(data["threshold"])
    rules = [
        Rule(name=r["name"], condition=r["condition"], delta=r["delta"])
        for r in data["rules"]
    ]
    return weights, threshold, RuleEngine(rules)


WEIGHTS: dict[str, float]
THRESHOLD: float
RULE_ENGINE: RuleEngine

WEIGHTS, THRESHOLD, RULE_ENGINE = _load_config(_CONFIG_PATH)

FEATURE_NAMES: list[str] = list(WEIGHTS.keys())


def reload_config(config_path: Path | None = None) -> None:
    global WEIGHTS, THRESHOLD, RULE_ENGINE, FEATURE_NAMES
    path = config_path or _CONFIG_PATH
    WEIGHTS, THRESHOLD, RULE_ENGINE = _load_config(path)
    FEATURE_NAMES = list(WEIGHTS.keys())


def compute_score(features: dict[str, float]) -> float:
    return sum(
        float(features.get(name, 0.0)) * weight for name, weight in WEIGHTS.items()
    )


def classify_score(score: float, features: dict[str, float] | None = None) -> str:
    adjusted_score = score

    if features:
        adjusted_score = RULE_ENGINE.apply(score, features)

    return "screen_photo" if adjusted_score >= THRESHOLD else "normal"
