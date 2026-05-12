"""Parameter optimization script for screen detector.

Iterates over different weight and threshold combinations to find optimal parameters
that maximize classification accuracy on the test dataset.

Usage:
    uv run python -m pytest test/test_parameter_optimization.py -s
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detector import ScreenDetector
from src.scoring.rules import WEIGHTS, THRESHOLD


def get_test_images():
    """Get all test images with their expected labels."""
    data_dir = project_root / "data" / "input"
    images = []

    # img/ directory: should all be "normal"
    img_dir = data_dir / "img"
    if img_dir.exists():
        for f in img_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                images.append((str(f), "normal"))

    # no_screen/ directory: should all be "normal"
    no_screen_dir = data_dir / "no_screen"
    if no_screen_dir.exists():
        for f in no_screen_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                images.append((str(f), "normal"))

    # photo/ directory: should all be "screen_photo"
    photo_dir = data_dir / "photo"
    if photo_dir.exists():
        for f in photo_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                images.append((str(f), "screen_photo"))

    return images


def compute_features(image_path: str) -> dict[str, float] | None:
    """Compute features for a single image."""
    try:
        detector = ScreenDetector()
        result = detector.detect(image_path)
        return result.get("features")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def compute_score_with_params(features: dict[str, float], weights: dict[str, float]) -> float:
    """Compute score with given weights."""
    return sum(float(features.get(name, 0.0)) * weight for name, weight in weights.items())


def classify_with_params(score: float, threshold: float) -> str:
    """Classify with given threshold."""
    return "screen_photo" if score >= threshold else "normal"


def evaluate_params(test_data: list[tuple[dict[str, float], str]], weights: dict[str, float], threshold: float) -> tuple[int, int, float]:
    """Evaluate parameters on test data.

    Returns:
        (correct, total, accuracy)
    """
    correct = 0
    total = len(test_data)

    for features, expected in test_data:
        score = compute_score_with_params(features, weights)
        predicted = classify_with_params(score, threshold)
        if predicted == expected:
            correct += 1

    return correct, total, correct / total if total > 0 else 0.0


def run_optimization():
    """Run parameter optimization."""
    print("=" * 80)
    print("Parameter Optimization for Screen Detector")
    print("=" * 80)

    # Step 1: Load test images
    print("\nStep 1: Loading test images...")
    test_images = get_test_images()
    print(f"Found {len(test_images)} test images")

    # Count by category
    normal_count = sum(1 for _, label in test_images if label == "normal")
    screen_photo_count = sum(1 for _, label in test_images if label == "screen_photo")
    print(f"  Normal: {normal_count}")
    print(f"  Screen photo: {screen_photo_count}")

    # Step 2: Compute features for all images
    print("\nStep 2: Computing features for all images...")
    test_data = []
    for image_path, expected in test_images:
        features = compute_features(image_path)
        if features is not None:
            test_data.append((features, expected))
        else:
            print(f"  Warning: Could not process {os.path.basename(image_path)}")

    print(f"Successfully computed features for {len(test_data)} images")

    # Step 3: Show current performance
    print("\nStep 3: Current performance with default parameters:")
    print(f"  Current weights: {WEIGHTS}")
    print(f"  Current threshold: {THRESHOLD}")

    correct, total, accuracy = evaluate_params(test_data, WEIGHTS, THRESHOLD)
    print(f"  Accuracy: {correct}/{total} = {accuracy:.4f}")

    # Step 4: Grid search over key parameters
    print("\nStep 4: Grid search over key parameters...")

    # Search over key weights to find optimal combination
    sensor_noise_range = [0.15, 0.18, 0.20, 0.215, 0.23]
    blackscreen_range = [0.03, 0.05, 0.07, 0.092]
    softness_range = [0.14, 0.16, 0.181, 0.20]
    frequency_range = [-0.30, -0.251, -0.20, -0.15]
    threshold_range = [0.22, 0.23, 0.24, 0.245, 0.25]

    best_accuracy = 0.0
    best_params = None
    total_combinations = (
        len(sensor_noise_range)
        * len(blackscreen_range)
        * len(softness_range)
        * len(frequency_range)
        * len(threshold_range)
    )
    print(f"  Testing {total_combinations} parameter combinations...")

    tested = 0
    for sn in sensor_noise_range:
        for bs in blackscreen_range:
            for sf in softness_range:
                for freq in frequency_range:
                    for th in threshold_range:
                        weights = WEIGHTS.copy()
                        weights["sensor_noise"] = sn
                        weights["blackscreen"] = bs
                        weights["softness"] = sf
                        weights["frequency"] = freq

                        correct, total, accuracy = evaluate_params(test_data, weights, th)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                "sensor_noise": sn,
                                "blackscreen": bs,
                                "softness": sf,
                                "frequency": freq,
                                "threshold": th,
                                "correct": correct,
                                "total": total,
                            }
                            print(f"  New best: accuracy={accuracy:.4f} ({correct}/{total}) "
                                  f"sn={sn}, bs={bs}, sf={sf}, freq={freq}, th={th}")

                        tested += 1
                        if tested % 100 == 0:
                            print(f"  Progress: {tested}/{total_combinations} ({tested/total_combinations*100:.1f}%)")

    # Step 5: Show results
    print("\n" + "=" * 80)
    print("Optimization Results")
    print("=" * 80)

    if best_params:
        print(f"\nBest parameters found:")
        print(f"  sensor_noise weight: {best_params['sensor_noise']}")
        print(f"  blackscreen weight: {best_params['blackscreen']}")
        print(f"  softness weight: {best_params['softness']}")
        print(f"  frequency weight: {best_params['frequency']}")
        print(f"  threshold: {best_params['threshold']}")
        print(f"  accuracy: {best_params['correct']}/{best_params['total']} = {best_accuracy:.4f}")

        # Show improvement
        print(f"\nImprovement over current parameters:")
        print(f"  Current accuracy: {correct}/{total} = {accuracy:.4f}")
        print(f"  Best accuracy: {best_params['correct']}/{best_params['total']} = {best_accuracy:.4f}")
        print(f"  Improvement: {best_accuracy - accuracy:.4f}")

        # Step 6: Detailed analysis with best parameters
        print("\nStep 6: Detailed analysis with best parameters:")
        best_weights = WEIGHTS.copy()
        best_weights["sensor_noise"] = best_params["sensor_noise"]
        best_weights["blackscreen"] = best_params["blackscreen"]
        best_weights["softness"] = best_params["softness"]
        best_weights["frequency"] = best_params["frequency"]
        best_threshold = best_params["threshold"]

        # Show misclassified images
        print("\nMisclassified images with best parameters:")
        misclassified = []
        for features, expected in test_data:
            score = compute_score_with_params(features, best_weights)
            predicted = classify_with_params(score, best_threshold)
            if predicted != expected:
                # Get filename from features (we need to track this)
                misclassified.append((score, predicted, expected))

        if misclassified:
            for score, predicted, expected in misclassified:
                print(f"  Score={score:.4f}, Predicted={predicted}, Expected={expected}")
        else:
            print("  No misclassified images!")

        # Step 7: Show weight comparison
        print("\nStep 7: Weight comparison:")
        print("  Parameter          Current    Optimized")
        print("  " + "-" * 45)
        for key in ["sensor_noise", "blackscreen", "softness", "frequency"]:
            current = WEIGHTS.get(key, 0.0)
            optimized = best_weights.get(key, 0.0)
            print(f"  {key:20s} {current:10.3f} {optimized:10.3f}")
        print(f"  {'threshold':20s} {THRESHOLD:10.3f} {best_threshold:10.3f}")

        # Step 8: Suggest code changes
        print("\nStep 8: Suggested code changes:")
        print("  # In src/scoring/rules.py:")
        print(f'  WEIGHTS = {{')
        for key, value in best_weights.items():
            print(f'      "{key}": {value},')
        print(f'  }}')
        print(f'  THRESHOLD = {best_threshold}')

    else:
        print("\nNo better parameters found!")

    return best_params, best_accuracy


def run_detailed_analysis():
    """Run detailed analysis showing per-image results."""
    print("\n" + "=" * 80)
    print("Detailed Per-Image Analysis")
    print("=" * 80)

    test_images = get_test_images()
    detector = ScreenDetector()

    results = []
    for image_path, expected in test_images:
        try:
            result = detector.detect(image_path)
            features = result.get("features")
            if features is None:
                continue
            score = compute_score_with_params(features, WEIGHTS)
            predicted = classify_with_params(score, THRESHOLD)

            filename = os.path.basename(image_path)
            results.append({
                "filename": filename,
                "expected": expected,
                "predicted": predicted,
                "score": score,
                "correct": predicted == expected,
                "features": features,
            })
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")

    # Show results
    print("\nResults:")
    print("  Filename                                    Expected      Predicted     Score    Correct")
    print("  " + "-" * 90)

    for r in results:
        correct_mark = "OK" if r["correct"] else "X"
        print(f"  {r['filename']:<42s} {r['expected']:<14s} {r['predicted']:<14s} {r['score']:.4f}    {correct_mark}")

    # Summary
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n  Total: {correct}/{total} = {correct/total:.4f}")

    # Show misclassified
    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        print(f"\n  Misclassified images ({len(misclassified)}):")
        for r in misclassified:
            print(f"    {r['filename']}: expected={r['expected']}, predicted={r['predicted']}, score={r['score']:.4f}")

            # Show top contributing features
            weights = WEIGHTS
            contributions = []
            for feat, val in r["features"].items():
                contrib = val * weights.get(feat, 0)
                contributions.append((feat, val, contrib))

            contributions.sort(key=lambda x: abs(x[2]), reverse=True)
            print(f"      Top features:")
            for feat, val, contrib in contributions[:5]:
                print(f"        {feat}: {val:.3f} * {weights.get(feat, 0):.3f} = {contrib:.4f}")


if __name__ == "__main__":
    # Run detailed analysis first
    run_detailed_analysis()

    # Then run optimization
    run_optimization()
