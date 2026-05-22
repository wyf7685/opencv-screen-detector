"""Accuracy tests for screen detector V3.

Tests accuracy based on folder structure:
- natural_photo/ -> expected "natural"
- screen_like/ -> expected "screen_like"
- screenshot/ -> expected "screen_like"
- hard_negative/ -> expected "screen_like"
- screen_photo/ -> expected "screen_photo"
"""

from pathlib import Path

import pytest

from inference.predictor import ScreenDetectorPredictor

# Folder to expected class mapping
FOLDER_TO_EXPECTED = {
    "natural_photo": "natural",
    "screen_like": "screen_like",
    "screenshot": "screen_like",
    "hard_negative": "screen_like",
    "screen_photo": "screen_photo",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@pytest.fixture(scope="module")
def predictor():
    """Create predictor instance."""
    try:
        return ScreenDetectorPredictor()
    except Exception as e:
        pytest.skip(f"Failed to load models: {e}")


def collect_images(data_dir: Path, folder: str) -> list[Path]:
    """Collect all images from a folder (recursive)."""
    folder_path = data_dir / folder
    if not folder_path.exists():
        return []
    return [f for f in folder_path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]


def test_accuracy_natural_folder(predictor, data_dir):
    """Test accuracy on natural_photo/ folder."""
    images = collect_images(data_dir, "natural_photo")
    if not images:
        pytest.skip("No natural_photo images found")

    correct = 0
    total = 0
    errors = []

    for img_path in images:
        try:
            result = predictor.predict(str(img_path))
            total += 1
            if result["class"] == "natural":
                correct += 1
            else:
                errors.append(f"{img_path.name}: predicted {result['class']}")
        except Exception as e:
            errors.append(f"{img_path.name}: error {e}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nNatural accuracy: {accuracy:.2%} ({correct}/{total})")
    if errors:
        print(f"Errors: {errors[:5]}")

    assert accuracy >= 0.80, f"Natural accuracy {accuracy:.2%} below 80%"


def test_accuracy_screen_photo_folder(predictor, data_dir):
    """Test accuracy on screen_photo/ folder."""
    images = collect_images(data_dir, "screen_photo")
    if not images:
        pytest.skip("No screen_photo images found")

    correct = 0
    total = 0
    errors = []

    for img_path in images:
        try:
            result = predictor.predict(str(img_path))
            total += 1
            if result["class"] == "screen_photo":
                correct += 1
            else:
                errors.append(f"{img_path.name}: predicted {result['class']}")
        except Exception as e:
            errors.append(f"{img_path.name}: error {e}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nScreen photo accuracy: {accuracy:.2%} ({correct}/{total})")
    if errors:
        print(f"Errors: {errors[:5]}")

    assert accuracy >= 0.75, f"Screen photo accuracy {accuracy:.2%} below 75%"


def test_accuracy_screenlike_folder(predictor, data_dir):
    """Test accuracy on screen_like/ and screenshot/ folders."""
    all_images = []
    for folder in ["screen_like", "screenshot", "hard_negative"]:
        all_images.extend(collect_images(data_dir, folder))

    if not all_images:
        pytest.skip("No screen_like/screenshot images found")

    correct = 0
    total = 0
    errors = []

    for img_path in all_images:
        try:
            result = predictor.predict(str(img_path))
            total += 1
            if result["class"] == "screen_like":
                correct += 1
            else:
                errors.append(f"{img_path.name}: predicted {result['class']}")
        except Exception as e:
            errors.append(f"{img_path.name}: error {e}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nScreen-like accuracy: {accuracy:.2%} ({correct}/{total})")
    if errors:
        print(f"Errors: {errors[:5]}")

    assert accuracy >= 0.65, f"Screen-like accuracy {accuracy:.2%} below 65%"


def test_overall_accuracy(predictor, data_dir):
    """Test overall accuracy across all folders."""
    results_by_class = {}

    for folder, expected_class in FOLDER_TO_EXPECTED.items():
        images = collect_images(data_dir, folder)
        if not images:
            continue

        for img_path in images:
            try:
                result = predictor.predict(str(img_path))
                predicted = result["class"]

                if expected_class not in results_by_class:
                    results_by_class[expected_class] = {
                        "correct": 0,
                        "total": 0,
                        "errors": [],
                    }

                results_by_class[expected_class]["total"] += 1
                if predicted == expected_class:
                    results_by_class[expected_class]["correct"] += 1
                else:
                    results_by_class[expected_class]["errors"].append(
                        f"{img_path.name}: {predicted}"
                    )
            except Exception as e:
                if expected_class not in results_by_class:
                    results_by_class[expected_class] = {
                        "correct": 0,
                        "total": 0,
                        "errors": [],
                    }
                results_by_class[expected_class]["errors"].append(
                    f"{img_path.name}: {e}"
                )

    # Print results
    print("\n" + "=" * 60)
    print("Accuracy Results")
    print("=" * 60)

    total_correct = 0
    total_count = 0

    for class_name, stats in results_by_class.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        total_correct += stats["correct"]
        total_count += stats["total"]
        print(f"{class_name}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
        if stats["errors"]:
            print(f"  Errors: {stats['errors'][:3]}")

    overall_accuracy = total_correct / total_count if total_count > 0 else 0
    print(f"\nOverall: {overall_accuracy:.2%} ({total_correct}/{total_count})")
    print("=" * 60)

    assert overall_accuracy >= 0.70, (
        f"Overall accuracy {overall_accuracy:.2%} below 70%"
    )


def test_predictor_returns_valid_structure(predictor, sample_natural_image):
    """Test predictor returns valid result structure."""
    result = predictor.predict(sample_natural_image)

    assert "class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "stage" in result
    assert "confidence_tier" in result
    assert "action" in result

    assert result["class"] in ["natural", "screen_like", "screen_photo", "unknown"]
    assert 0 <= result["confidence"] <= 1
    assert result["stage"] in [1, 2]
    assert result["confidence_tier"] in ["high", "medium", "low", "ood"]
    assert result["action"] in ["accept", "review", "ignore"]
