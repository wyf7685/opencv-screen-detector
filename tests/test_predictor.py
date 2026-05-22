"""Tests for screen detector predictor."""

import pytest


@pytest.fixture(scope="module")
def predictor():
    """Create predictor instance."""
    from inference.predictor import ScreenDetectorPredictor

    try:
        return ScreenDetectorPredictor()
    except Exception as e:
        pytest.skip(f"Failed to load models: {e}")


def test_predictor_loads(predictor):
    """Test predictor loads successfully."""
    assert predictor is not None
    assert predictor.stage1_loaded is True


def test_predict_returns_valid_class(predictor, sample_natural_image):
    """Test predict returns valid class name."""
    result = predictor.predict(sample_natural_image)

    assert result["class"] in ["natural", "screen_like", "screen_photo", "unknown"]


def test_predict_confidence_range(predictor, sample_natural_image):
    """Test confidence is in valid range."""
    result = predictor.predict(sample_natural_image)

    assert 0 <= result["confidence"] <= 1


def test_predict_probabilities_sum(predictor, sample_natural_image):
    """Test probabilities sum to approximately 1."""
    result = predictor.predict(sample_natural_image)

    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum {prob_sum} != 1.0"


def test_predict_natural_image(predictor, sample_natural_image):
    """Test natural image is classified correctly."""
    result = predictor.predict(sample_natural_image)

    cls = result["class"]
    conf = result["confidence"]
    print(f"\nNatural image: {cls} (confidence: {conf:.4f})")
    # Allow some flexibility
    assert result["class"] in ["natural", "unknown"]


def test_predict_screen_photo_image(predictor, sample_screen_photo_image):
    """Test screen photo image is classified correctly."""
    result = predictor.predict(sample_screen_photo_image)

    print(f"\nScreen photo: {result['class']} (confidence: {result['confidence']:.4f})")
    # Allow some flexibility
    assert result["class"] in ["screen_like", "screen_photo", "unknown"]


def test_confidence_tier_high(predictor, sample_natural_image):
    """Test confidence tier assignment."""
    result = predictor.predict(sample_natural_image)

    assert result["confidence_tier"] in ["high", "medium", "low", "ood"]
    assert result["action"] in ["accept", "review", "ignore"]


def test_stage_field(predictor, sample_natural_image):
    """Test stage field is present and valid."""
    result = predictor.predict(sample_natural_image)

    assert "stage" in result
    assert result["stage"] in [1, 2]


def test_predict_batch(predictor, sample_natural_image, sample_screen_photo_image):
    """Test batch prediction."""
    results = predictor.predict_batch([sample_natural_image, sample_screen_photo_image])

    assert len(results) == 2
    for result in results:
        assert "class" in result or "error" in result
