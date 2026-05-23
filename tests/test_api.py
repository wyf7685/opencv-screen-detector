"""Tests for API endpoints."""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient

    from inference.api import app

    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "stage1_model_loaded" in data
    assert "stage2_model_loaded" in data


def test_detect_upload_natural(client, sample_natural_image):
    """Test upload detection with natural image."""
    with Path(sample_natural_image).open("rb") as f:
        response = client.post(
            "/api/detect/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] in ["natural", "screen_like", "screen_photo", "unknown"]
    assert "confidence" in data
    assert "stage" in data


def test_detect_upload_screen_photo(client, sample_screen_photo_image):
    """Test upload detection with screen photo."""
    with Path(sample_screen_photo_image).open("rb") as f:
        response = client.post(
            "/api/detect/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] in ["natural", "screen_like", "screen_photo", "unknown"]


def test_detect_upload_invalid_file(client):
    """Test upload with invalid file."""
    response = client.post(
        "/api/detect/upload",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_detect_url_invalid(client):
    """Test URL detection with invalid URL."""
    response = client.post(
        "/api/detect",
        json={"url": "https://example.com/nonexistent.jpg"},
    )

    assert response.status_code in [400, 500]


def test_detect_response_structure(client, sample_natural_image):
    """Test detection response has all required fields."""
    with Path(sample_natural_image).open("rb") as f:
        response = client.post(
            "/api/detect/upload",
            files={"file": ("test.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()

    required_fields = [
        "image_id",
        "class_name",
        "confidence",
        "probabilities",
        "stage",
        "confidence_tier",
        "action",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
