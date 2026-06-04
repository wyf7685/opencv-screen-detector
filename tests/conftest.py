"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root):
    """Data input directory."""
    return project_root / "data" / "input"


@pytest.fixture
def sample_natural_image(data_dir):
    """Return a path to a natural_photo image."""
    natural_dir = data_dir / "natural_photo"
    if not natural_dir.exists():
        pytest.skip("natural_photo directory not found")

    # Check root directory first
    for f in natural_dir.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return str(f)

    # Check subdirectories
    for subdir in natural_dir.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    return str(f)

    pytest.skip("No natural_photo images found")


@pytest.fixture
def sample_screen_photo_image(data_dir):
    """Return a path to a screen_photo image."""
    screen_dir = data_dir / "screen_photo"
    if not screen_dir.exists():
        pytest.skip("screen_photo directory not found")

    for f in screen_dir.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return str(f)

    pytest.skip("No screen_photo images found")


@pytest.fixture
def sample_screenlike_image(data_dir):
    """Return a path to a screenshot image."""
    screenshot_dir = data_dir / "screenshot"
    if not screenshot_dir.exists():
        pytest.skip("screenshot directory not found")

    for f in screenshot_dir.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return str(f)

    pytest.skip("No screenshot images found")


@pytest.fixture
def sample_screenshot_image(data_dir):
    """Return a path to a screenshot image."""
    screenshot_dir = data_dir / "screenshot"
    if not screenshot_dir.exists():
        pytest.skip("screenshot directory not found")

    for f in screenshot_dir.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return str(f)

    pytest.skip("No screenshot images found")
