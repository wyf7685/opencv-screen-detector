"""Tests for FFT transform module."""

import numpy as np


def test_fft_spectrum_shape():
    """Test FFT spectrum output shape."""
    from inference.fft_transform import compute_fft_spectrum

    # Create dummy image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    result = compute_fft_spectrum(image, size=224)

    assert result.shape == (1, 1, 224, 224)


def test_fft_spectrum_different_sizes():
    """Test FFT spectrum with different sizes."""
    from inference.fft_transform import compute_fft_spectrum

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    for size in [128, 224, 256]:
        result = compute_fft_spectrum(image, size=size)
        assert result.shape == (1, 1, size, size)


def test_fft_spectrum_range():
    """Test FFT spectrum values are in reasonable range."""
    from inference.fft_transform import compute_fft_spectrum

    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    result = compute_fft_spectrum(image, size=224)

    # After normalization, values should be in reasonable range
    assert result.min() > -5  # Not too negative
    assert result.max() < 10  # Not too large


def test_fft_spectrum_grayscale_input():
    """Test FFT spectrum with grayscale input."""
    from inference.fft_transform import compute_fft_spectrum

    # Grayscale image
    image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

    result = compute_fft_spectrum(image, size=224)

    assert result.shape == (1, 1, 224, 224)


def test_fft_spectrum_color_input():
    """Test FFT spectrum with color input."""
    from inference.fft_transform import compute_fft_spectrum

    # Color image (BGR)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    result = compute_fft_spectrum(image, size=224)

    assert result.shape == (1, 1, 224, 224)


def test_fft_spectrum_from_bytes():
    """Test FFT spectrum from bytes input."""
    import cv2

    from inference.fft_transform import compute_fft_spectrum_from_bytes

    # Create dummy image and encode to bytes
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", image)
    image_bytes = buffer.tobytes()

    result = compute_fft_spectrum_from_bytes(image_bytes, size=224)

    assert result.shape == (1, 1, 224, 224)


def test_fft_spectrum_real_image(sample_natural_image):
    """Test FFT spectrum with real image file."""
    import cv2

    from inference.fft_transform import compute_fft_spectrum

    image = cv2.imread(sample_natural_image)
    assert image is not None, f"Failed to load image: {sample_natural_image}"

    result = compute_fft_spectrum(image, size=224)

    assert result.shape == (1, 1, 224, 224)
    assert not np.isnan(result).any(), "FFT spectrum contains NaN"
    assert not np.isinf(result).any(), "FFT spectrum contains Inf"
