"""Image preprocessing for screen detector V3 inference."""

from pathlib import Path

import cv2
import numpy as np

from . import config


def normalize_rgb(
    image: np.ndarray,
    image_size: int | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> np.ndarray:
    """Normalize a BGR or RGB numpy image to model input format (NCHW).

    Core preprocessing pipeline: BGR→RGB → resize → normalize → NCHW.

    Args:
        image: Input image as numpy array (H, W, C) in BGR or RGB format.
            If 3-channel, assumed BGR and converted to RGB.
        image_size: Target spatial size. Defaults to config.IMAGE_SIZE.
        mean: Per-channel normalization mean. Defaults to config.MEAN.
        std: Per-channel normalization std. Defaults to config.STD.

    Returns:
        Preprocessed image as numpy array (1, C, H, W).
    """
    size = image_size if image_size is not None else config.IMAGE_SIZE
    m = mean if mean is not None else config.MEAN
    s = std if std is not None else config.STD

    # Convert BGR to RGB if 3-channel
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, (size, size))

    # Normalize to [0, 1] then apply ImageNet stats
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(m, dtype=np.float32)) / np.array(s, dtype=np.float32)

    # HWC → CHW → NCHW
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)


def preprocess_image(image_path: Path) -> np.ndarray:
    """Load image from disk and preprocess for model inference.

    Args:
        image_path: Path to image file.

    Returns:
        Preprocessed image as numpy array (1, C, H, W).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return normalize_rgb(image)
