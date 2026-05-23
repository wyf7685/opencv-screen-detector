"""Image preprocessing for screen detector V2 inference."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import config


def preprocess_image(image_path: Path) -> np.ndarray:
    """Preprocess image for model inference.

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image as numpy array (1, C, H, W)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    image = (image - mean) / std

    # Convert to NCHW format
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(image, axis=0)  # Add batch dimension


def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Preprocess image from bytes.

    Args:
        image_bytes: Image data as bytes

    Returns:
        Preprocessed image as numpy array (1, C, H, W)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from bytes")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    image = (image - mean) / std

    # Convert to NCHW format
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(image, axis=0)  # Add batch dimension


def preprocess_pil_image(pil_image: Image.Image) -> np.ndarray:
    """Preprocess PIL image.

    Args:
        pil_image: PIL Image object

    Returns:
        Preprocessed image as numpy array (1, C, H, W)
    """
    # Convert to RGB if needed
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize
    pil_image = pil_image.resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    # Convert to numpy array
    image = np.array(pil_image, dtype=np.float32)

    # Normalize to [0, 1]
    image = image / 255.0

    # Apply ImageNet normalization
    mean = np.array(config.MEAN, dtype=np.float32)
    std = np.array(config.STD, dtype=np.float32)
    image = (image - mean) / std

    # Convert to NCHW format
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(image, axis=0)  # Add batch dimension
