"""Configuration for screen detector V3 inference.

Two-stage CNN + FFT Branch architecture.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
INFERENCE_ROOT = PROJECT_ROOT / "inference"
MODELS_DIR = INFERENCE_ROOT / "models"
OUTPUT_DIR = INFERENCE_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "upload"

# Two-stage model paths
STAGE1_MODEL_PATH = MODELS_DIR / "stage1_natural_vs_screenlike.onnx"
STAGE2_MODEL_PATH = MODELS_DIR / "stage2_screenlike_vs_screenphoto.onnx"

# Image processing
IMAGE_SIZE = 224
INPUT_CHANNELS = 3

# Class names
CLASS_NAMES = ["natural", "screen_like", "screen_photo"]

# Confidence thresholds
CONFIDENCE_HIGH = 0.92  # >= 直接输出
CONFIDENCE_MEDIUM = 0.75  # >= 人工审核
OOD_THRESHOLD = 0.50  # < 此值为 unknown

# API
API_HOST = "0.0.0.0"  # noqa: S104
API_PORT = 8325

# Normalization (ImageNet stats)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
