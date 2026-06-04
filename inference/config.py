"""Configuration for screen detector V3 inference.

Two-stage CNN + FFT Branch architecture.

Usage:
    # Default settings (reads from filesystem layout)
    from inference.config import settings

    # Override for testing
    from inference.config import configure
    configure(upload_dir=Path("/tmp/test_uploads"))
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Inference configuration with sensible defaults.

    All paths are derived from PROJECT_ROOT unless overridden.
    """

    # Root paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Derived paths (computed in __post_init__)
    inference_root: Path = field(init=False)
    models_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    upload_dir: Path = field(init=False)
    index_file: Path = field(init=False)

    # Model paths
    stage1_model_path: Path = field(init=False)
    stage2_model_path: Path = field(init=False)

    # Image processing
    image_size: int = 224
    input_channels: int = 3

    # Class names
    class_names: list[str] = field(
        default_factory=lambda: ["natural", "screenshot", "screen_photo"]
    )

    # Confidence thresholds
    confidence_high: float = 0.92  # >= accept
    confidence_medium: float = 0.75  # >= review
    ood_threshold: float = 0.50  # < unknown

    # API
    api_host: str = "0.0.0.0"  # noqa: S104
    api_port: int = 8325

    # Normalization (ImageNet stats)
    mean: list[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    std: list[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )

    def __post_init__(self) -> None:
        self.inference_root = self.project_root / "inference"
        self.models_dir = self.inference_root / "models"
        self.output_dir = self.inference_root / "output"
        self.data_dir = self.project_root / "data"
        self.upload_dir = self.data_dir / "upload"
        self.index_file = self.upload_dir / "index.json"
        self.stage1_model_path = (
            self.models_dir / "stage1_natural_vs_screenshot.onnx"
        )
        self.stage2_model_path = (
            self.models_dir / "stage2_screenshot_vs_screenphoto.onnx"
        )


# Global settings instance
settings = Settings()

# Module-level aliases for backward compatibility
PROJECT_ROOT = settings.project_root
INFERENCE_ROOT = settings.inference_root
MODELS_DIR = settings.models_dir
OUTPUT_DIR = settings.output_dir
DATA_DIR = settings.data_dir
UPLOAD_DIR = settings.upload_dir
STAGE1_MODEL_PATH = settings.stage1_model_path
STAGE2_MODEL_PATH = settings.stage2_model_path
IMAGE_SIZE = settings.image_size
INPUT_CHANNELS = settings.input_channels
CLASS_NAMES = settings.class_names
CONFIDENCE_HIGH = settings.confidence_high
CONFIDENCE_MEDIUM = settings.confidence_medium
OOD_THRESHOLD = settings.ood_threshold
API_HOST = settings.api_host
API_PORT = settings.api_port
MEAN = settings.mean
STD = settings.std


def configure(**kwargs: object) -> Settings:
    """Override settings at runtime (useful for testing).

    Args:
        **kwargs: Any field of Settings to override.

    Returns:
        The new global Settings instance.

    Example:
        configure(upload_dir=Path("/tmp/test"), api_port=9999)
    """
    global settings, PROJECT_ROOT, INFERENCE_ROOT, MODELS_DIR, OUTPUT_DIR
    global DATA_DIR, UPLOAD_DIR, STAGE1_MODEL_PATH, STAGE2_MODEL_PATH
    global IMAGE_SIZE, INPUT_CHANNELS, CLASS_NAMES, CONFIDENCE_HIGH
    global CONFIDENCE_MEDIUM, OOD_THRESHOLD, API_HOST, API_PORT, MEAN, STD

    # Build new settings with overrides
    current = settings
    merged = {
        "project_root": kwargs.get("project_root", current.project_root),
        "image_size": kwargs.get("image_size", current.image_size),
        "input_channels": kwargs.get("input_channels", current.input_channels),
        "class_names": kwargs.get("class_names", current.class_names),
        "confidence_high": kwargs.get("confidence_high", current.confidence_high),
        "confidence_medium": kwargs.get("confidence_medium", current.confidence_medium),
        "ood_threshold": kwargs.get("ood_threshold", current.ood_threshold),
        "api_host": kwargs.get("api_host", current.api_host),
        "api_port": kwargs.get("api_port", current.api_port),
        "mean": kwargs.get("mean", current.mean),
        "std": kwargs.get("std", current.std),
    }

    # Allow explicit path overrides
    new_settings = Settings(**merged)
    if "upload_dir" in kwargs:
        new_settings.upload_dir = Path(kwargs["upload_dir"])
        new_settings.index_file = new_settings.upload_dir / "index.json"
    if "index_file" in kwargs:
        new_settings.index_file = Path(kwargs["index_file"])
    if "models_dir" in kwargs:
        new_settings.models_dir = Path(kwargs["models_dir"])
        new_settings.stage1_model_path = (
            new_settings.models_dir / "stage1_natural_vs_screenlike.onnx"
        )
        new_settings.stage2_model_path = (
            new_settings.models_dir / "stage2_screenlike_vs_screenphoto.onnx"
        )

    settings = new_settings

    # Refresh module-level aliases
    PROJECT_ROOT = settings.project_root
    INFERENCE_ROOT = settings.inference_root
    MODELS_DIR = settings.models_dir
    OUTPUT_DIR = settings.output_dir
    DATA_DIR = settings.data_dir
    UPLOAD_DIR = settings.upload_dir
    STAGE1_MODEL_PATH = settings.stage1_model_path
    STAGE2_MODEL_PATH = settings.stage2_model_path
    IMAGE_SIZE = settings.image_size
    INPUT_CHANNELS = settings.input_channels
    CLASS_NAMES = settings.class_names
    CONFIDENCE_HIGH = settings.confidence_high
    CONFIDENCE_MEDIUM = settings.confidence_medium
    OOD_THRESHOLD = settings.ood_threshold
    API_HOST = settings.api_host
    API_PORT = settings.api_port
    MEAN = settings.mean
    STD = settings.std

    return settings
