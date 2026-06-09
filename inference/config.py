"""Configuration for screen detector V3 inference.

Two-stage CNN + FFT Branch architecture.

Usage:
    # Default settings (reads from filesystem layout)
    from inference.config import settings

    # Override for testing
    from inference.config import configure
    configure(upload_dir=Path("/tmp/test_uploads"))
"""

import functools
from pathlib import Path
from typing import Any, overload
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Inference configuration with sensible defaults.

    All paths are derived from project_root unless overridden.
    """

    # Root paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)

    # Derived paths
    @property
    def inference_root(self) -> Path:
        return self.project_root / "inference"

    @functools.cached_property
    def models_dir(self) -> Path:
        return self.inference_root / "models"

    @property
    def output_dir(self) -> Path:
        return self.inference_root / "output"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @functools.cached_property
    def upload_dir(self) -> Path:
        return self.data_dir / "upload"

    @functools.cached_property
    def index_file(self) -> Path:
        return self.upload_dir / "index.json"

    # Model paths
    @functools.cached_property
    def stage1_model_path(self) -> Path:
        return self.models_dir / "stage1_natural_vs_screenshot.onnx"

    @functools.cached_property
    def stage2_model_path(self) -> Path:
        return self.models_dir / "stage2_screenshot_vs_screenphoto.onnx"

    # Image processing
    image_size: int = 224
    input_channels: int = 3

    # Class names
    class_names: list[str] = Field(
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
    mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])


# Global settings instance
settings = Settings()


@overload
def configure() -> Settings: ...
@overload
def configure(
    *,
    project_root: Path | None = None,
    upload_dir: Path | None = None,
    index_file: Path | None = None,
    models_dir: Path | None = None,
    image_size: int | None = None,
    input_channels: int | None = None,
    class_names: list[str] | None = None,
    confidence_high: float | None = None,
    confidence_medium: float | None = None,
    ood_threshold: float | None = None,
    api_host: str | None = None,
    api_port: int | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> Settings: ...


def configure(**kwargs: Any) -> Settings:
    """Override settings at runtime (useful for testing).

    Args:
        **kwargs: Any field of Settings to override.

    Returns:
        The updated settings object.

    Example:
        configure(upload_dir=Path("/tmp/test"), api_port=9999)
    """
    global settings

    if not kwargs:
        return settings

    upload_dir = kwargs.pop("upload_dir", None)
    index_file = kwargs.pop("index_file", None)
    models_dir = kwargs.pop("models_dir", None)

    # Create new settings with provided kwargs
    new_settings = Settings.model_validate({**settings.model_dump(), **kwargs})

    # Update cached properties if provided
    if upload_dir is not None:
        # Clear cached_property cache and set new value
        if "upload_dir" in settings.__dict__:
            del settings.__dict__["upload_dir"]
        settings.__dict__["upload_dir"] = Path(upload_dir)

        if "index_file" in settings.__dict__:
            del settings.__dict__["index_file"]
        settings.__dict__["index_file"] = Path(upload_dir) / "index.json"
    elif index_file is not None:
        if "index_file" in settings.__dict__:
            del settings.__dict__["index_file"]
        settings.__dict__["index_file"] = Path(index_file)

    if models_dir is not None:
        if "models_dir" in settings.__dict__:
            del settings.__dict__["models_dir"]
        settings.__dict__["models_dir"] = Path(models_dir)

        if "stage1_model_path" in settings.__dict__:
            del settings.__dict__["stage1_model_path"]
        settings.__dict__["stage1_model_path"] = (
            Path(models_dir) / "stage1_natural_vs_screenshot.onnx"
        )

        if "stage2_model_path" in settings.__dict__:
            del settings.__dict__["stage2_model_path"]
        settings.__dict__["stage2_model_path"] = (
            Path(models_dir) / "stage2_screenshot_vs_screenphoto.onnx"
        )

    # Update regular fields
    for key in Settings.model_fields:
        if key not in ("upload_dir", "index_file", "models_dir",
                       "stage1_model_path", "stage2_model_path"):
            setattr(settings, key, getattr(new_settings, key))

    return settings
