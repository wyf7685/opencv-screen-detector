"""Export trained models to ONNX format.

Exports two models:
- Stage 1: natural vs screenshot
- Stage 2: screenshot vs screen_photo

Both models have dual inputs (RGB + FFT) with dynamic batch axes (修正 #8).
"""

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from . import config
from .model import load_model


def export_to_onnx(
    checkpoint_path: str,
    onnx_path: str,
    opset_version: int = 11,
    verify: bool = True,
):
    """Export PyTorch model to ONNX format with dual inputs.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        onnx_path: Path to save ONNX model
        opset_version: ONNX opset version
        verify: Whether to verify exported model
    """
    device = "cpu"

    # Load model
    model = load_model(checkpoint_path, device=device)
    model.eval()

    # Create dummy inputs (RGB + FFT)
    dummy_rgb = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    dummy_fft = torch.randn(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)

    # Export to ONNX with dual inputs and dynamic axes (修正 #8)
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_fft),
        onnx_path,
        opset_version=opset_version,
        input_names=["rgb_input", "fft_input"],
        output_names=["output"],
        dynamic_axes={
            "rgb_input": {0: "batch_size"},
            "fft_input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    if verify:
        verify_onnx_model(onnx_path, model, dummy_rgb, dummy_fft)

    return onnx_path


def verify_onnx_model(onnx_path, pytorch_model, dummy_rgb, dummy_fft) -> None:
    """Verify ONNX model produces same output as PyTorch model."""
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(onnx_path)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_rgb, dummy_fft).numpy()

    # ONNX inference
    ort_inputs = {
        "rgb_input": dummy_rgb.numpy(),
        "fft_input": dummy_fft.numpy(),
    }
    onnx_output = ort_session.run(None, ort_inputs)[0]

    np.testing.assert_allclose(  # pyright: ignore[reportCallIssue]
        pytorch_output,
        onnx_output,  # pyright: ignore[reportArgumentType]
        rtol=1e-03,
        atol=1e-05,
    )

    print(f"ONNX model verified: {onnx_path}")


def export_to_torchscript(
    checkpoint_path: str,
    torchscript_path: str,
):
    """Export PyTorch model to TorchScript format."""
    device = "cpu"

    model = load_model(checkpoint_path, device=device)
    model.eval()

    dummy_rgb = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    dummy_fft = torch.randn(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)

    scripted_model: Any = torch.jit.trace(model, (dummy_rgb, dummy_fft))
    scripted_model.save(torchscript_path)

    return torchscript_path


def main() -> None:
    """Main entry point for exporting both stage models."""
    models_dir = config.PROJECT_ROOT / "inference" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Export Stage 1: natural vs screenshot
    stage1_checkpoint = str(config.CHECKPOINT_DIR / "stage1_best.pth")
    stage1_onnx = str(models_dir / "stage1_natural_vs_screenshot.onnx")

    if Path(stage1_checkpoint).exists():
        print("Exporting Stage 1 model...")
        export_to_onnx(
            checkpoint_path=stage1_checkpoint,
            onnx_path=stage1_onnx,
            opset_version=11,
            verify=True,
        )
        print(f"Stage 1 exported to: {stage1_onnx}")
    else:
        print(f"Stage 1 checkpoint not found: {stage1_checkpoint}")

    # Export Stage 2: screenshot vs screen_photo
    stage2_checkpoint = str(config.CHECKPOINT_DIR / "stage2_best.pth")
    stage2_onnx = str(models_dir / "stage2_screenshot_vs_screenphoto.onnx")

    if Path(stage2_checkpoint).exists():
        print("Exporting Stage 2 model...")
        export_to_onnx(
            checkpoint_path=stage2_checkpoint,
            onnx_path=stage2_onnx,
            opset_version=11,
            verify=True,
        )
        print(f"Stage 2 exported to: {stage2_onnx}")
    else:
        print(f"Stage 2 checkpoint not found: {stage2_checkpoint}")
