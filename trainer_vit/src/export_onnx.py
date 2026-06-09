"""ViT ONNX export.

Export ViT model to ONNX format for deployment.
"""

from pathlib import Path

import onnx
import torch
from loguru import logger
from onnx import checker

from .model import ViTScreenDetector, load_vit_model


def export_to_onnx(
    model: ViTScreenDetector,
    output_path: str,
    image_size: int = 224,
    opset_version: int = 17,
) -> None:
    """Export ViT model to ONNX format.

    Args:
        model: ViT model to export
        output_path: Path to save ONNX model
        image_size: Input image size
        opset_version: ONNX opset version
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Export
    logger.info(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Verify
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    checker.check_model(onnx_model)

    # Get file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    logger.info("ONNX model exported successfully!")
    logger.info(f"File size: {file_size:.2f} MB")
    logger.info(f"Opset version: {opset_version}")


def export_from_checkpoint(
    checkpoint_path: str,
    output_path: str = "outputs/vit_model.onnx",
    image_size: int = 224,
    opset_version: int = 17,
) -> None:
    """Export ViT model from checkpoint to ONNX.

    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path to save ONNX model
        image_size: Input image size
        opset_version: ONNX opset version
    """
    # Load model
    logger.info(f"Loading model from {checkpoint_path}...")
    model = load_vit_model(checkpoint_path)

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Export
    export_to_onnx(model, output_path, image_size, opset_version)


def main() -> None:
    """Main export entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Export ViT model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/vit_model.onnx",
        help="Path to save ONNX model",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    export_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_size=args.image_size,
        opset_version=args.opset_version,
    )


if __name__ == "__main__":
    main()
