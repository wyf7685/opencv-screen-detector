"""ViT validation and metrics computation.

Compute accuracy, precision, recall, F1-score, confusion matrix.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .model import ViTScreenDetector, load_vit_model


def validate_model(
    model: ViTScreenDetector,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate model on validation set.

    Args:
        model: ViT model
        val_loader: Validation data loader
        device: Device to validate on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds) * 100.0
    avg_loss = total_loss / len(val_loader)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_preds,
        "labels": all_labels,
    }


def compute_metrics(
    model: ViTScreenDetector,
    val_loader: DataLoader,
    device: torch.device,
    label_names: list[str],
) -> dict:
    """Compute comprehensive metrics for model evaluation.

    Args:
        model: ViT model
        val_loader: Validation data loader
        device: Device to evaluate on
        label_names: List of class names

    Returns:
        Dictionary with all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Build per-class metrics
    classes = {}
    for i, name in enumerate(label_names):
        classes[name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
        }

    # Build confusion matrix dict
    confusion = {}
    for i, name in enumerate(label_names):
        confusion[name] = {}
        for j, name2 in enumerate(label_names):
            confusion[name][name2] = int(cm[i][j])

    return {
        "model": "ViT-B16",
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_score": float(f1_macro),
        "classes": classes,
        "confusion_matrix": confusion,
        "total_samples": len(all_labels),
    }


def validate_from_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str = "outputs",
    device: str = "cpu",
) -> dict:
    """Validate model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to data directory
        output_dir: Path to output directory
        device: Device to validate on

    Returns:
        Dictionary with validation metrics
    """
    from .dataset import LABEL_NAMES, create_dataloaders

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = load_vit_model(checkpoint_path, str(device))

    # Create dataloaders
    logger.info("Creating dataloaders...")
    _, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
        val_split=0.2,
    )

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(model, val_loader, device, LABEL_NAMES)

    # Save metrics
    metrics_path = output_path / "metrics.json"
    with Path(metrics_path).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("Validation complete!")
    logger.info(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Precision: {metrics['precision'] * 100:.2f}%")
    logger.info(f"Recall: {metrics['recall'] * 100:.2f}%")
    logger.info(f"F1-score: {metrics['f1_score'] * 100:.2f}%")
    logger.info(f"Metrics saved to: {metrics_path}")

    return metrics


def main() -> None:
    """Main validation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate ViT model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../data/input",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)",
    )

    args = parser.parse_args()

    validate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
