"""Training module for screen detector V3.

Two-stage training with Mixed Precision:
- Stage 1: natural vs screen_like
- Stage 2: screenshot vs screen_photo
"""

# pyright: reportPrivateImportUsage=none
import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import config
from .augment import get_train_transforms, get_val_transforms
from .dataset import create_data_loaders
from .model import create_model, load_model, save_model
from .validate import (
    plot_confusion_matrix,
    plot_training_history,
    print_metrics,
    validate_model,
)


def train_one_epoch(
    model: nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion,
    optimizer,
    device: str = "cpu",
    use_amp: bool = True,
):
    """Train model for one epoch with Mixed Precision."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Mixed Precision (修正 #10)
    amp_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    for rgb, fft, labels in train_loader:
        rgb = rgb.to(device)
        fft = fft.to(device)
        labels = labels.to(device)

        # Forward pass with AMP
        with torch.amp.autocast("cuda" if device == "cuda" else "cpu", enabled=use_amp):
            outputs = model(rgb, fft)
            loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item() * rgb.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_stage(
    data_map: dict[str, list[str]],
    stage_name: str,
    class_names: list[str],
    data_dir: Path | None = None,
    epochs_head: int = config.EPOCHS_HEAD,
    epochs_finetune: int = config.EPOCHS_FINETUNE,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    device: str | None = None,
):
    """Train a single stage of the two-stage CNN.

    Args:
        data_map: Data mapping {class_name: [source_dirs]}
        stage_name: Stage identifier (e.g., "stage1", "stage2")
        class_names: Class names for this stage
        data_dir: Data directory
        epochs_head: Epochs for head training
        epochs_finetune: Epochs for fine-tuning
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if data_dir is None:
        data_dir = config.DATA_DIR

    print(f"\n{'=' * 60}")
    print(f"Training {stage_name}: {' vs '.join(class_names)}")
    print(f"{'=' * 60}")

    # Create data loaders
    train_loader, val_loader, full_dataset = create_data_loaders(
        data_map=data_map,
        data_dir=data_dir,
        transform_train=get_train_transforms(),
        transform_val=get_val_transforms(),
        batch_size=batch_size,
    )

    print(f"Dataset size: {len(full_dataset)} images")
    train_size = int(len(full_dataset) * config.TRAIN_VAL_SPLIT)
    val_size = len(full_dataset) - train_size
    print(f"Train/Val split: {train_size}/{val_size}")

    # Create model
    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
    )
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    # ==========================================
    # Stage A: Train classification head
    # ==========================================
    print(f"\n[Stage A] Training classification head ({epochs_head} epochs)")

    optimizer = optim.AdamW(
        list(model.classifier.parameters()) + list(model.freq_branch.parameters()),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_head)

    for epoch in range(epochs_head):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_metrics = validate_model(model, val_loader, device, class_names)
        val_acc = val_metrics["accuracy"]

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(0.0)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model,
                str(config.CHECKPOINT_DIR / f"{stage_name}_best.pth"),
                epoch=epoch,
                optimizer_state_dict=optimizer.state_dict(),
                best_val_acc=best_val_acc,
            )

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch + 1}/{epochs_head} - "
            f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
            f"Val Acc: {val_acc:.4f} - Time: {elapsed:.1f}s"
        )

    # ==========================================
    # Stage B: Fine-tune with unfrozen layers
    # ==========================================
    print(f"\n[Stage B] Fine-tuning ({epochs_finetune} epochs)")

    model.unfreeze_backbone(num_layers=2)

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": learning_rate * 0.1},
            {"params": model.freq_branch.parameters(), "lr": learning_rate * 0.1},
            {"params": model.classifier.parameters(), "lr": learning_rate},
            {"params": model.spatial_norm.parameters(), "lr": learning_rate},
            {"params": model.freq_norm.parameters(), "lr": learning_rate},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_finetune)

    for epoch in range(epochs_finetune):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_metrics = validate_model(model, val_loader, device, class_names)
        val_acc = val_metrics["accuracy"]

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(0.0)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model,
                str(config.CHECKPOINT_DIR / f"{stage_name}_best.pth"),
                epoch=epochs_head + epoch,
                optimizer_state_dict=optimizer.state_dict(),
                best_val_acc=best_val_acc,
            )

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch + 1}/{epochs_finetune} - "
            f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
            f"Val Acc: {val_acc:.4f} - Time: {elapsed:.1f}s"
        )

    # ==========================================
    # Final evaluation
    # ==========================================

    best_model = load_model(
        str(config.CHECKPOINT_DIR / f"{stage_name}_best.pth"),
        device=device,
    )
    best_model = best_model.to(device)

    final_metrics = validate_model(best_model, val_loader, device, class_names)
    print_metrics(final_metrics, class_names)

    plot_confusion_matrix(
        final_metrics,
        class_names,
        save_path=str(config.LOG_DIR / f"{stage_name}_confusion_matrix.png"),
    )

    plot_training_history(
        history,
        save_path=str(config.LOG_DIR / f"{stage_name}_training_history.png"),
    )

    save_model(
        model,
        str(config.CHECKPOINT_DIR / f"{stage_name}_final.pth"),
        epoch=epochs_head + epochs_finetune - 1,
        best_val_acc=best_val_acc,
    )

    return model, history, final_metrics


def main():
    """Main entry point for two-stage training."""
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: natural vs screen_like
    _, _, metrics1 = train_stage(
        data_map=config.STAGE1_DATA_MAP,
        stage_name="stage1",
        class_names=config.CLASS_NAMES_STAGE1,
    )

    # Stage 2: screenshot vs screen_photo
    _, _, metrics2 = train_stage(
        data_map=config.STAGE2_DATA_MAP,
        stage_name="stage2",
        class_names=config.CLASS_NAMES_STAGE2,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Stage 1 Accuracy: {metrics1['accuracy']:.4f}")
    print(f"Stage 2 Accuracy: {metrics2['accuracy']:.4f}")
    print("=" * 60)
