"""ViT training pipeline for screen detector.

改进版本，包含以下技术：
1. 使用预训练权重 (ImageNet-1k)
2. ViT-Small 模型 (22M 参数)
3. 两阶段迁移学习
4. Mixup + CutMix 数据增强
5. Stochastic Depth (drop_path_rate=0.1)
6. Label Smoothing (0.1)
7. Layer-wise Learning Rate Decay (LLRD)
8. 增加训练轮数 (100 epochs)
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import LABEL_NAMES, create_dataloaders
from .model import ViTScreenDetector, create_vit_model, save_vit_model
from .transforms import MixUpCutMixWrapper
from .validate import compute_metrics, validate_model

# Default config - 改进版本
DEFAULT_CONFIG = {
    "model_name": "vit_small_patch16_224",
    "num_classes": 3,
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "stage1_epochs": 10,  # Head only (增加)
    "stage2_epochs": 90,  # Fine-tune (大幅增加)
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "val_split": 0.2,
    "seed": 42,
    "drop_path_rate": 0.1,  # Stochastic Depth
    "label_smoothing": 0.1,  # Label Smoothing
    "mixup_prob": 0.5,  # Mixup/CutMix 概率
    "use_mixup": True,  # 是否使用 Mixup/CutMix
}


def get_layerwise_lr_params(
    model: ViTScreenDetector, lr: float, decay: float = 0.85
) -> list[dict]:
    """Get layer-wise learning rate parameters for LLRD.

    Args:
        model: ViT model
        lr: Base learning rate
        decay: Learning rate decay factor

    Returns:
        List of parameter groups with different learning rates
    """
    param_groups = []

    # Get model layers (for ViT-Small: 12 transformer blocks)
    if hasattr(model.model, "blocks"):
        blocks = model.model.blocks
        num_layers = len(blocks)

        # Embedding layer (lowest learning rate)
        embed_params = list(model.model.patch_embed.parameters())
        if hasattr(model.model, "cls_token"):
            embed_params.append(model.model.cls_token)
        if hasattr(model.model, "pos_embed"):
            embed_params.append(model.model.pos_embed)
        param_groups.append({
            "params": embed_params,
            "lr": lr * (decay ** num_layers),
            "name": "embeddings"
        })

        # Transformer blocks (increasing learning rate)
        for i, block in enumerate(blocks):
            block_lr = lr * (decay ** (num_layers - i - 1))
            param_groups.append({
                "params": list(block.parameters()),
                "lr": block_lr,
                "name": f"block_{i}"
            })

        # Head (highest learning rate)
        head_params = list(model.model.head.parameters())
        param_groups.append({
            "params": head_params,
            "lr": lr,
            "name": "head"
        })
    else:
        # Fallback: all parameters with same learning rate
        param_groups.append({
            "params": list(model.parameters()),
            "lr": lr,
            "name": "all"
        })

    return param_groups


def train_one_epoch(
    model: ViTScreenDetector,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    mixup_cutmix: MixUpCutMixWrapper | None = None,
) -> dict:
    """Train model for one epoch.

    Args:
        model: ViT model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        mixup_cutmix: Mixup/CutMix wrapper (optional)

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup/CutMix if enabled
        if mixup_cutmix is not None:
            images, labels_onehot = mixup_cutmix(images, labels)
            # For accuracy calculation, use original labels
            _, labels_orig = labels.max(1) if labels.dim() > 1 else (None, labels)
        else:
            labels_onehot = labels
            labels_orig = labels

        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        if labels_onehot.dim() > 1:
            # One-hot labels (from Mixup/CutMix)
            log_probs = torch.log_softmax(outputs, dim=1)
            loss = -torch.sum(log_probs * labels_onehot, dim=1).mean()
        else:
            # Standard labels
            loss = criterion(outputs, labels_onehot)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels_orig.size(0)
        correct += predicted.eq(labels_orig).sum().item()

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def train_model(
    data_dir: str,
    output_dir: str = "outputs",
    config: dict | None = None,
) -> dict:
    """Train ViT model with improved techniques.

    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        config: Training configuration (optional)

    Returns:
        Dictionary with training results
    """
    # Merge config
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        val_split=cfg["val_split"],
        image_size=cfg["image_size"],
        seed=cfg["seed"],
    )

    # Create model with improved settings
    logger.info("Creating ViT-Small model with pretrained weights...")
    model = create_vit_model(
        model_name=cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=True,  # 使用预训练权重
        freeze_backbone=False,
        drop_path_rate=cfg["drop_path_rate"],  # Stochastic Depth
    ).to(device)

    # Loss function with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    # Mixup/CutMix wrapper
    mixup_cutmix = None
    if cfg["use_mixup"]:
        mixup_cutmix = MixUpCutMixWrapper(
            num_classes=cfg["num_classes"],
            prob=cfg["mixup_prob"]
        )
        logger.info("Mixup/CutMix enabled")

    # Stage 1: Train head only
    logger.info("=== Stage 1: Training classification head ===")
    model.freeze_backbone()

    # Use layer-wise learning rate decay
    param_groups = get_layerwise_lr_params(model, cfg["learning_rate"])
    optimizer_stage1 = AdamW(
        param_groups,
        weight_decay=cfg["weight_decay"],
    )

    best_val_acc = 0.0

    for epoch in range(1, cfg["stage1_epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{cfg['stage1_epochs']}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            optimizer_stage1, device, epoch, mixup_cutmix
        )

        # Validate
        val_metrics = validate_model(model, val_loader, device)

        logger.info(
            f"  Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"  Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_vit_model(
                model,
                str(output_path / "vit_checkpoint.pth"),
                epoch,
                optimizer_stage1.state_dict(),
                best_val_acc,
            )

    # Stage 2: Fine-tune entire model
    logger.info("=== Stage 2: Fine-tuning entire model ===")
    model.unfreeze_backbone()

    # Use layer-wise learning rate decay with lower base LR
    param_groups = get_layerwise_lr_params(model, cfg["learning_rate"] * 0.1)
    optimizer_stage2 = AdamW(
        param_groups,
        weight_decay=cfg["weight_decay"],
    )

    scheduler = CosineAnnealingLR(
        optimizer_stage2,
        T_max=cfg["stage2_epochs"],
        eta_min=1e-6,
    )

    for epoch in range(1, cfg["stage2_epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{cfg['stage2_epochs']}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            optimizer_stage2, device, epoch, mixup_cutmix
        )

        # Validate
        val_metrics = validate_model(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"  Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"  Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )
        logger.info(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_vit_model(
                model,
                str(output_path / "vit_checkpoint.pth"),
                cfg["stage1_epochs"] + epoch,
                optimizer_stage2.state_dict(),
                best_val_acc,
            )

    # Compute final metrics
    logger.info("=== Computing final metrics ===")
    final_metrics = compute_metrics(model, val_loader, device, LABEL_NAMES)

    # Save metrics
    metrics_path = output_path / "metrics.json"
    with Path(metrics_path).open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Metrics saved to: {metrics_path}")

    return final_metrics


def main() -> None:
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ViT model")
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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--stage1-epochs", type=int, default=10, help="Stage 1 epochs")
    parser.add_argument("--stage2-epochs", type=int, default=90, help="Stage 2 epochs")

    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
    }

    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
