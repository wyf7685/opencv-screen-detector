"""ViT model for screen detector.

Vision Transformer (ViT-Small) with transfer learning support.
"""

import timm
import torch
import torch.nn as nn

# Default config - 使用 ViT-Small 而非 ViT-Base
MODEL_NAME = "vit_small_patch16_224"
NUM_CLASSES = 3


class ViTScreenDetector(nn.Module):
    """ViT-Small model for screen detection.

    Architecture:
    - ViT-Small backbone (ImageNet pretrained, 22M params)
    - Stochastic Depth (drop_path_rate=0.1)
    - Classification head (3 classes)

    Args:
        model_name: timm model name
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone parameters
        drop_path_rate: Stochastic depth rate
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Create ViT model with stochastic depth
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters except classifier head."""
        for name, param in self.model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, 224, 224)

        Returns:
            Classification logits (B, num_classes)
        """
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head.

        Args:
            x: Input image tensor (B, 3, 224, 224)

        Returns:
            Feature tensor (B, hidden_dim)
        """
        # Get features from penultimate layer
        features = self.model.forward_features(x)
        # Use CLS token (first token)
        return features[:, 0]


def create_vit_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    drop_path_rate: float = 0.1,
) -> ViTScreenDetector:
    """Create a ViT screen detector model.

    Args:
        model_name: timm model name
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone parameters
        drop_path_rate: Stochastic depth rate

    Returns:
        ViTScreenDetector model
    """
    return ViTScreenDetector(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        drop_path_rate=drop_path_rate,
    )


def load_vit_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> ViTScreenDetector:
    """Load ViT model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded ViTScreenDetector model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_vit_model(
        model_name=checkpoint.get("model_name", MODEL_NAME),
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_vit_model(
    model: ViTScreenDetector,
    checkpoint_path: str,
    epoch: int,
    optimizer_state_dict: dict | None = None,
    best_val_acc: float = 0.0,
) -> None:
    """Save ViT model checkpoint.

    Args:
        model: ViTScreenDetector model
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        optimizer_state_dict: Optimizer state dict
        best_val_acc: Best validation accuracy
    """
    checkpoint = {
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }

    if optimizer_state_dict:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict

    torch.save(checkpoint, checkpoint_path)
