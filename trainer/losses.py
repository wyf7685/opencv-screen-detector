"""Custom loss functions for screen detector training.

Includes:
- FocalLoss: For handling hard examples and class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter, higher value focuses more on hard examples.
               Default: 2.0
        alpha: Class weights tensor. If None, no class weighting.
        reduction: Reduction mode ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Model output logits (B, C)
            targets: Ground truth labels (B,)

        Returns:
            Computed focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            # Apply class weights
            alpha_t = self.alpha.to(inputs.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def create_criterion(
    use_focal_loss: bool = True,
    class_weights: list[float] | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Create loss criterion based on configuration.

    Args:
        use_focal_loss: Whether to use Focal Loss
        class_weights: Class weights for imbalanced dataset
        focal_gamma: Focal loss gamma parameter

    Returns:
        Loss criterion module
    """
    alpha = torch.tensor(class_weights) if class_weights else None

    if use_focal_loss:
        return FocalLoss(gamma=focal_gamma, alpha=alpha)
    return nn.CrossEntropyLoss(weight=alpha)
