"""
Loss functions for sleep staging.

- FocalLoss: handles severe class imbalance better than weighted CE for rare
  stages (N1 in particular). Default gamma=2.
- build_loss: factory that returns either CE or Focal based on config.

Usage:
    criterion = build_loss(
        loss_type=cfg["training"].get("loss", "focal"),
        class_weights=class_weights,
        gamma=cfg["training"].get("focal_gamma", 2.0),
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    L = - alpha_c * (1 - p_c)^gamma * log(p_c)

    Args:
        gamma: focusing parameter (>=0). gamma=0 reduces to CE.
        alpha: optional per-class weight tensor (same shape as class_weights in CE).
        reduction: 'mean' | 'sum' | 'none'
        label_smoothing: applied to one-hot targets before computing the loss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : (B, C) or (B, C, T)
        # targets: (B,)   or (B, T)
        if logits.dim() == 3:
            B, C, T = logits.shape
            logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
            targets_flat = targets.reshape(-1)
        else:
            logits_flat = logits
            targets_flat = targets

        log_probs = F.log_softmax(logits_flat, dim=-1)            # (N, C)
        probs = log_probs.exp()
        N, C = log_probs.shape

        # Build (smoothed) one-hot targets
        with torch.no_grad():
            valid = (targets_flat >= 0)
            tgt = targets_flat.clamp(min=0)
            one_hot = F.one_hot(tgt, num_classes=C).float()
            if self.label_smoothing > 0:
                one_hot = (
                    one_hot * (1.0 - self.label_smoothing)
                    + self.label_smoothing / C
                )
            one_hot = one_hot * valid.unsqueeze(-1).float()

        # alpha_c per-sample
        if self.alpha is not None:
            alpha = self.alpha.to(logits_flat.device).unsqueeze(0)  # (1, C)
            alpha_t = (one_hot * alpha).sum(dim=-1)
        else:
            alpha_t = torch.ones(N, device=logits_flat.device)

        focal_weight = (1.0 - probs).pow(self.gamma)               # (N, C)
        # focal CE: - sum_c [ smoothed_target_c * focal_weight_c * log p_c ]
        loss_per_sample = -(one_hot * focal_weight * log_probs).sum(dim=-1)   # (N,)
        loss_per_sample = alpha_t * loss_per_sample

        if not valid.any():
            return loss_per_sample.sum() * 0.0

        if self.reduction == "mean":
            return loss_per_sample[valid].mean()
        if self.reduction == "sum":
            return loss_per_sample[valid].sum()
        return loss_per_sample


def build_loss(
    loss_type: str = "focal",
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> nn.Module:
    """Factory: returns FocalLoss for 'focal', CrossEntropyLoss for 'ce'."""
    loss_type = (loss_type or "focal").lower()
    if loss_type == "focal":
        logger.info(f"Loss: FocalLoss (gamma={gamma}, label_smoothing={label_smoothing}, "
                    f"weighted={'yes' if class_weights is not None else 'no'})")
        return FocalLoss(
            gamma=gamma,
            alpha=class_weights,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    if loss_type in ("ce", "crossentropy", "cross_entropy"):
        logger.info(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing}, "
                    f"weighted={'yes' if class_weights is not None else 'no'})")
        return nn.CrossEntropyLoss(
            weight=class_weights,
            reduction=reduction,
            label_smoothing=label_smoothing,
            ignore_index=-100,
        )
    raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'focal' or 'ce'.")
