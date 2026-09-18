"""
Loss Functions
================
Focal Loss for robust binary classification.

WHY FOCAL LOSS (not plain BCELoss):
  Even though our dataset is nearly balanced (48.6% / 51.4%), Focal Loss
  provides an important benefit: it down-weights easy, confidently-classified
  samples and focuses learning on hard, uncertain examples.

  In blood smear images:
  - Easy negatives: clearly healthy cells → model assigns 0.01 probability
  - Hard positives: blast cells with mild morphological changes → 0.45 probability
  - These ambiguous cases are where error happens and where we need to focus

  Focal Loss formula:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
      p_t = p     if y=1 (positive label)
            1-p   if y=0 (negative label)
      gamma = focusing parameter (2.0 standard)
              when an example is easy (p_t → 1), (1-p_t)^gamma → 0, loss → 0
              when an example is hard (p_t → 0), (1-p_t)^gamma → 1, full loss
      alpha = class balancing weight (0.25 = downwt negatives slightly)

WHY LABEL SMOOTHING:
  Prevents the model from becoming overconfident (assigning 0.99 or 0.01)
  on training examples. Overconfidence → high loss on val/test → poor calibration.
  Label smoothing replaces hard labels {0,1} with soft {ε/2, 1-ε/2}.
"""


import torch
import torch.nn.functional as F
from torch import nn

# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for binary classification.

    Args:
        alpha:           Class balance factor. 0.25 = slightly down-weight negatives.
        gamma:           Focusing exponent. 2.0 is standard from the original paper.
        reduction:       'mean' | 'sum' | 'none'
        label_smoothing: If > 0, apply label smoothing to targets.
                         Recommended: 0.05–0.15 for small medical datasets.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,  # (B, 1) raw model outputs (before sigmoid)
        targets: torch.Tensor,  # (B,) or (B, 1) float labels {0.0, 1.0}
    ) -> torch.Tensor:
        targets = targets.view(-1, 1).float()
        logits = logits.view(-1, 1).float()

        # Label smoothing
        if self.label_smoothing > 0:
            targets = (
                targets * (1 - self.label_smoothing)
                + (1 - targets) * self.label_smoothing
            )

        # Binary cross-entropy per element (numerically stable via log-sum-exp)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Probability of the TRUE class
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_w = (1.0 - p_t) ** self.gamma

        # Alpha weight: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal Loss
        loss = alpha_t * focal_w * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Weighted BCE Loss  (simpler alternative)
# ─────────────────────────────────────────────────────────────────────────────


class WeightedBCELoss(nn.Module):
    """
    BCEWithLogitsLoss with fixed positive class weight.

    Use when class imbalance is the only concern (no hard-easy mining needed).
    pos_weight = num_negatives / num_positives (e.g. 1.64 for IDB combined).
    """

    def __init__(self, pos_weight: float = 1.64) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits.view(-1), targets.view(-1).float(), pos_weight=pw
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_loss(
    name: str = "focal",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: float = 1.64,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """Return the configured loss function."""
    if name == "focal":
        return FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "bce_weighted":
        return WeightedBCELoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss: '{name}'. Options: focal, bce, bce_weighted")
