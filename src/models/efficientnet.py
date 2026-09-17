"""
EfficientNet-B0 Model Definition
==================================
Three-phase transfer learning strategy:

  Phase 1 — Head Only (epochs 1–5):
    Backbone completely frozen. Only the custom classification head
    is trained. This lets the randomly-initialized head quickly adapt
    to the task without destroying pretrained feature representations.

  Phase 2 — Partial Unfreeze (epochs 6–20):
    Last 3 MBConv blocks (features[5], features[6], features[7]) unfrozen.
    These are the highest-level feature extractors — they learn task-specific
    patterns (blast cell morphology, nuclear shape) while early blocks retain
    low-level edge/texture detectors.

  Phase 3 — Full Fine-tune (epochs 21–50):
    Entire backbone unfrozen with very small LR (1e-5).
    Allows low-level features (stain color, edge orientation) to also
    adapt to blood-smear microscopy domain.

Custom Head design:
    AdaptiveAvgPool → Dropout(0.4) → Linear(1280→512) → SiLU →
    Dropout(0.3) → Linear(512→1)

    - AdaptiveAvgPool: keeps spatial position invariance
    - Two-layer head: non-linear feature combination before sigmoid
    - SiLU (Swish): same activation as EfficientNet backbone → smooth gradient flow
    - Binary output (1 neuron): used with BCEWithLogitsLoss / Focal Loss
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 with custom binary classification head.

    Args:
        pretrained:  Load ImageNet weights via torchvision.
        dropout:     Dropout rate before first FC layer.
        num_classes: Output neurons. 1 = binary (sigmoid), >1 = softmax.
        freeze_bn:   Whether to keep BatchNorm layers frozen even during
                     backbone fine-tuning. Recommended True for small datasets —
                     BN statistics from ImageNet are better than re-estimating
                     on 1300 images.
    """

    # EfficientNet-B0 backbone in-features from AdaptiveAvgPool
    BACKBONE_OUT_FEATURES = 1280

    def __init__(
        self,
        pretrained:  bool = True,
        dropout:     float = 0.40,
        num_classes: int   = 1,
        freeze_bn:   bool  = True,
    ) -> None:
        super().__init__()

        # ── Load pretrained backbone ──────────────────────────────────
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.efficientnet_b0(weights=weights)

        # ── Strip original classifier, keep feature extractor only ───
        # backbone.features: Sequential of 9 blocks (features[0]–features[8])
        # backbone.avgpool:  AdaptiveAvgPool2d(1, 1)
        self.features  = backbone.features
        self.avgpool   = backbone.avgpool

        # ── Custom classification head ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(self.BACKBONE_OUT_FEATURES, 512),
            nn.SiLU(inplace=True),              # Swish — matches backbone activations
            nn.Dropout(p=0.30, inplace=False),
            nn.Linear(512, num_classes),
        )

        # ── Initialize custom head with Kaiming Normal ────────────────
        self._init_head()

        # ── Freeze entire backbone at construction (Phase 1) ─────────
        self._freeze_backbone()

        # ── Optionally keep BN frozen throughout training ─────────────
        self.freeze_bn = freeze_bn

    # ── Initialization ────────────────────────────────────────────────────

    def _init_head(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True
        # Re-freeze BN if configured
        if self.freeze_bn:
            self._freeze_batchnorm()

    def _freeze_batchnorm(self) -> None:
        """Keep BN in eval mode so running statistics are never updated."""
        for module in self.features.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    # ── Phase Management (called by Trainer) ─────────────────────────────

    def set_phase(self, phase: int) -> List[dict]:
        """
        Switch to a training phase and return param groups for optimizer.

        Phase 1: head only
        Phase 2: head + last 3 feature blocks
        Phase 3: full backbone + head

        Returns:
            List of param-group dicts for optimizer construction.
        """
        if phase == 1:
            self._freeze_backbone()
            return [
                {"params": self.classifier.parameters(), "lr": 1e-3, "name": "head"},
            ]

        elif phase == 2:
            self._freeze_backbone()
            # Unfreeze features[5], [6], [7], [8]
            unfreeze_blocks = [5, 6, 7, 8]
            for i in unfreeze_blocks:
                for p in self.features[i].parameters():
                    p.requires_grad = True
            if self.freeze_bn:
                self._freeze_batchnorm()
            return [
                {"params": self.classifier.parameters(), "lr": 1e-3, "name": "head"},
                {"params": [p for i in unfreeze_blocks
                            for p in self.features[i].parameters()
                            if p.requires_grad],
                 "lr": 1e-4, "name": "backbone_top"},
            ]

        elif phase == 3:
            self._unfreeze_backbone()
            return [
                {"params": self.classifier.parameters(), "lr": 1e-4, "name": "head"},
                {"params": self.features.parameters(),   "lr": 1e-5, "name": "backbone"},
            ]

        else:
            raise ValueError(f"Unknown phase: {phase}. Must be 1, 2, or 3.")

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float tensor, normalized with ImageNet stats.
        Returns:
            (B, 1) raw logits — apply sigmoid externally for probabilities.
        """
        feat = self.features(x)          # (B, 1280, H', W')
        feat = self.avgpool(feat)         # (B, 1280, 1,  1)
        feat = torch.flatten(feat, 1)    # (B, 1280)
        out  = self.classifier(feat)     # (B, 1)
        return out

    # ── Introspection ─────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def train(self, mode: bool = True) -> "EfficientNetB0":
        """Override train() to keep BN frozen if configured."""
        super().train(mode)
        if mode and self.freeze_bn:
            self._freeze_batchnorm()
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    pretrained:  bool  = True,
    dropout:     float = 0.40,
    num_classes: int   = 1,
    freeze_bn:   bool  = True,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> EfficientNetB0:
    """
    Build and optionally restore the EfficientNet-B0 model.

    Args:
        pretrained:       Use ImageNet initialization.
        dropout:          Head dropout rate.
        num_classes:      1 for binary.
        freeze_bn:        Keep BN eval'd during fine-tuning.
        checkpoint_path:  If provided, load saved weights (for inference/resume).
        device:           Target device.
    Returns:
        EfficientNetB0 model on the specified device.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0(
        pretrained  = pretrained,
        dropout     = dropout,
        num_classes = num_classes,
        freeze_bn   = freeze_bn,
    )

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        # Support both bare state_dict and wrapped checkpoint formats
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")

    model = model.to(device)
    return model


import os  # noqa: E402 (imported at bottom to avoid circular at top)
