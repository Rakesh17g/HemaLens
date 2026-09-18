"""
Grad-CAM  &  Grad-CAM++
========================
Gradient-weighted Class Activation Mapping for EfficientNet-B0.

Grad-CAM (Selvaraju et al., 2017) visualises WHICH spatial regions in the
input image most influenced the model's prediction, by:

  1. Forward-passing the image through the model.
  2. Back-propagating the class score through the network to the target
     convolutional layer, capturing gradients.
  3. Averaging the gradient tensor over spatial dims → per-channel weights.
  4. Weighting the activations by those channel weights → CAM.
  5. ReLU + normalise → heatmap ∈ [0, 1].

Target layer: features[8]  (the last MBConv block of EfficientNet-B0,
              output shape ≈ (B, 320, 7, 7) for 224×224 input).
              This is the most semantically rich feature map.

Grad-CAM++ (Chattopadhyay et al., 2018) uses second-order gradients for
better localisation of multiple instances or small objects. Enabled via
``method="gradcam++"``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger("ALLGradCAM")


# ─────────────────────────────────────────────────────────────────────────────
# Hook container
# ─────────────────────────────────────────────────────────────────────────────


class _Hooks:
    """Stores one forward activation tensor and one backward gradient tensor."""

    def __init__(self) -> None:
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None

    def forward_hook(self, module, input, output):
        self.activation = output.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].detach()


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM  (main class)
# ─────────────────────────────────────────────────────────────────────────────


class GradCAM:
    """
    Grad-CAM / Grad-CAM++ visualiser for EfficientNet-B0.

    Args
    ----
    model        : Trained EfficientNetB0 (or any nn.Module with a
                   ``features`` Sequential attribute).
    target_layer : Which features block to hook.  Default=8 (last block).
    method       : "gradcam" or "gradcam++" (Chattopadhyay 2018).
    device       : Inference device.

    Usage
    -----
    >>> cam    = GradCAM(model, device=device)
    >>> result = cam(image_tensor)   # image_tensor: (1, 3, H, W) or (3, H, W)
    >>> result.heatmap     # (H, W) float in [0, 1]
    >>> result.overlay     # (H, W, 3) uint8 BGR overlay on original image
    >>> result.colormap    # (H, W, 3) uint8 pure-color heatmap
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: int = 8,  # features[8] — last MBConv block
        method: str = "gradcam",
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.method = method.lower()
        self._hooks = _Hooks()
        self._fwd_handle = None
        self._bwd_handle = None

        # Resolve target layer
        try:
            self._target = model.features[target_layer]  # type: ignore
        except (AttributeError, IndexError):
            raise ValueError(
                f"model.features[{target_layer}] not found.  "
                "Pass a model with a .features Sequential attribute."
            )
        logger.info(f"GradCAM ready | layer=features[{target_layer}] | method={method}")

    # ── Registration / cleanup ────────────────────────────────────────────────

    def _register(self) -> None:
        self._fwd_handle = self._target.register_forward_hook(self._hooks.forward_hook)  # type: ignore
        self._bwd_handle = self._target.register_full_backward_hook(  # type: ignore
            self._hooks.backward_hook
        )

    def _remove(self) -> None:
        if self._fwd_handle:
            self._fwd_handle.remove()
        if self._bwd_handle:
            self._bwd_handle.remove()

    # ── Core computation ──────────────────────────────────────────────────────

    def __call__(
        self,
        image: torch.Tensor,  # (1,C,H,W) or (C,H,W)
        target_class: int = 1,  # 1 = ALL+, 0 = Healthy
    ) -> GradCAMResult:
        """
        Compute Grad-CAM heatmap for *image*.

        Args
        ----
        image        : Pre-processed image tensor (ImageNet-normalised).
        target_class : Class index to back-propagate.

        Returns
        -------
        GradCAMResult
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        self.model.eval()
        self._register()

        # ── Temporarily enable requires_grad on target-layer params ──────────
        # The backbone is frozen (requires_grad=False) by EfficientNetB0.__init__
        # so that training only updates the head (Phase 1). But GradCAM's
        # register_full_backward_hook on model.features[target_layer] only fires
        # when at least one param in that layer participates in the gradient
        # graph. Fix: snapshot → enable → backward → restore.
        _saved_grad = {id(p): p.requires_grad for p in self._target.parameters()}  # type: ignore
        for p in self._target.parameters():  # type: ignore
            p.requires_grad_(True)

        try:
            # ── Forward (needs full gradient graph — NO torch.no_grad) ────────
            image = image.detach()  # don't compute grad w.r.t. input
            logit = self.model(image)  # (1, 1)
            prob = float(torch.sigmoid(logit).item())

            # ── Backward on the target class score ───────────────────────────
            self.model.zero_grad()
            score = logit[0, 0]  # scalar
            score.backward()

            act = self._hooks.activation  # (1, C, h, w)
            grad = self._hooks.gradient  # (1, C, h, w)

            if act is None or grad is None:
                raise RuntimeError("Hooks did not capture tensors.")

            # ── CAM weights ──────────────────────────────────────────────────
            if self.method == "gradcam":
                # Global average pooling of gradients → per-channel weight
                weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            elif self.method == "gradcam++":
                # α = grad² / (2·grad² + act·grad³)
                grad2 = grad**2
                grad3 = grad**3
                denom = 2 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True)
                denom = torch.where(denom != 0, denom, torch.ones_like(denom))
                alpha = grad2 / denom
                weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)

            else:
                raise ValueError(f"Unknown method: '{self.method}'")

            # ── Weighted activation sum ───────────────────────────────────────
            cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, h, w)
            cam = F.relu(cam)  # clip negatives

            # ── Resize to input resolution ───────────────────────────────────
            H, W = image.shape[2], image.shape[3]
            cam_up = F.interpolate(
                cam, size=(H, W), mode="bilinear", align_corners=False
            )  # (1, 1, H, W)
            cam_np = cam_up.squeeze().cpu().numpy()  # (H, W)

            # ── Normalise to [0, 1] ───────────────────────────────────────────
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_np = np.zeros_like(cam_np)

        finally:
            self._remove()
            # Restore original requires_grad state (keep backbone frozen)
            for p in self._target.parameters():  # type: ignore
                p.requires_grad_(_saved_grad.get(id(p), False))
            self.model.zero_grad()

        # ── Colour heatmap (MAGMA colormap — perceptually uniform) ────────────
        cam_uint8 = (cam_np * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_MAGMA)  # BGR

        # ── Overlay on denormalised original image ────────────────────────────
        orig_np = _denorm_to_uint8(image.squeeze(0).cpu())  # (H,W,3) RGB
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig_bgr, 0.50, colormap, 0.50, 0)  # BGR

        return GradCAMResult(
            heatmap=cam_np,
            colormap=cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB),
            overlay=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            original=orig_np,
            probability=prob,
            method=self.method,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


class GradCAMResult:
    """
    Container for all Grad-CAM outputs.

    Attributes
    ----------
    heatmap     : (H, W)    float32 array in [0, 1].  Raw activation map.
    colormap    : (H, W, 3) uint8 RGB.  Pure heatmap coloured with MAGMA.
    overlay     : (H, W, 3) uint8 RGB.  50/50 blend of original + heatmap.
    original    : (H, W, 3) uint8 RGB.  Denormalised input image.
    probability : float.  Model's sigmoid output.
    method      : str.    "gradcam" or "gradcam++".
    """

    def __init__(
        self,
        heatmap: np.ndarray,
        colormap: np.ndarray,
        overlay: np.ndarray,
        original: np.ndarray,
        probability: float,
        method: str,
    ) -> None:
        self.heatmap = heatmap
        self.colormap = colormap
        self.overlay = overlay
        self.original = original
        self.probability = probability
        self.method = method

    def save(self, path: str) -> str:
        """Save the overlay image to disk. Returns the path."""
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rgb = self.overlay
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        return path

    def save_all(self, prefix: str) -> dict:
        """
        Save original, heatmap, colormap, and overlay with prefix.

        Returns
        -------
        Dict mapping name → path.
        """
        import os

        os.makedirs(
            os.path.dirname(os.path.abspath(prefix + "_x.png")) or ".", exist_ok=True
        )
        paths = {}
        for name, arr in [
            ("original", self.original),
            ("heatmap", _float_to_rgb(self.heatmap)),
            ("colormap", self.colormap),
            ("overlay", self.overlay),
        ]:
            p = f"{prefix}_{name}.png"
            bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, bgr)
            paths[name] = p
        return paths


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and convert (C,H,W) tensor → (H,W,3) uint8.
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN  # denormalise
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _float_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert (H,W) float [0,1] → (H,W,3) uint8 greyscale RGB."""
    grey = (arr * 255).astype(np.uint8)
    return np.stack([grey, grey, grey], axis=-1)
