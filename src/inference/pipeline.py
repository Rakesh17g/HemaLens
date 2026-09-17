"""
InferencePipeline
==================
Single-entry-point orchestrator for the full ALL detection workflow:

  PIL Image
    → preprocess()          shared (224×224 + ImageNet normalise)
    → ConfidenceEstimator   forward pass + entropy + MC Dropout
    → GradCAM               backward pass on features[layer]
    → PipelineResult        single dataclass with ALL outputs

Design principles
-----------------
* **One model load** — caller passes the already-loaded nn.Module.
  In Streamlit this comes from @st.cache_resource; in scripts it is
  built with build_model() + load_checkpoint().
* **No duplicate preprocessing** — the normalised tensor is computed
  once and reused for both Confidence and Grad-CAM.
* **Framework-agnostic output** — PipelineResult holds only
  numpy arrays and Python primitives; no torch tensors leak outside,
  so it is safely picklable for @st.cache_data.
* **Composable** — each sub-module (ConfidenceEstimator, GradCAM,
  MedicalReportGenerator) can still be used independently; the
  pipeline simply wires them together.

Usage
-----
    pipeline = InferencePipeline(
        model     = model,
        device    = device,
        threshold = 0.50,
        mc_passes = 20,
        gradcam_method = "gradcam",
        gradcam_layer  = 8,
    )
    result = pipeline.run(pil_image)   # accepts PIL.Image

    # All fields available on result:
    result.probability        # float
    result.prediction         # "ALL+" | "Healthy"
    result.confidence         # float
    result.risk_level         # "LOW" | "MEDIUM" | "HIGH"
    result.clinical_note      # str
    result.boundary_score     # float
    result.entropy_score      # float
    result.mcdrop_score       # float
    result.original_rgb       # (H,W,3) uint8 numpy
    result.heatmap            # (H,W)   float32 [0,1]
    result.colormap           # (H,W,3) uint8 numpy  (MAGMA)
    result.overlay            # (H,W,3) uint8 numpy  (50/50 blend)
    result.to_dict()          # flat JSON-serialisable dict
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvF
from PIL import Image

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference.confidence import ConfidenceEstimator, ConfidenceResult
from src.explainability.gradcam import GradCAM, GradCAMResult

logger = logging.getLogger("ALLPipeline")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# PipelineResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Unified output of InferencePipeline.run().

    All numpy arrays; no torch tensors.
    Safe for pickle / @st.cache_data / JSON (via to_dict).
    """
    # ── Input ─────────────────────────────────────────────────────────────────
    original_rgb:   np.ndarray     # (H,W,3) uint8 — denormalised source image
    filename:       str            # Original file name (or "unknown")

    # ── Prediction ────────────────────────────────────────────────────────────
    probability:    float          # sigmoid output ∈ [0,1]
    prediction:     str            # "ALL+" | "Healthy"
    confidence:     float          # combined confidence ∈ [0,1]
    risk_level:     str            # "LOW" | "MEDIUM" | "HIGH"
    clinical_note:  str
    boundary_score: float
    entropy_score:  float
    mcdrop_score:   float
    mcdrop_passes:  int
    threshold:      float
    weights:        Dict[str, float]

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    heatmap:        np.ndarray     # (H,W) float32 [0,1]
    colormap:       np.ndarray     # (H,W,3) uint8 MAGMA
    overlay:        np.ndarray     # (H,W,3) uint8 50/50 blend
    gradcam_method: str
    gradcam_layer:  int

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    image_size:     int            # e.g. 224

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a flat dict of all scalar fields (numpy arrays excluded).
        Suitable for JSON serialisation.
        """
        return {
            "filename":       self.filename,
            "probability":    round(self.probability,  4),
            "prediction":     self.prediction,
            "confidence":     round(self.confidence,   4),
            "risk_level":     self.risk_level,
            "clinical_note":  self.clinical_note,
            "boundary_score": round(self.boundary_score, 4),
            "entropy_score":  round(self.entropy_score,  4),
            "mcdrop_score":   round(self.mcdrop_score,   4),
            "mcdrop_passes":  self.mcdrop_passes,
            "threshold":      round(self.threshold, 4),
            "weights":        self.weights,
            "gradcam_method": self.gradcam_method,
            "gradcam_layer":  self.gradcam_layer,
            "image_size":     self.image_size,
        }

    @property
    def is_positive(self) -> bool:
        return "ALL" in self.prediction.upper()

    @property
    def risk_emoji(self) -> str:
        return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(self.risk_level, "⚪")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helper (module-level so it can be called standalone)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    pil_image:   Image.Image,
    target_size: int = 224,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Resize + ImageNet-normalise a PIL image.

    Returns
    -------
    (tensor, original_rgb)
      tensor       : (1,3,H,W) float32  ImageNet-normalised
      original_rgb : (H,W,3)   uint8    denormalised, for display
    """
    img_res  = pil_image.convert("RGB").resize((target_size, target_size),
                                               Image.BILINEAR)
    orig_rgb = np.array(img_res, dtype=np.uint8)
    tensor   = tvF.to_tensor(img_res)
    tensor   = tvF.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)
    return tensor.unsqueeze(0), orig_rgb          # (1,3,H,W), (H,W,3)


# ─────────────────────────────────────────────────────────────────────────────
# InferencePipeline
# ─────────────────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    Orchestrates the complete inference workflow for one image.

    Parameters
    ----------
    model          : Trained EfficientNetB0 (or any compatible nn.Module).
                     Should already be on the target device.
    device         : torch.device for inference.
    threshold      : Decision boundary for binary classification.
    mc_passes      : Number of MC Dropout passes (0 = disabled).
    gradcam_method : "gradcam" or "gradcam++".
    gradcam_layer  : EfficientNet features block index to hook (0–8).
    use_amp        : Use AMP for mixed-precision inference (GPU only).
    image_size     : Resize target (default 224).
    """

    def __init__(
        self,
        model:          nn.Module,
        device:         torch.device,
        threshold:      float = 0.50,
        mc_passes:      int   = 0,
        gradcam_method: str   = "gradcam",
        gradcam_layer:  int   = 8,
        use_amp:        bool  = False,
        image_size:     int   = 224,
    ) -> None:
        self.model          = model
        self.device         = device
        self.threshold      = threshold
        self.mc_passes      = mc_passes
        self.gradcam_method = gradcam_method
        self.gradcam_layer  = gradcam_layer
        self.use_amp        = use_amp
        self.image_size     = image_size

        # Instantiate sub-modules once; they're stateless between calls
        self._estimator = ConfidenceEstimator(
            threshold         = threshold,
            mc_dropout_passes = mc_passes,
            use_amp           = use_amp,
        )
        self._cam = GradCAM(
            model        = model,
            target_layer = gradcam_layer,
            method       = gradcam_method,
            device       = device,
        )
        logger.info(
            f"InferencePipeline ready | "
            f"threshold={threshold:.2f} | mc_passes={mc_passes} | "
            f"gradcam={gradcam_method}[{gradcam_layer}] | "
            f"device={device}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        image:    "Image.Image | np.ndarray",
        filename: str = "unknown",
    ) -> PipelineResult:
        """
        Run the full pipeline on a single image.

        Parameters
        ----------
        image    : PIL.Image or (H,W,3) uint8 numpy array.
        filename : Original filename — recorded in the result metadata.

        Returns
        -------
        PipelineResult with all fields populated.
        """
        # ── Step 1: Preprocess ────────────────────────────────────────────────
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        tensor, original_rgb = preprocess(image, self.image_size)

        logger.debug(f"  [{filename}] preprocessed → {tensor.shape}")

        # ── Step 2: Confidence estimation ─────────────────────────────────────
        conf: ConfidenceResult = self._estimator.estimate(
            self.model, tensor, device=self.device
        )
        logger.debug(
            f"  confidence={conf.confidence:.4f} | "
            f"risk={conf.risk_level} | pred={conf.prediction}"
        )

        # ── Step 3: Grad-CAM ──────────────────────────────────────────────────
        cam: GradCAMResult = self._cam(tensor)
        logger.debug(
            f"  gradcam max_activation={cam.heatmap.max():.4f} | "
            f"method={cam.method}"
        )

        # ── Step 4: Assemble result ────────────────────────────────────────────
        return PipelineResult(
            # Input
            original_rgb   = original_rgb,
            filename       = filename,
            # Prediction
            probability    = conf.probability,
            prediction     = conf.prediction,
            confidence     = conf.confidence,
            risk_level     = conf.risk_level,
            clinical_note  = conf.clinical_note,
            boundary_score = conf.boundary_score,
            entropy_score  = conf.entropy_score,
            mcdrop_score   = conf.mcdrop_score,
            mcdrop_passes  = conf.mcdrop_passes,
            threshold      = conf.threshold,
            weights        = conf.weights,
            # Grad-CAM
            heatmap        = cam.heatmap,
            colormap       = cam.colormap,
            overlay        = cam.overlay,
            gradcam_method = cam.method,
            gradcam_layer  = self.gradcam_layer,
            # Meta
            image_size     = self.image_size,
        )

    def run_from_bytes(
        self,
        image_bytes: bytes,
        filename:    str = "unknown",
    ) -> PipelineResult:
        """
        Convenience wrapper — accepts raw image bytes (e.g. from
        st.file_uploader's .read()).
        """
        import io
        pil = Image.open(io.BytesIO(image_bytes))
        return self.run(pil, filename=filename)
