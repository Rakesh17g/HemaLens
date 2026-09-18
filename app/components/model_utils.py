"""
Shared model utilities for the Streamlit app.
=============================================
Single source of truth for model loading, image preprocessing,
and the complete end-to-end inference pipeline.

Public functions
----------------
load_model()          @st.cache_resource — load once, share across pages
preprocess_image()    PIL → (tensor, original_rgb)
run_full_pipeline()   @st.cache_data — complete workflow, returns flat dict
run_prediction()      @st.cache_data — confidence only (legacy, unchanged)
run_gradcam()         @st.cache_data — gradcam only (legacy, unchanged)
init_session()        seed session_state defaults
has_image()           True if image bytes in session state
checkpoint_exists()   True if checkpoint file exists on disk

Why run_full_pipeline() instead of two separate calls?
-------------------------------------------------------
The old approach called load_model() + preprocess_image() twice — once
inside run_prediction() and once inside run_gradcam(). Even though
load_model() is @st.cache_resource, the preprocessing happened twice.

run_full_pipeline() delegates to InferencePipeline which preprocesses
once and reuses the tensor for both ConfidenceEstimator and GradCAM.
The individual run_prediction() and run_gradcam() functions are kept
as-is for the existing per-feature pages.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchvision.transforms.functional as tvF
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("ALLApp")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Model loader  (@st.cache_resource — lives for the entire session)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device_name: str = "cpu"):
    """
    Load EfficientNet-B0 from a checkpoint.
    Cached across ALL pages for the lifetime of the server process.

    Returns
    -------
    (model, device)  or  (None, device) if checkpoint not found.
    """
    from src.models.efficientnet import build_model

    device = torch.device(device_name)
    
    # Resolve relative paths against ROOT to support running from any directory
    if not os.path.isabs(checkpoint_path):
        resolved_path = ROOT / checkpoint_path
        if resolved_path.exists():
            checkpoint_path = str(resolved_path)

    if not os.path.exists(checkpoint_path):
        return None, device

    model = build_model(pretrained=False, device=device)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path} on {device}")
    return model, device


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing (shared helper — no caching, cheap operation)
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_image(
    pil_image: Image.Image,
    target_size: int = 224,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Resize + ImageNet-normalise a PIL image.

    Returns
    -------
    (tensor, original_rgb)
      tensor       : (1,3,H,W) float32 normalised
      original_rgb : (H,W,3)   uint8   for display
    """
    img_res = pil_image.convert("RGB").resize(
        (target_size, target_size), Image.BILINEAR  # type: ignore
    )
    orig = np.array(img_res, dtype=np.uint8)
    tensor = tvF.to_tensor(img_res)
    tensor = tvF.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)
    return tensor.unsqueeze(0), orig


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline  (@st.cache_data — cached per (bytes, settings) tuple)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def run_full_pipeline(
    image_bytes: bytes,
    checkpoint: str,
    threshold: float = 0.50,
    mc_passes: int = 0,
    gradcam_method: str = "gradcam",
    gradcam_layer: int = 8,
    filename: str = "unknown",
) -> dict:
    """
    Execute the complete Upload→Preprocess→Predict→GradCAM workflow.

    Uses InferencePipeline internally — single preprocessing,
    shared model, no duplicated code paths.

    Returns
    -------
    Flat dict with all fields from PipelineResult:
      probability, prediction, confidence, risk_level, clinical_note,
      boundary_score, entropy_score, mcdrop_score, mcdrop_passes,
      threshold, weights, gradcam_method, gradcam_layer, image_size,
      original_rgb (ndarray), heatmap (ndarray), colormap (ndarray),
      overlay (ndarray), filename.
    On failure: {"error": str}
    """
    from src.inference.pipeline import InferencePipeline

    device_name = get_device()
    model, device = load_model(checkpoint, device_name)
    if model is None:
        return {"error": f"Checkpoint not found: {checkpoint}"}

    pipeline = InferencePipeline(
        model=model,
        device=device,
        threshold=threshold,
        mc_passes=mc_passes,
        gradcam_method=gradcam_method,
        gradcam_layer=gradcam_layer,
    )

    result = pipeline.run_from_bytes(image_bytes, filename=filename)

    return {
        # scalars
        "filename": result.filename,
        "probability": result.probability,
        "prediction": result.prediction,
        "confidence": result.confidence,
        "risk_level": result.risk_level,
        "clinical_note": result.clinical_note,
        "boundary_score": result.boundary_score,
        "entropy_score": result.entropy_score,
        "mcdrop_score": result.mcdrop_score,
        "mcdrop_passes": result.mcdrop_passes,
        "threshold": result.threshold,
        "weights": result.weights,
        "gradcam_method": result.gradcam_method,
        "gradcam_layer": result.gradcam_layer,
        "image_size": result.image_size,
        # arrays
        "original_rgb": result.original_rgb,
        "heatmap": result.heatmap,
        "colormap": result.colormap,
        "overlay": result.overlay,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF bytes  (@st.cache_data)
# ─────────────────────────────────────────────────────────────────────────────


# @st.cache_data(show_spinner=False)  # DO NOT USE CACHE FOR PDF GENERATION; causes zombie failures
def generate_pdf_bytes(
    pipeline_result_dict: dict,
    patient_id: str = "ANON",
    sample_id: str = "N/A",
    institution: str = "ALL Detection AI System",
    analyst: str = "AI Diagnostic Assistant",
    model_version: str = "EfficientNet-B0 v1.0",
) -> bytes:
    """
    Generate PDF bytes from a pipeline result dict.

    Wraps MedicalReportGenerator.generate_bytes() so the PDF is
    cached per (result + metadata) — no PDF is regenerated on rerun.

    Returns
    -------
    bytes : Raw PDF content for st.download_button.
    On failure: b"" (empty bytes)
    """
    from src.reports.report_generator import MedicalReportData, MedicalReportGenerator

    try:
        # Reconstruct a minimal PipelineResult-like namespace from the dict
        # to use from_pipeline_result() without importing the dataclass
        class _Proxy:
            pass

        pr = _Proxy()
        for k, v in pipeline_result_dict.items():
            setattr(pr, k, v)

        data = MedicalReportData.from_pipeline_result(
            pr,
            patient_id=patient_id,
            sample_id=sample_id,
            institution=institution,
            analyst=analyst,
            model_version=model_version,
        )
        gen = MedicalReportGenerator(output_dir="/tmp")  # dir unused for bytes
        return gen.generate_bytes(data)

    except Exception as exc:  # noqa: BLE001
        import traceback

        st.error(f"Generate PDF Bytes Failed: {exc!r}\n{traceback.format_exc()}")
        logger.error(f"PDF generation failed: {exc}")
        return b""


# ─────────────────────────────────────────────────────────────────────────────
# Legacy per-feature functions (kept for existing pages 3 & 4)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def run_prediction(
    image_bytes: bytes,
    checkpoint: str,
    threshold: float = 0.50,
    mc_passes: int = 0,
) -> dict:
    """Confidence estimation only — used by page 3 (Prediction)."""
    from src.inference import ConfidenceEstimator

    device_name = get_device()
    model, device = load_model(checkpoint, device_name)
    if model is None:
        return {"error": f"Checkpoint not found: {checkpoint}"}

    pil = Image.open(io.BytesIO(image_bytes))
    tensor, orig_rgb = preprocess_image(pil)

    estimator = ConfidenceEstimator(threshold=threshold, mc_dropout_passes=mc_passes)
    r = estimator.estimate(model, tensor, device=device)

    return {
        "probability": r.probability,
        "prediction": r.prediction,
        "confidence": r.confidence,
        "risk_level": r.risk_level,
        "boundary_score": r.boundary_score,
        "entropy_score": r.entropy_score,
        "mcdrop_score": r.mcdrop_score,
        "mcdrop_passes": r.mcdrop_passes,
        "clinical_note": r.clinical_note,
        "threshold": r.threshold,
        "original_rgb": orig_rgb,
    }


@st.cache_data(show_spinner=False)
def run_gradcam(
    image_bytes: bytes,
    checkpoint: str,
    layer: int = 8,
    method: str = "gradcam",
) -> dict:
    """GradCAM only — used by page 4 (GradCAM Visualization)."""
    from src.explainability import GradCAM

    device_name = get_device()
    model, device = load_model(checkpoint, device_name)
    if model is None:
        return {"error": f"Checkpoint not found: {checkpoint}"}

    pil = Image.open(io.BytesIO(image_bytes))
    tensor, _ = preprocess_image(pil)

    cam = GradCAM(model, target_layer=layer, method=method, device=device)
    result = cam(tensor)

    return {
        "original": result.original,
        "heatmap": result.heatmap,
        "colormap": result.colormap,
        "overlay": result.overlay,
        "probability": result.probability,
        "method": result.method,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────


def init_session() -> None:
    """Seed all session_state keys with safe defaults (idempotent)."""
    defaults = {
        "uploaded_bytes": None,
        "uploaded_filename": None,
        "checkpoint_path": "models/checkpoints/efficientnet_b0_best.pth",
        "threshold": 0.50,
        "mc_passes": 0,
        "gradcam_layer": 8,
        "gradcam_method": "gradcam",
        "prediction_result": None,
        "gradcam_result": None,
        "pipeline_result": None,  # ← new: full pipeline result
        "eval_metrics": None,
        # report metadata
        "patient_id": "ANON",
        "sample_id": "N/A",
        "analyst": "AI Diagnostic Assistant",
        "institution": "ALL Detection AI System",
        "model_version": "EfficientNet-B0 v1.0",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def has_image() -> bool:
    return st.session_state.get("uploaded_bytes") is not None


def has_prediction() -> bool:
    return st.session_state.get("prediction_result") is not None


def has_pipeline_result() -> bool:
    return st.session_state.get("pipeline_result") is not None


def checkpoint_exists() -> bool:
    path = st.session_state.get("checkpoint_path", "")
    if not path:
        return False
    if not os.path.isabs(path):
        resolved = ROOT / path
        if resolved.exists():
            return True
    return os.path.exists(path)
