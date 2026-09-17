"""
src/inference
=============
Inference-time utilities: confidence estimation, risk classification,
and the unified end-to-end pipeline.

Public API
----------
    from src.inference import ConfidenceEstimator, RiskLevel, ConfidenceResult
    from src.inference import InferencePipeline, PipelineResult, preprocess
    from src.inference.confidence_viz import generate_confidence_plots

Quick start — single confidence estimate
-----------------------------------------
    estimator = ConfidenceEstimator(threshold=0.50, mc_dropout_passes=20)
    result    = estimator.estimate(model, image_tensor)
    print(result)

Quick start — full pipeline
----------------------------
    pipeline = InferencePipeline(
        model     = model,
        device    = device,
        threshold = 0.50,
        mc_passes = 20,
        gradcam_method = "gradcam",
    )
    result = pipeline.run(pil_image)
    print(result.prediction, result.confidence, result.risk_level)
    # numpy arrays ready for display:
    result.original_rgb    # (H,W,3) uint8
    result.overlay         # (H,W,3) Grad-CAM overlay
    # generate PDF bytes in one line:
    from src.reports import MedicalReportData, MedicalReportGenerator
    data      = MedicalReportData.from_pipeline_result(result)
    pdf_bytes = MedicalReportGenerator().generate_bytes(data)
"""

from src.inference.confidence import (
    ConfidenceEstimator,
    ConfidenceResult,
    RiskLevel,
)
from src.inference.pipeline import (
    InferencePipeline,
    PipelineResult,
    preprocess,
)

__all__ = [
    # Confidence
    "ConfidenceEstimator",
    "ConfidenceResult",
    "RiskLevel",
    # Pipeline
    "InferencePipeline",
    "PipelineResult",
    "preprocess",
]
