"""
src/reports
===========
Professional PDF diagnostic report generation.

Public API
----------
    from src.reports import MedicalReportGenerator, MedicalReportData, build_report

Quick start (manual)
--------------------
    from src.reports import MedicalReportGenerator, MedicalReportData

    data = MedicalReportData(
        original_image   = original_np,       # (H,W,3) uint8 RGB
        heatmap_overlay  = overlay_np,         # (H,W,3) uint8 RGB  (Grad-CAM)
        prediction       = "ALL+",
        probability      = 0.94,
        confidence       = 0.87,
        risk_level       = "LOW",
        clinical_note    = "High-confidence positive. Immediate review recommended.",
        boundary_score   = 0.88,
        entropy_score    = 0.85,
        mcdrop_score     = 0.0,
        threshold        = 0.50,
        model_version    = "EfficientNet-B0 v1.0",
        patient_id       = "P-00123",
        sample_id        = "SLD-2026-0042",
    )
    gen  = MedicalReportGenerator(output_dir="logs/reports")
    path = gen.generate(data)
    print(f"Saved: {path}")

One-call convenience
--------------------
    from src.reports import build_report

    path = build_report(
        model            = model,
        image_tensor     = image_tensor,
        confidence_result = conf_result,   # ConfidenceResult from ConfidenceEstimator
        output_dir       = "logs/reports",
    )
"""

from src.reports.report_generator import (
    MedicalReportGenerator,
    MedicalReportData,
    build_report,
)

__all__ = [
    "MedicalReportGenerator",
    "MedicalReportData",
    "build_report",
]
