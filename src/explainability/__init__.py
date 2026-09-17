"""
src/explainability
==================
Gradient-based visual explanation tools.

Public API
----------
    from src.explainability import GradCAM, GradCAMResult

Quick start
-----------
    cam    = GradCAM(model, device=device)
    result = cam(image_tensor)          # (1,C,H,W) or (C,H,W)
    result.overlay    # (H,W,3) uint8 RGB — heatmap blended on image
    result.heatmap    # (H,W)   float32 in [0,1]
    result.save("output/cam_overlay.png")
"""

from src.explainability.gradcam import GradCAM, GradCAMResult

__all__ = ["GradCAM", "GradCAMResult"]
