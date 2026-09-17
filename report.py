"""
report.py — Medical Report CLI
================================
Generate a professional PDF diagnostic report for a single blood-smear image.

Pipeline
--------
  1. Load model checkpoint
  2. Pre-process image (resize → ImageNet normalise)
  3. Run forward pass → prediction probability
  4. Run Grad-CAM → attention heatmap overlay
  5. Run ConfidenceEstimator → confidence + risk level
  6. Generate PDF with MedicalReportGenerator

Usage
-----
  # Minimal: image path + checkpoint
  python report.py --image path/to/cell.png --checkpoint models/checkpoints/best.pth

  # With metadata
  python report.py \\
      --image       data/processed/test/all/img_001.png \\
      --checkpoint  models/checkpoints/efficientnet_b0_best.pth \\
      --patient-id  P-00123 \\
      --sample-id   SLD-2026-0042 \\
      --analyst     "Dr. Smith" \\
      --output-dir  logs/reports \\
      --mc-passes   20 \\
      --threshold   0.50 \\
      --gradcam-method  gradcam++

Output
------
  logs/reports/<REPORT_ID>_<DATE>.pdf
  logs/reports/gradcam_overlay.png   (optional, with --save-cam)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from PIL import Image

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.efficientnet import build_model
from src.inference            import ConfidenceEstimator
from src.explainability       import GradCAM
from src.reports              import MedicalReportGenerator, MedicalReportData


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ALLReport")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ALL Detection — Medical PDF Report Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--image",      required=True, help="Path to input cell image")
    p.add_argument("--checkpoint", required=True, help="Path to model .pth checkpoint")

    # Output
    p.add_argument("--output-dir", default="logs/reports",
                   help="Directory to save the PDF report")
    p.add_argument("--save-cam", action="store_true",
                   help="Also save Grad-CAM overlay PNG alongside the PDF")

    # Model
    p.add_argument("--model-version", default="EfficientNet-B0 v1.0",
                   help="Human-readable model version tag in the report")
    p.add_argument("--threshold", type=float, default=0.50,
                   help="Decision threshold (0=more sensitive, 1=more specific)")
    p.add_argument("--gradcam-method", default="gradcam",
                   choices=["gradcam", "gradcam++"],
                   help="GradCAM variant to use for heatmap")
    p.add_argument("--gradcam-layer", type=int, default=8,
                   help="EfficientNet features block index to hook (0–8)")

    # Confidence
    p.add_argument("--mc-passes", type=int, default=0,
                   help="MC Dropout passes (0=disabled, 20=recommended)")

    # Report metadata
    p.add_argument("--patient-id",   default="ANON",
                   help="Anonymised patient identifier")
    p.add_argument("--sample-id",    default="N/A",
                   help="Slide / sample reference ID")
    p.add_argument("--institution",  default="ALL Detection AI System",
                   help="Reporting institution name")
    p.add_argument("--analyst",      default="AI Diagnostic Assistant",
                   help="Reviewing analyst or physician name")

    # Device
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Image loading & pre-processing
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(path: str, size: int = 224) -> tuple:
    """
    Load, resize, normalise image.

    Returns
    -------
    (tensor, original_rgb)
      tensor       : (1,3,H,W) float ImageNet-normalised
      original_rgb : (H,W,3) uint8 RGB for display
    """
    img     = Image.open(path).convert("RGB")
    img_res = img.resize((size, size), Image.BILINEAR)
    orig    = np.array(img_res, dtype=np.uint8)

    tensor = tvF.to_tensor(img_res)
    tensor = tvF.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)
    return tensor.unsqueeze(0), orig   # (1,3,H,W), (H,W,3)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(model, path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "  → Train the model first:  python train.py"
        )
    state = torch.load(path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"  Checkpoint epoch: {state.get('epoch', '?')}")
    else:
        model.load_state_dict(state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    logger.info("\n" + "=" * 60)
    logger.info("  ALL DETECTION — MEDICAL REPORT GENERATOR")
    logger.info("=" * 60)
    logger.info(f"  Image      : {args.image}")
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  Output dir : {args.output_dir}")

    # ── Device ────────────────────────────────────────────────────────────────
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    logger.info(f"  Device     : {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("\n[1/5] Loading model …")
    model = build_model(pretrained=False, device=device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # ── Image ─────────────────────────────────────────────────────────────────
    logger.info("\n[2/5] Loading and pre-processing image …")
    image_tensor, original_rgb = load_image(args.image)
    logger.info(f"  Image shape: {original_rgb.shape}")

    # ── Confidence estimation ─────────────────────────────────────────────────
    logger.info("\n[3/5] Estimating prediction + confidence …")
    estimator = ConfidenceEstimator(
        threshold         = args.threshold,
        mc_dropout_passes = args.mc_passes,
    )
    conf_result = estimator.estimate(model, image_tensor, device=device)
    print(conf_result)   # formatted summary to stdout

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    logger.info("\n[4/5] Generating Grad-CAM heatmap …")
    cam_engine = GradCAM(
        model        = model,
        target_layer = args.gradcam_layer,
        method       = args.gradcam_method,
        device       = device,
    )
    cam_result = cam_engine(image_tensor)

    if args.save_cam:
        cam_out = os.path.join(args.output_dir, "gradcam_overlay.png")
        os.makedirs(args.output_dir, exist_ok=True)
        cam_result.save(cam_out)
        logger.info(f"  Grad-CAM overlay saved: {cam_out}")

    # ── Report ────────────────────────────────────────────────────────────────
    logger.info("\n[5/5] Generating PDF report …")
    data = MedicalReportData(
        original_image   = cam_result.original,     # denormalised RGB
        heatmap_overlay  = cam_result.overlay,       # Grad-CAM overlay
        prediction       = conf_result.prediction,
        probability      = conf_result.probability,
        confidence       = conf_result.confidence,
        risk_level       = conf_result.risk_level,
        clinical_note    = conf_result.clinical_note,
        boundary_score   = conf_result.boundary_score,
        entropy_score    = conf_result.entropy_score,
        mcdrop_score     = conf_result.mcdrop_score,
        threshold        = conf_result.threshold,
        model_version    = args.model_version,
        patient_id       = args.patient_id,
        sample_id        = args.sample_id,
        institution      = args.institution,
        analyst          = args.analyst,
    )
    gen  = MedicalReportGenerator(
        output_dir  = args.output_dir,
        institution = args.institution,
    )
    pdf_path = gen.generate(data)

    logger.info("\n" + "=" * 60)
    logger.info("  REPORT COMPLETE")
    logger.info(f"  PDF: {pdf_path}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
