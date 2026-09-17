"""
Evaluation Entry Point
=======================
Standalone script to evaluate a trained ALL detection model checkpoint.

Usage examples
--------------
    # Evaluate best checkpoint on test set (default)
    python evaluate.py

    # Use a custom checkpoint
    python evaluate.py --checkpoint models/checkpoints/last_epoch.pth

    # Evaluate on val split with a fixed threshold
    python evaluate.py --split val --threshold 0.40

    # Skip plots (metrics + text report only)
    python evaluate.py --no-plots

    # Force CPU even if CUDA is available
    python evaluate.py --cpu

    # Override output directory
    python evaluate.py --output-dir logs/my_eval_run

What this script does
---------------------
  1. Load eval_config.yaml (+ CLI overrides)
  2. Build dataset / DataLoader for the requested split
  3. Build the EfficientNet-B0 model and load checkpoint weights
  4. Run EvaluationEngine.run()  → inference + all metrics
  5. Run EvaluationEngine.save() → JSON, text report, 8 Matplotlib plots
  6. Print the full summary to stdout and log file.

All 8 plots are saved to --output-dir/<split>/
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.efficientnet import build_model
from src.data.dataset import build_dataloaders
from src.data.augmentation import get_val_transforms
from src.evaluation import EvaluationEngine


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs/evaluation") -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(log_dir, "evaluate.log"), mode="w"
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ALL Detection — Model Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default="configs/eval_config.yaml",
        help="Path to YAML evaluation config",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to model checkpoint (.pth). Overrides config.",
    )
    p.add_argument(
        "--split", default=None, choices=["train", "val", "test"],
        help="Which dataset split to evaluate. Overrides config.",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold [0,1]. Overrides config. None = auto Youden.",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Root directory for all output artifacts.",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating Matplotlib visualizations.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Inference batch size.",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference even if CUDA is available.",
    )
    p.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed-precision inference.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading + CLI overrides
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Overwrite config values with any CLI flags provided."""
    eval_cfg = cfg.setdefault("evaluation", {})

    if args.checkpoint:
        cfg.setdefault("checkpoint", {})["path"] = args.checkpoint
    if args.split:
        eval_cfg["splits"] = [args.split]
    if args.threshold is not None:
        eval_cfg["threshold"] = args.threshold
    if args.output_dir:
        eval_cfg["output_dir"] = args.output_dir
    if args.batch_size:
        eval_cfg["batch_size"] = args.batch_size
    if args.no_amp:
        eval_cfg["use_amp"] = False
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    model:  torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Load model weights from a checkpoint file."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "  → Train the model first with:  python train.py"
        )
    logger.info(f"  Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # Support both raw state_dict and wrapped checkpoints
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        epoch = state.get("epoch", "?")
        val   = state.get("metrics", {}).get("auc_roc", "?")
        logger.info(f"  Checkpoint: epoch={epoch}  val_auc={val}")
    else:
        model.load_state_dict(state)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder (val/test transforms only — no augmentation at eval time)
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_loaders(
    cfg:    Dict[str, Any],
    splits: list,
) -> Dict[str, Any]:
    """
    Build DataLoaders for the requested splits.

    Uses val-time transforms (no flip/colour-jitter augmentation).
    """
    eval_cfg  = cfg.get("evaluation", {})
    ds_cfg    = cfg.get("dataset", {})
    tfm       = get_val_transforms(ds_cfg.get("target_size", 224))
    batch_sz  = eval_cfg.get("batch_size", 32)
    workers   = ds_cfg.get("num_workers", 4)

    # build_dataloaders returns a dict with 'train', 'val', 'test' keys
    all_loaders = build_dataloaders(
        processed_root   = ds_cfg.get("processed_root", "data/processed"),
        augmented_root   = ds_cfg.get("augmented_root", "data/augmented"),
        train_transform  = tfm,   # no-op; only val/test loaders are used
        val_transform    = tfm,
        batch_size       = batch_sz,
        num_workers      = workers,
        pin_memory       = ds_cfg.get("pin_memory", True),
        use_augmented    = False,               # never augment at eval time
        use_weighted_sampler = False,
    )
    return {s: all_loaders[s] for s in splits if s in all_loaders}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    eval_cfg = cfg.get("evaluation", {})
    out_root = eval_cfg.get("output_dir", "logs/evaluation")

    # ── Logging ───────────────────────────────────────────────────────────
    setup_logging(out_root)
    logger = logging.getLogger("ALLEvaluation")

    logger.info("\n" + "=" * 60)
    logger.info("  ALL DETECTION — EVALUATION")
    logger.info("=" * 60)
    logger.info(f"  Config: {args.config}")

    # ── Device ────────────────────────────────────────────────────────────
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    logger.info(f"  Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = cfg.get("model", {})
    logger.info("\n[1/4] Building model …")
    model = build_model(
        pretrained  = False,   # weights come from checkpoint
        dropout     = model_cfg.get("dropout", 0.40),
        num_classes = model_cfg.get("num_classes", 1),
        device      = device,
    )

    ckpt_path = cfg.get("checkpoint", {}).get(
        "path", "models/checkpoints/efficientnet_b0_best.pth"
    )
    model = load_checkpoint(model, ckpt_path, device, logger)
    model.eval()

    # ── DataLoaders ───────────────────────────────────────────────────────
    splits = eval_cfg.get("splits", ["test"])
    logger.info(f"\n[2/4] Building DataLoaders for splits: {splits} …")
    loaders = build_eval_loaders(cfg, splits)

    # ── EvaluationEngine ──────────────────────────────────────────────────
    logger.info("\n[3/4] Initialising EvaluationEngine …")
    engine = EvaluationEngine(
        model       = model,
        device      = device,
        cfg         = cfg,
        threshold   = eval_cfg.get("threshold", None),
        use_amp     = eval_cfg.get("use_amp", True) and not args.no_amp,
        class_names = cfg.get("classes", {}).get(
            "names", ["Healthy (0)", "ALL+ (1)"]
        ),
    )

    # ── Evaluate each split ───────────────────────────────────────────────
    logger.info("\n[4/4] Running evaluation …")
    all_results: Dict[str, Any] = {}

    for split, loader in loaders.items():
        results = engine.run(loader, split=split)

        split_out = os.path.join(out_root, split)
        if args.no_plots:
            # Save metrics + report only
            engine.save_metrics(results, split_out)
            engine.save_report(results, split_out)
            engine.save_arrays(results, split_out)
        else:
            engine.save(results, output_dir=split_out)

        all_results[split] = results

    # ── Final summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 60)
    for split, res in all_results.items():
        logger.info(
            f"  [{split.upper():>5}]  "
            f"AUC={res['auc_roc']:.4f}  "
            f"F1={res['f1']:.4f}  "
            f"Sens={res['sensitivity']:.4f}  "
            f"Spec={res['specificity']:.4f}  "
            f"MCC={res['mcc']:.4f}"
        )
    logger.info(f"\n  All artifacts saved to: {os.path.abspath(out_root)}/")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
