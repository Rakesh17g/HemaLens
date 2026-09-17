"""
Training Entry Point
=====================
Usage:
    python train.py                           # use configs/train_config.yaml
    python train.py --config configs/train_config.yaml
    python train.py --epochs 30 --batch-size 32
    python train.py --resume models/checkpoints/last_epoch.pth
    python train.py --eval-only              # only run test evaluation

This script:
  1. Loads configuration (YAML + CLI overrides)
  2. Sets random seeds for reproducibility
  3. Builds model, loss, datasets, and all training components
  4. Calls trainer.fit()
  5. Calls trainer.evaluate_test()
  6. Saves final summary
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models.efficientnet import build_model
from src.data.dataset import build_dataloaders
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.loss import build_loss
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer
from src.evaluation import EvaluationEngine


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs/training") -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "train.log"), mode="w"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    WHY ALL FOUR:
      - Python random: used by some data augmentation / sampling
      - numpy: used by sklearn, albumentations, scipy
      - torch: used by model weight init and all operations
      - CUDA: non-determinism from GPU parallel operations
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (slower):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False   # disable auto-tuning


# ─────────────────────────────────────────────────────────────────────────────
# Config Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/train_config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply command-line argument overrides on top of the YAML config."""
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["optimizer"]["head_lr"] = args.lr
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout
    if args.no_aug:
        cfg["dataset"]["use_augmented"] = False
    if args.output_dir:
        cfg["output"]["results_dir"] = args.output_dir
        cfg["logging"]["tensorboard_dir"] = os.path.join(args.output_dir, "tensorboard")
        cfg["checkpoint"]["dir"] = os.path.join(args.output_dir, "checkpoints")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Component Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_all_components(cfg: Dict[str, Any], device: torch.device):
    """Instantiate every training component from the config."""
    logger = logging.getLogger("ALLTrainer")

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("\n[1/6] Building EfficientNet-B0 model...")
    model = build_model(
        pretrained  = cfg["model"]["pretrained"],
        dropout     = cfg["model"]["dropout"],
        num_classes = cfg["model"]["num_classes"],
        freeze_bn   = True,
        device      = device,
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    logger.info("[2/6] Building loss function...")
    criterion = build_loss(
        name            = cfg["loss"]["name"],
        focal_alpha     = cfg["loss"]["focal_alpha"],
        focal_gamma     = cfg["loss"]["focal_gamma"],
        pos_weight      = cfg["loss"]["pos_weight"],
        label_smoothing = cfg["training"]["label_smoothing"],
    ).to(device)

    # ── Augmentation pipelines ─────────────────────────────────────────────
    logger.info("[3/6] Building data augmentation pipelines...")
    target_size    = cfg["dataset"]["target_size"]
    train_tfm      = get_train_transforms(target_size)
    val_tfm        = get_val_transforms(target_size)

    # ── Datasets & DataLoaders ─────────────────────────────────────────────
    logger.info("[4/6] Building datasets and DataLoaders...")
    loaders = build_dataloaders(
        processed_root        = cfg["dataset"]["processed_root"],
        augmented_root        = cfg["dataset"]["augmented_root"],
        train_transform       = train_tfm,
        val_transform         = val_tfm,
        batch_size            = cfg["training"]["batch_size"],
        num_workers           = cfg["dataset"]["num_workers"],
        pin_memory            = cfg["dataset"]["pin_memory"] and device.type == "cuda",
        use_augmented         = cfg["dataset"]["use_augmented"],
        use_weighted_sampler  = True,
    )

    # ── Early Stopping ─────────────────────────────────────────────────────
    logger.info("[5/6] Configuring early stopping + checkpoint manager...")
    es_cfg = cfg["early_stopping"]
    early_stopping = EarlyStopping(
        monitor      = es_cfg["monitor"].replace("val_", ""),
        patience     = es_cfg["patience"],
        min_delta    = es_cfg["min_delta"],
        mode         = "max",
        restore_best = es_cfg["restore_best"],
        verbose      = True,
    )

    # ── Checkpoint Manager ─────────────────────────────────────────────────
    ckpt_cfg = cfg["checkpoint"]
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir  = ckpt_cfg["dir"],
        monitor         = es_cfg["monitor"].replace("val_", ""),
        mode            = "max",
        save_best_only  = ckpt_cfg["save_best_only"],
        save_last       = ckpt_cfg["save_last"],
        filename        = ckpt_cfg["filename"],
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    logger.info("[6/6] Constructing Trainer...")
    trainer = Trainer(
        model          = model,
        criterion      = criterion,
        loaders        = loaders,
        cfg            = cfg,
        device         = device,
        early_stopping = early_stopping,
        checkpoint_mgr = checkpoint_mgr,
    )

    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Optional: Resume from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def resume_from_checkpoint(trainer: Trainer, checkpoint_path: str) -> int:
    """Load a training checkpoint and restore trainer state. Returns start epoch."""
    logger = logging.getLogger("ALLTrainer")
    logger.info(f"\n  Resuming from: {checkpoint_path}")
    state = trainer.ckpt.load(path=checkpoint_path)

    trainer.model.load_state_dict(state["model_state_dict"])
    trainer.current_phase = state.get("current_phase", 1)
    trainer.history       = state.get("history", {})

    # Rebuild optimizer for the correct phase
    trainer.optimizer = trainer._build_optimizer(trainer.current_phase)
    trainer.optimizer.load_state_dict(state["optimizer_state_dict"])

    if state.get("scheduler_state_dict"):
        trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
    if state.get("early_stopping_state"):
        trainer.es.load_state_dict(state["early_stopping_state"])

    start_epoch = state.get("epoch", 0)
    logger.info(f"  Resumed from epoch {start_epoch}")
    return start_epoch


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ALL Detection — EfficientNet-B0 Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/train_config.yaml",
        help="Path to YAML training config"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint to resume from"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training and only run test evaluation"
    )
    # Override config values from CLI
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--dropout",     type=float, default=None)
    parser.add_argument("--no-aug",      action="store_true",
                        help="Disable offline augmented data")
    parser.add_argument("--output-dir",  default=None,
                        help="Override all output directories")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--cpu",         action="store_true",
                        help="Force CPU (even if CUDA available)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Logging & Seed ────────────────────────────────────────────────────
    setup_logging()
    set_seed(args.seed)
    logger = logging.getLogger("ALLTrainer")

    # ── Device ────────────────────────────────────────────────────────────
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    logger.info(f"\n  Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    logger.info(f"  Config: {args.config}")

    # ── Build all components ──────────────────────────────────────────────
    trainer = build_all_components(cfg, device)

    # ── Optional resume ───────────────────────────────────────────────────
    if args.resume:
        resume_from_checkpoint(trainer, args.resume)

    # ── Train or Eval-only ────────────────────────────────────────────────
    if not args.eval_only:
        history = trainer.fit()
        logger.info("\n  Training complete.")
    else:
        logger.info("  Eval-only mode — loading best checkpoint...")
        try:
            state = trainer.ckpt.load()
            trainer.model.load_state_dict(state["model_state_dict"])
        except FileNotFoundError:
            logger.warning("  No checkpoint found. Evaluating untrained model.")

    # ── Final Test Evaluation (via Trainer's built-in loop) ───────────
    test_metrics = trainer.evaluate_test()

    # ── Save test results JSON ─────────────────────────────────
    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "test_results.json")
    with open(results_path, "w") as f:
        clean = {k: float(v) for k, v in test_metrics.items() if not k.startswith("_")}
        json.dump(clean, f, indent=2)
    logger.info(f"\n  Test results saved: {results_path}")

    # ── Full Evaluation with Visualizations ───────────────────────
    eval_cfg = cfg.get("evaluation", {})
    if eval_cfg.get("run_after_training", True):
        logger.info("\n  Running full evaluation with visualizations...")
        engine = EvaluationEngine(
            model       = trainer.model,
            device      = device,
            cfg         = cfg,
            threshold   = eval_cfg.get("threshold", None),
            use_amp     = eval_cfg.get("use_amp", True),
            class_names = cfg.get("classes", {}).get(
                "names", ["Healthy (0)", "ALL+ (1)"]
            ),
        )
        eval_splits = eval_cfg.get("splits", ["val", "test"])
        eval_root   = eval_cfg.get("output_dir", "logs/evaluation")
        for split in eval_splits:
            if split not in trainer.loaders:
                logger.warning(f"  Split '{split}' not found in loaders, skipping.")
                continue
            results = engine.run(trainer.loaders[split], split=split)
            engine.save(results, output_dir=os.path.join(eval_root, split))
        logger.info(
            f"\n  Evaluation complete. Plots saved to: "
            f"{os.path.abspath(eval_root)}/"
        )
    else:
        logger.info("  Skipping post-training evaluation (run_after_training=false).")


if __name__ == "__main__":
    main()
