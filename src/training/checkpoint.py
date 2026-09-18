"""
Checkpoint Manager
===================
Saves and loads model checkpoints during training.

Saves full training state (not just model weights) so training can be
resumed exactly from any checkpoint:
  - model state_dict
  - optimizer state_dict
  - scheduler state_dict
  - early stopping state
  - epoch number, best metric value
  - full training history (loss/metrics per epoch)
"""

import json
import logging
import os
from typing import Any

import torch
from torch import nn

logger = logging.getLogger("ALLTrainer")


class CheckpointManager:
    """
    Manages model checkpointing during training.

    Args:
        checkpoint_dir:  Directory to save .pth files.
        monitor:         Metric to compare for "best" checkpoint.
        mode:            'max' or 'min' for the monitored metric.
        save_best_only:  If True, only keep the best checkpoint.
        save_last:       Always save the most recent epoch too.
        filename:        Base filename for the best checkpoint.
    """

    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        monitor: str = "val_auc",
        mode: str = "max",
        save_best_only: bool = True,
        save_last: bool = True,
        filename: str = "efficientnet_b0_best.pth",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.filename = filename

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0

        os.makedirs(checkpoint_dir, exist_ok=True)

    @property
    def best_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.filename)

    @property
    def last_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "last_epoch.pth")

    def _is_best(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        early_stopping: Any,
        metrics: dict[str, float],
        history: dict[str, list],
        current_phase: int,
    ) -> bool:
        """
        Save checkpoint. Returns True if this was a new best.

        Saves a comprehensive state that allows full training resumption.
        """
        value = metrics.get(self.monitor, 0.0)
        is_best = self._is_best(value)

        state = {
            # ── Core training state ─────────────────────────────────
            "epoch": epoch,
            "current_phase": current_phase,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            if hasattr(scheduler, "state_dict")
            else {},
            "early_stopping_state": early_stopping.state_dict(),
            # ── Performance snapshot ────────────────────────────────
            "best_metric_value": self.best_value,
            "best_epoch": self.best_epoch,
            "current_metrics": {
                k: v for k, v in metrics.items() if not k.startswith("_")
            },  # skip raw arrays
            "history": history,
        }

        # Always save last
        if self.save_last:
            torch.save(state, self.last_path)

        # Save best if improved
        if is_best:
            if self.mode == "max":
                self.best_value = value
            else:
                self.best_value = value
            self.best_epoch = epoch

            torch.save(state, self.best_path)
            logger.info(
                f"  [Checkpoint] NEW BEST saved → {self.filename}  "
                f"({self.monitor}={value:.5f})"
            )

        return is_best

    def load(
        self,
        path: str | None = None,
    ) -> dict[str, Any]:
        """
        Load a checkpoint from disk.

        Args:
            path: Checkpoint file path. Defaults to best_path.
        Returns:
            Full state dict.
        """
        path = path or self.best_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        logger.info(f"  [Checkpoint] Loaded: {path}  (epoch={state.get('epoch', '?')})")
        return state

    def save_metrics_json(
        self,
        history: dict[str, list],
        path: str = "logs/training/metrics.json",
    ) -> None:
        """Save full training history as JSON for analysis/visualization."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert numpy types to Python natives for JSON serialization
        clean = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        logger.info(f"  [Checkpoint] Metrics history saved: {path}")
