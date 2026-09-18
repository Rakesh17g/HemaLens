"""
Early Stopping Callback
========================
Monitors a validation metric and stops training when it stops improving.

Design notes:
  - Decoupled from the Trainer — passed in as a dependency
  - Supports both "higher is better" (AUC, F1) and "lower is better" (loss)
  - Stores the epoch number and metric value at the best checkpoint
  - Optional: restore_best_weights automatically reloads best model state
"""

import copy
import logging

from torch import nn

logger = logging.getLogger("ALLTrainer")


class EarlyStopping:
    """
    Early stopping with best-weights restoration.

    Args:
        monitor:        Metric name to watch (e.g. 'val_auc', 'val_loss').
        patience:       Epochs with no improvement before stopping.
        min_delta:      Minimum absolute change to count as improvement.
        mode:           'max' for metrics like AUC/F1; 'min' for loss.
        restore_best:   Reload weights from best epoch when stopped.
        verbose:        Log patience countdown every epoch.
    """

    def __init__(
        self,
        monitor: str = "val_auc",
        patience: int = 12,
        min_delta: float = 1e-4,
        mode: str = "max",
        restore_best: bool = True,
        verbose: bool = True,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.verbose = verbose

        # State
        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.best_epoch: int = 0
        self.best_weights: dict | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def step(self, value: float, epoch: int, model: nn.Module) -> bool:
        """
        Check metric and update state.

        Args:
            value:  Current epoch's monitored metric value.
            epoch:  Current epoch number (1-indexed).
            model:  The model (for best-weights snapshot).

        Returns:
            True if training should stop, False to continue.
        """
        if self._is_improvement(value):
            if self.verbose:
                logger.info(
                    f"  [EarlyStopping] {self.monitor} improved: "
                    f"{self.best_value:.5f} → {value:.5f}  "
                    f"(epoch {self.best_epoch} → {epoch})"
                )
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                # Deep-copy state dict — detach from GPU to save memory
                self.best_weights = copy.deepcopy(
                    {k: v.cpu() for k, v in model.state_dict().items()}
                )
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs  "
                    f"(best {self.monitor}={self.best_value:.5f} @ epoch {self.best_epoch})"
                )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"  [EarlyStopping] TRIGGERED — restoring best weights "
                    f"from epoch {self.best_epoch}"
                )
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)

        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_value = state["best_value"]
        self.best_epoch = state["best_epoch"]
        self.counter = state["counter"]
        self.should_stop = state["should_stop"]
