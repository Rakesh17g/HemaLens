"""
Main Training Engine
======================
The Trainer class orchestrates:
  1. Phase management (3-phase gradual unfreezing)
  2. Training loop with mixed-precision (torch.amp)
  3. Validation loop
  4. Progress bars (tqdm)
  5. TensorBoard logging
  6. Early stopping
  7. Checkpoint saving
  8. LR scheduling (cosine warm-up)
  9. Gradient clipping
  10. Final test-set evaluation

Architecture: Trainer is framework-agnostic — it receives all components
as constructor arguments (dependency injection). This makes every component
independently unit-testable.
"""

import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.checkpoint import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.metrics import MetricAccumulator

logger = logging.getLogger("ALLTrainer")


# ─────────────────────────────────────────────────────────────────────────────
# Cosine Warm-Up Scheduler
# ─────────────────────────────────────────────────────────────────────────────


class CosineWarmupScheduler:
    """
    Linear warm-up for `warmup_epochs`, then CosineAnnealingWarmRestarts.

    WHY WARM-UP:
        In Phase 1, we start with a freshly initialized head. Without warm-up,
        a large LR causes the head to produce noisy gradients that propagate
        into the backbone (even partially frozen). Warm-up ramps LR from
        lr*0.1 → lr over the first `warmup_epochs`, stabilizing early training.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 3,
        T_0: int = 10,
        T_mult: int = 1,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr
        )
        self._epoch = 0

    def step(self) -> None:
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            # Linear warm-up: scale = epoch / warmup_epochs
            scale = self._epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = max(self.min_lr, base_lr * scale)
        else:
            self._cosine.step()

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "cosine": self._cosine.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._cosine.load_state_dict(state["cosine"])

    def get_lrs(self) -> list:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class Trainer:
    """
    Full training engine for the ALL detection model.

    Args:
        model:           The EfficientNetB0 model.
        criterion:       Loss function (FocalLoss or BCEWithLogitsLoss).
        loaders:         Dict with 'train', 'val', 'test' DataLoaders.
        cfg:             Full config dict loaded from train_config.yaml.
        device:          torch.device.
        early_stopping:  EarlyStopping instance.
        checkpoint_mgr:  CheckpointManager instance.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        loaders: dict[str, DataLoader],
        cfg: dict[str, Any],
        device: torch.device,
        early_stopping: EarlyStopping,
        checkpoint_mgr: CheckpointManager,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.loaders = loaders
        self.cfg = cfg
        self.device = device
        self.es = early_stopping
        self.ckpt = checkpoint_mgr

        # ── Config shortcuts ──────────────────────────────────────────────
        self.train_cfg = cfg["training"]
        self.opt_cfg = cfg["optimizer"]
        self.sched_cfg = cfg["scheduler"]
        self.log_cfg = cfg["logging"]
        self.unfreeze = cfg["model"]["unfreeze_schedule"]
        self.ckpt_cfg = cfg["checkpoint"]

        # ── Training state ────────────────────────────────────────────────
        self.current_phase = 1
        self.current_epoch = 0
        self.best_threshold = 0.50
        self.history: dict[str, list] = defaultdict(list)

        # ── Mixed precision ───────────────────────────────────────────────
        self.use_amp = (
            self.train_cfg.get("mixed_precision", False)
            and self.device.type == "cuda"  # AMP is a no-op on CPU anyway
        )
        self.scaler = GradScaler(enabled=self.use_amp)

        # ── TensorBoard ───────────────────────────────────────────────────
        tb_dir = self.log_cfg.get("tensorboard_dir", "logs/tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)

        # ── Initialize Phase 1 ────────────────────────────────────────────
        self.optimizer = self._build_optimizer(phase=1)
        self.scheduler = self._build_scheduler()

        logger.info("Trainer initialised.")
        self._log_param_counts()

    # ── Optimizer & Scheduler Construction ───────────────────────────────

    def _build_optimizer(self, phase: int) -> optim.Optimizer:
        """Build AdamW with per-phase param groups."""
        param_groups = self.model.set_phase(phase)  # type: ignore
        # Apply weight decay and betas from config to every group
        for pg in param_groups:
            pg.setdefault("weight_decay", self.opt_cfg["weight_decay"])
        return optim.AdamW(
            param_groups,
            betas=tuple(self.opt_cfg["betas"]),
            eps=self.opt_cfg["eps"],
        )

    def _build_scheduler(self) -> CosineWarmupScheduler:
        return CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.sched_cfg["warmup_epochs"],
            T_0=self.sched_cfg["T_0"],
            T_mult=self.sched_cfg["T_mult"],
            min_lr=self.sched_cfg["min_lr"],
        )

    def _transition_phase(self, new_phase: int) -> None:
        """Switch to a new training phase — rebuilds optimizer + scheduler."""
        if new_phase == self.current_phase:
            return
        logger.info(f"\n{'=' * 55}")
        logger.info(f"  PHASE TRANSITION:  {self.current_phase} → {new_phase}")
        logger.info(f"{'=' * 55}")
        self.current_phase = new_phase
        self.optimizer = self._build_optimizer(phase=new_phase)
        self.scheduler = self._build_scheduler()
        self._log_param_counts()

    # ── Training Loop ─────────────────────────────────────────────────────

    def _train_epoch(self) -> dict[str, float]:
        """Run one full training epoch. Returns epoch metrics."""
        self.model.train()
        accumulator = MetricAccumulator()
        loader = self.loaders["train"]
        clip_norm = self.train_cfg.get("gradient_clip", 1.0)

        pbar = tqdm(
            loader,
            desc=f"  Epoch {self.current_epoch:>3} [Train]",
            ncols=90,
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward (with optional AMP) ─────────────────────────
            with autocast(enabled=self.use_amp):
                logits = self.model(images)  # (B, 1)
                loss = self.criterion(logits, labels)

            # ── Backward ────────────────────────────────────────────
            self.scaler.scale(loss).backward()

            # ── Gradient clipping (before optimizer step) ───────────
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                max_norm=clip_norm,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── Accumulate ──────────────────────────────────────────
            accumulator.update(logits.detach(), labels.detach(), float(loss.item()))

            # ── Progress bar update ─────────────────────────────────
            running_loss = accumulator.compute_running_loss()
            pbar.set_postfix({"loss": f"{running_loss:.4f}"}, refresh=False)

            # ── TensorBoard batch-level log ─────────────────────────
            global_step = (self.current_epoch - 1) * len(loader) + batch_idx
            if batch_idx % self.log_cfg.get("log_interval", 10) == 0:
                self.writer.add_scalar(
                    "Batch/train_loss", float(loss.item()), global_step
                )

                if self.log_cfg.get("log_grad_norm", True):
                    total_norm = (
                        sum(
                            p.grad.data.norm(2).item() ** 2
                            for p in self.model.parameters()
                            if p.requires_grad and p.grad is not None
                        )
                        ** 0.5
                    )
                    self.writer.add_scalar("Batch/grad_norm", total_norm, global_step)

        pbar.close()
        return accumulator.compute(threshold=self.best_threshold)

    # ── Validation / Test Loop ────────────────────────────────────────────

    @torch.no_grad()
    def _eval_epoch(self, split: str = "val") -> dict[str, float]:
        """Run evaluation on val or test split."""
        self.model.eval()
        accumulator = MetricAccumulator()
        loader = self.loaders[split]

        pbar = tqdm(
            loader,
            desc=f"  Epoch {self.current_epoch:>3} [{split.capitalize():>4}]",
            ncols=90,
            leave=False,
        )

        for batch in pbar:
            # test_loader returns (img, label, path) — ignore path
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            accumulator.update(logits, labels, float(loss.item()))
            pbar.set_postfix({"loss": f"{accumulator.compute_running_loss():.4f}"})

        pbar.close()
        return accumulator.compute()

    # ── TensorBoard Epoch Logging ─────────────────────────────────────────

    def _log_epoch_to_tensorboard(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        epoch: int,
    ) -> None:
        # Scalar groups
        metric_pairs = {
            "Loss": ("loss", "loss"),
            "AUC-ROC": ("auc_roc", "auc_roc"),
            "Accuracy": ("accuracy", "accuracy"),
            "F1": ("f1", "f1"),
            "Sensitivity": ("sensitivity", "sensitivity"),
            "Specificity": ("specificity", "specificity"),
        }
        for tag, (tk, vk) in metric_pairs.items():
            self.writer.add_scalars(
                f"Epoch/{tag}",
                {"Train": train_metrics.get(tk, 0), "Val": val_metrics.get(vk, 0)},
                epoch,
            )

        # Learning rates
        for i, pg in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f"LR/group_{pg.get('name', i)}", pg["lr"], epoch)

        # Optimal threshold
        self.writer.add_scalar(
            "Threshold/youden_optimal",
            val_metrics.get("optimal_threshold", 0.5),
            epoch,
        )

        # Phase
        self.writer.add_scalar("Training/phase", self.current_phase, epoch)

    # ── History Tracking ──────────────────────────────────────────────────

    def _update_history(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        for k, v in train_metrics.items():
            if not k.startswith("_"):
                self.history[f"train_{k}"].append(float(v))
        for k, v in val_metrics.items():
            if not k.startswith("_"):
                self.history[f"val_{k}"].append(float(v))
        self.history["epoch"].append(self.current_epoch)
        self.history["phase"].append(self.current_phase)
        self.history["lr"].append(self.scheduler.get_lrs()[0])

    # ── Console Summary ───────────────────────────────────────────────────

    def _print_epoch_summary(
        self,
        train_m: dict[str, float],
        val_m: dict[str, float],
        elapsed: float,
    ) -> None:
        is_best = self.ckpt.best_epoch == self.current_epoch
        star = " ★ NEW BEST" if is_best else ""
        logger.info(
            f"\n  Epoch {self.current_epoch:>3}/{self.cfg['training']['epochs']}  "
            f"[Phase {self.current_phase}]  [{elapsed:.1f}s]{star}"
        )
        logger.info(
            f"  {'':4} {'Loss':>8} {'AUC':>8} {'Acc':>7} "
            f"{'F1':>7} {'Sens':>7} {'Spec':>7} {'Thr':>6}"
        )
        for prefix, m in [("Train", train_m), ("  Val", val_m)]:
            logger.info(
                f"  {prefix}: "
                f"{m.get('loss', 0):8.4f} "
                f"{m.get('auc_roc', 0):8.4f} "
                f"{m.get('accuracy', 0):7.4f} "
                f"{m.get('f1', 0):7.4f} "
                f"{m.get('sensitivity', 0):7.4f} "
                f"{m.get('specificity', 0):7.4f} "
                f"{m.get('optimal_threshold', 0.5):6.3f}"
            )

    # ── Main fit() Method ─────────────────────────────────────────────────

    def fit(self) -> dict[str, list]:
        """
        Run the full training loop.

        Returns:
            Training history dict with per-epoch metrics.
        """
        epochs = self.train_cfg["epochs"]
        phase1_end = self.unfreeze["phase1_epochs"]
        phase2_end = phase1_end + self.unfreeze["phase2_epochs"]

        logger.info("\n" + "=" * 55)
        logger.info("  ALL DETECTION — TRAINING START")
        logger.info(
            f"  Epochs: {epochs}  |  Device: {self.device}  |  AMP: {self.use_amp}"
        )
        logger.info(f"  Phase 1: epochs 1–{phase1_end}   (head only)")
        logger.info(
            f"  Phase 2: epochs {phase1_end + 1}–{phase2_end}  (unfreeze top blocks)"
        )
        logger.info(f"  Phase 3: epochs {phase2_end + 1}–{epochs} (full fine-tune)")
        logger.info("=" * 55)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            t0 = time.time()

            # ── Phase transitions ────────────────────────────────────
            if epoch == phase1_end + 1:
                self._transition_phase(2)
            elif epoch == phase2_end + 1:
                self._transition_phase(3)

            # ── Train ────────────────────────────────────────────────
            train_metrics = self._train_epoch()

            # ── Validate ─────────────────────────────────────────────
            val_metrics = self._eval_epoch("val")

            # Update best threshold from validation
            self.best_threshold = val_metrics.get("optimal_threshold", 0.50)

            # ── LR Step ──────────────────────────────────────────────
            self.scheduler.step()

            # ── Checkpoint ───────────────────────────────────────────
            self.ckpt.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.es,
                metrics=val_metrics,
                history=self.history,
                current_phase=self.current_phase,
            )

            # ── History & Logging ─────────────────────────────────────
            self._update_history(train_metrics, val_metrics)
            self._log_epoch_to_tensorboard(train_metrics, val_metrics, epoch)
            self._print_epoch_summary(train_metrics, val_metrics, time.time() - t0)

            # ── Early Stopping ────────────────────────────────────────
            monitor_val = val_metrics.get(
                self.cfg["early_stopping"]["monitor"].replace("val_", ""), 0.0
            )
            if self.es.step(monitor_val, epoch, self.model):
                logger.info(f"\n  Early stopping at epoch {epoch}.")
                break

        # Save metrics history
        self.ckpt.save_metrics_json(
            self.history, path=self.cfg["output"]["metrics_file"]
        )
        self.writer.close()

        logger.info("\n" + "=" * 55)
        logger.info(f"  Training complete. Best epoch: {self.es.best_epoch}")
        logger.info(
            f"  Best {self.cfg['early_stopping']['monitor']}: {self.es.best_value:.5f}"
        )
        logger.info("=" * 55)

        return dict(self.history)

    # ── Test Evaluation ───────────────────────────────────────────────────

    def evaluate_test(self) -> dict[str, float]:
        """Run final evaluation on held-out test set."""
        logger.info("\n  Evaluating on test set...")
        self.current_epoch = 0  # suppress epoch display
        test_metrics = self._eval_epoch("test")
        logger.info("\n  ── TEST RESULTS ──────────────────────────────")
        for k in [
            "accuracy",
            "auc_roc",
            "auprc",
            "f1",
            "sensitivity",
            "specificity",
            "mcc",
            "optimal_threshold",
        ]:
            logger.info(f"  {k:>20}: {test_metrics.get(k, 0):.4f}")
        logger.info(
            f"  {'Confusion':>20}: "
            f"TP={test_metrics['tp']} TN={test_metrics['tn']} "
            f"FP={test_metrics['fp']} FN={test_metrics['fn']}"
        )
        return test_metrics

    # ── Introspection ─────────────────────────────────────────────────────

    def _log_param_counts(self) -> None:
        p = self.model.count_parameters()  # type: ignore
        logger.info(
            f"  Parameters — total: {p['total']:,}  "
            f"trainable: {p['trainable']:,}  "
            f"frozen: {p['frozen']:,}"
        )
