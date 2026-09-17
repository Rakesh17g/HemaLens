"""
Model Evaluator
================
Collects predictions from a DataLoader, computes the complete
medical-grade metric suite, and returns structured results.

Separated from Trainer so evaluation can be run independently
(e.g. evaluate a saved checkpoint without re-training).
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    balanced_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.metrics import find_optimal_threshold

logger = logging.getLogger("ALLEvaluator")


# ─────────────────────────────────────────────────────────────────────────────
# Core Inference Runner
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    use_amp:   bool = False,
    desc:      str  = "Evaluating",
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """
    Run full inference over a DataLoader.

    Args:
        model:    Trained model in eval mode.
        loader:   DataLoader yielding (image, label) or (image, label, path).
        device:   Computation device.
        use_amp:  Use mixed precision (GPU only).
        desc:     tqdm description string.

    Returns:
        y_true:  (N,) int array of ground-truth labels {0, 1}.
        y_prob:  (N,) float array of predicted probabilities [0, 1].
        paths:   List of file paths if loader returns them, else None.
    """
    model.eval()

    all_probs:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_paths:  List[str]        = []
    has_paths = False

    pbar = tqdm(loader, desc=f"  {desc}", ncols=88, leave=True)

    for batch in pbar:
        # Support (img, label) and (img, label, path) loaders
        if len(batch) == 3:
            images, labels, paths = batch
            all_paths.extend(paths)
            has_paths = True
        else:
            images, labels = batch

        images = images.to(device)

        with autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(images)                        # (B, 1)
            probs  = torch.sigmoid(logits).cpu().numpy()  # (B, 1)

        all_probs.append(probs.reshape(-1))
        all_labels.append(labels.numpy().reshape(-1))

    pbar.close()

    y_prob  = np.concatenate(all_probs).astype(np.float32)
    y_true  = np.concatenate(all_labels).astype(np.int32)

    return y_true, y_prob, (all_paths if has_paths else None)


# ─────────────────────────────────────────────────────────────────────────────
# Complete Metric Suite
# ─────────────────────────────────────────────────────────────────────────────

def compute_full_metrics(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: Optional[float] = None,
    class_names: List[str]     = ["Healthy (0)", "ALL+ (1)"],
) -> Dict[str, Any]:
    """
    Compute the complete evaluation metric suite.

    If threshold is None, Youden's J statistic is used to find
    the clinically optimal operating point automatically.

    Args:
        y_true:      Ground-truth binary labels {0, 1}.
        y_prob:      Predicted probabilities [0, 1].
        threshold:   Decision boundary. None = auto via Youden's J.
        class_names: Names for the two classes [negative, positive].

    Returns:
        Dict containing all metrics, curve arrays, and report strings.
    """
    # ── Optimal threshold ─────────────────────────────────────────────────
    opt_thresh, youden_j = find_optimal_threshold(y_true, y_prob)
    t = threshold if threshold is not None else opt_thresh
    y_pred = (y_prob >= t).astype(int)

    # ── Confusion Matrix components ───────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity  = tp / (tp + fn + 1e-8)   # True Positive Rate (Recall for +)
    specificity  = tn / (tn + fp + 1e-8)   # True Negative Rate
    precision    = tp / (tp + fp + 1e-8)   # Positive Predictive Value
    npv          = tn / (tn + fn + 1e-8)   # Negative Predictive Value
    fpr_thresh   = fp / (fp + tn + 1e-8)   # False Positive Rate at threshold
    fnr          = fn / (fn + tp + 1e-8)   # False Negative Rate (Miss Rate)

    # ── Aggregate scalars ─────────────────────────────────────────────────
    accuracy          = float(accuracy_score(y_true, y_pred))
    balanced_acc      = float(balanced_accuracy_score(y_true, y_pred))
    f1                = float(f1_score(y_true, y_pred, zero_division=0))
    f1_macro          = float(f1_score(y_true, y_pred, average="macro",   zero_division=0))
    f1_weighted       = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc               = float(matthews_corrcoef(y_true, y_pred))
    precision_score_v = float(precision_score(y_true, y_pred, zero_division=0))
    recall_score_v    = float(recall_score(y_true, y_pred, zero_division=0))

    try:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_roc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = 0.0

    # ── Curve Arrays (for plotting) ───────────────────────────────────────
    fpr_arr, tpr_arr, roc_thresholds   = roc_curve(y_true, y_prob, pos_label=1)
    prec_arr, rec_arr, pr_thresholds   = precision_recall_curve(y_true, y_prob)

    # ── Classification Report ─────────────────────────────────────────────
    report_str  = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        # ── Scalar metrics ────────────────────────────────────────────
        "accuracy":        accuracy,
        "balanced_acc":    balanced_acc,
        "precision":       precision_score_v,
        "recall":          recall_score_v,       # same as sensitivity
        "f1":              f1,
        "f1_macro":        f1_macro,
        "f1_weighted":     f1_weighted,
        "auc_roc":         auc_roc,
        "auprc":           auprc,
        "mcc":             mcc,
        "sensitivity":     float(sensitivity),
        "specificity":     float(specificity),
        "npv":             float(npv),
        "fnr":             float(fnr),
        "fpr_at_threshold": float(fpr_thresh),
        "youden_j":        float(youden_j),
        "optimal_threshold": float(opt_thresh),
        "applied_threshold": float(t),
        # ── Confusion matrix counts ───────────────────────────────────
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion_matrix": cm,
        # ── Curve data for plots ──────────────────────────────────────
        "roc_fpr":          fpr_arr,
        "roc_tpr":          tpr_arr,
        "roc_thresholds":   roc_thresholds,
        "pr_precision":     prec_arr,
        "pr_recall":        rec_arr,
        "pr_thresholds":    pr_thresholds,
        # ── Reports ───────────────────────────────────────────────────
        "classification_report_str":  report_str,
        "classification_report_dict": report_dict,
        # ── Raw arrays (for advanced plots) ──────────────────────────
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "class_names": class_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator Facade
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    High-level evaluator that ties inference + metric computation together.

    Usage:
        evaluator = ModelEvaluator(model, device)
        results   = evaluator.evaluate(test_loader)
        evaluator.save_results(results, "logs/evaluation")
    """

    CLASS_NAMES = ["Healthy (0)", "ALL+ (1)"]

    def __init__(
        self,
        model:     nn.Module,
        device:    torch.device,
        use_amp:   bool = False,
        threshold: Optional[float] = None,
    ) -> None:
        self.model     = model
        self.device    = device
        self.use_amp   = use_amp
        self.threshold = threshold   # None = auto Youden's J

    def evaluate(
        self,
        loader: DataLoader,
        split:  str = "test",
    ) -> Dict[str, Any]:
        """
        Run inference + compute all metrics for the given DataLoader.

        Args:
            loader: DataLoader for the split to evaluate.
            split:  Name used in logging ('test', 'val', etc.)
        Returns:
            Full metrics dict from compute_full_metrics().
        """
        logger.info(f"\n  Running evaluation on '{split}' split...")

        y_true, y_prob, paths = run_inference(
            self.model, loader, self.device,
            use_amp=self.use_amp,
            desc=f"{split.capitalize()}",
        )

        results = compute_full_metrics(
            y_true, y_prob,
            threshold   = self.threshold,
            class_names = self.CLASS_NAMES,
        )
        results["split"] = split
        results["paths"] = paths

        self._print_summary(results)
        return results

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print a clean tabular summary to the console."""
        split = results.get("split", "test")
        print(f"\n{'─'*55}")
        print(f"  EVALUATION RESULTS  [{split.upper()}]")
        print(f"{'─'*55}")
        rows = [
            ("Accuracy",           results["accuracy"]),
            ("Balanced Accuracy",  results["balanced_acc"]),
            ("AUC-ROC",            results["auc_roc"]),
            ("AUPRC",              results["auprc"]),
            ("F1 Score",           results["f1"]),
            ("Precision",          results["precision"]),
            ("Recall/Sensitivity", results["sensitivity"]),
            ("Specificity",        results["specificity"]),
            ("NPV",                results["npv"]),
            ("MCC",                results["mcc"]),
            ("Youden's J",         results["youden_j"]),
            ("Optimal Threshold",  results["optimal_threshold"]),
            ("Applied Threshold",  results["applied_threshold"]),
        ]
        for name, val in rows:
            bar_len = int(val * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {name:<22} {val:.4f}  {bar}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP={results['tp']:>4}  FP={results['fp']:>4}")
        print(f"    FN={results['fn']:>4}  TN={results['tn']:>4}")
        print(f"\n{results['classification_report_str']}")

    def save_results(
        self,
        results:   Dict[str, Any],
        output_dir: str,
        prefix:    str = "",
    ) -> str:
        """
        Save scalar metrics to a JSON file (excludes numpy arrays).

        Args:
            results:    Full results dict.
            output_dir: Directory to write to.
            prefix:     Optional filename prefix.
        Returns:
            Path to saved JSON file.
        """
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{prefix}metrics.json" if prefix else "metrics.json"
        path  = os.path.join(output_dir, fname)

        # Serialize only JSON-safe scalars
        safe = {
            k: (float(v) if isinstance(v, (float, np.floating)) else
                int(v)   if isinstance(v, (int,   np.integer))  else
                v)
            for k, v in results.items()
            if isinstance(v, (float, int, str, np.floating, np.integer))
        }
        with open(path, "w") as f:
            json.dump(safe, f, indent=2)
        logger.info(f"  Metrics saved: {path}")
        return path
