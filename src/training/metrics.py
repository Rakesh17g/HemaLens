"""
Training Metrics
=================
Medical-grade evaluation metrics for binary classification.

For leukemia detection, we track (in priority order):
  1. AUC-ROC     — threshold-independent, best single metric for medical AI
  2. Sensitivity — recall for positive class (missing cancer = worst outcome)
  3. Specificity — recall for negative class (false alarms waste resources)
  4. F1 Score    — harmonic mean of precision/recall
  5. Accuracy    — useful but misleading if classes are imbalanced
  6. MCC         — Matthews Correlation Coefficient, robust to imbalance

We also compute optimal operating threshold from the ROC curve
(Youden's J statistic: max(sensitivity + specificity - 1)).
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    average_precision_score,
    roc_curve,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true:    np.ndarray,       # (N,) int  {0, 1}
    y_prob:    np.ndarray,       # (N,) float probabilities [0, 1]
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Compute the full medical-grade metric suite.

    Args:
        y_true:    Ground-truth binary labels.
        y_prob:    Predicted probabilities (sigmoid output).
        threshold: Decision boundary for binary prediction.

    Returns:
        Dict of metric_name → float value.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # ── Confusion Matrix components ───────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-8)   # True Positive Rate
    specificity = tn / (tn + fp + 1e-8)   # True Negative Rate
    precision   = tp / (tp + fp + 1e-8)   # Positive Predictive Value
    npv         = tn / (tn + fn + 1e-8)   # Negative Predictive Value

    # ── Aggregate metrics ─────────────────────────────────────────────
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = 0.0

    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    return {
        "accuracy":    acc,
        "auc_roc":     auc,
        "auprc":       auprc,
        "f1":          f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision,
        "npv":         npv,
        "mcc":         mcc,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "threshold":   threshold,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """
    Find the operating threshold that maximizes Youden's J statistic.

    Youden's J = Sensitivity + Specificity − 1  (range [−1, 1])
    Maximizing J balances sensitivity and specificity optimally.

    In medical AI, you might want to shift the threshold lower (e.g. 0.35)
    to prioritize sensitivity over specificity (miss fewer cancer cases).

    Returns:
        (optimal_threshold, best_youden_j)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    specificity = 1.0 - fpr
    j_scores    = tpr + specificity - 1.0          # Youden's J
    best_idx    = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Batch Accumulator (used in training loop)
# ─────────────────────────────────────────────────────────────────────────────

class MetricAccumulator:
    """
    Accumulates per-batch predictions and targets across an epoch,
    then computes full metrics once per epoch (not per-batch).

    This is important because metrics like AUC-ROC and Youden's J
    are only meaningful across the FULL epoch, not per mini-batch.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._probs:  list = []    # list of numpy float arrays
        self._labels: list = []    # list of numpy int arrays
        self._losses: list = []    # list of float loss values
        self._n_batches: int = 0

    def update(
        self,
        logits:  torch.Tensor,   # (B, 1) or (B,) raw logits
        labels:  torch.Tensor,   # (B,) float labels {0, 1}
        loss:    float,
    ) -> None:
        with torch.no_grad():
            probs = torch.sigmoid(logits.detach().cpu()).view(-1).numpy()
            lbls  = labels.detach().cpu().view(-1).long().numpy()
        self._probs.append(probs)
        self._labels.append(lbls)
        self._losses.append(float(loss))
        self._n_batches += 1

    def compute(self, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute all metrics over the accumulated epoch.

        Args:
            threshold: Decision boundary. If None, uses Youden's J to find optimal.
        Returns:
            Dict of metric name → value, plus 'loss' and 'optimal_threshold'.
        """
        y_prob  = np.concatenate(self._probs)
        y_true  = np.concatenate(self._labels)
        avg_loss = float(np.mean(self._losses))

        # Find optimal threshold if not specified
        opt_thresh, youden_j = find_optimal_threshold(y_true, y_prob)
        t = threshold if threshold is not None else opt_thresh

        metrics = compute_metrics(y_true, y_prob, threshold=t)
        metrics["loss"]               = avg_loss
        metrics["optimal_threshold"]  = opt_thresh
        metrics["youden_j"]           = youden_j
        # Store raw arrays for ROC curve plotting
        metrics["_y_true"] = y_true
        metrics["_y_prob"] = y_prob

        return metrics

    def compute_running_loss(self) -> float:
        """Fast loss-only computation for mid-epoch progress bars."""
        return float(np.mean(self._losses)) if self._losses else 0.0
