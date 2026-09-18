"""
Evaluation Engine
==================
The top-level orchestrator for model evaluation.

Responsibilities
----------------
1. Run inference over any DataLoader            (via run_inference)
2. Compute the full medical-grade metric suite  (via compute_full_metrics)
3. Render every visualization with Matplotlib   (via generate_all_plots)
4. Persist metrics to JSON + plots to disk
5. Expose a clean, standalone CLI

Key design principles
---------------------
- Zero coupling to Trainer: accepts any nn.Module + DataLoader
- Dependency-injected config (dict or YAML path)
- Every sub-step can be called independently
- All arrays (ROC, PR, CM) are stored in the results dict so plots
  can be regenerated at any time without re-running inference

Metrics produced
----------------
  Scalar metrics
  ──────────────
  accuracy           · balanced_accuracy  · precision        · recall
  f1 (binary)        · f1_macro           · f1_weighted      · auc_roc
  auprc              · mcc (Matthews CC)  · sensitivity      · specificity
  npv                · fnr                · fpr_at_threshold · youden_j
  optimal_threshold  · applied_threshold

  Confusion matrix (2×2 numpy array + TP/TN/FP/FN counts)

  Curve arrays (for reproduction / further analysis)
  ────────────────────────────────────────────────────
  roc_fpr  · roc_tpr  · roc_thresholds
  pr_precision · pr_recall · pr_thresholds

  Reports
  ───────
  classification_report_str  (sklearn text)
  classification_report_dict (dict, per-class)

Visualizations saved
--------------------
  1. metrics_dashboard.png       — gauge scorecard for all scalars
  2. roc_curve.png               — ROC + AUC + Youden's J marker
  3. pr_curve.png                — PR curve + AUPRC + prevalence baseline
  4. confusion_matrix.png        — annotated heatmap (TP/TN/FP/FN)
  5. classification_report.png   — styled sklearn table
  6. threshold_analysis.png      — F1/Sens/Spec vs threshold sweep
  7. probability_distribution.png— predicted prob histogram by class
  8. full_report.png             — single-page composite PDF-style report
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# ── Project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.inference.confidence import ConfidenceEstimator
from src.inference.confidence_viz import generate_confidence_plots
from src.training.evaluator import compute_full_metrics, run_inference
from src.training.visualizer import generate_all_plots

logger = logging.getLogger("ALLEvaluation")


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationEngine
# ─────────────────────────────────────────────────────────────────────────────


class EvaluationEngine:
    """
    Unified evaluation engine for the ALL detection model.

    Args
    ----
    model:      Trained nn.Module in CPU/CUDA memory.
    device:     torch.device to run inference on.
    cfg:        Config dict with optional ``evaluation`` section.
    threshold:  Fixed decision threshold.  None → auto via Youden's J.
    use_amp:    Use mixed-precision inference (GPU only).
    class_names: Human-readable class labels [negative, positive].

    Quick start
    -----------
    >>> engine  = EvaluationEngine(model, device)
    >>> results = engine.run(test_loader, split="test")
    >>> engine.save(results, output_dir="logs/evaluation/test")
    """

    CLASS_NAMES: list[str] = ["Healthy (0)", "ALL+ (1)"]  # noqa: RUF012

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cfg: dict[str, Any] | None = None,
        threshold: float | None = None,
        use_amp: bool = False,
        class_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.cfg = cfg or {}
        self.use_amp = use_amp
        self.class_names = class_names or self.CLASS_NAMES

        # Threshold: from arg → config → None (auto)
        eval_cfg = self.cfg.get("evaluation", {})
        self.threshold = (
            threshold if threshold is not None else eval_cfg.get("threshold", None)
        )

        # Output directories
        self.default_output_dir = eval_cfg.get("output_dir", "logs/evaluation")
        self.model_name = eval_cfg.get("model_name", "EfficientNet-B0")

        logger.info(
            f"EvaluationEngine ready | device={device} | "
            f"threshold={'auto' if self.threshold is None else self.threshold:.3f}"
        )

    # ── Core: inference → metrics ─────────────────────────────────────────────

    def run_inference(
        self,
        loader: DataLoader,
        split: str = "test",
    ):
        """
        Run full inference over *loader* and return raw arrays.

        Returns
        -------
        y_true : (N,) int ndarray   ground-truth labels {0, 1}
        y_prob : (N,) float ndarray predicted probabilities [0, 1]
        paths  : list[str] | None   file paths if loader provides them
        """
        logger.info(f"  Running inference on '{split}' split …")
        t0 = time.perf_counter()

        y_true, y_prob, paths = run_inference(
            model=self.model,
            loader=loader,
            device=self.device,
            use_amp=self.use_amp and self.device.type == "cuda",
            desc=f"{split.capitalize()} Inference",
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"  Inference complete | N={len(y_true)} | "
            f"pos_rate={y_true.mean():.3f} | {elapsed:.1f}s"
        )
        return y_true, y_prob, paths

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute the full metric suite from pre-collected arrays.

        All metrics listed in the module docstring are included in
        the returned dict, along with numpy arrays for curves and
        the raw y_true / y_prob / y_pred arrays.
        """
        logger.info("  Computing metrics …")
        results = compute_full_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=self.threshold,
            class_names=self.class_names,
        )
        self._log_scalar_summary(results)
        return results

    # ── Combined: run inference + compute metrics in one call ─────────────────

    def run(
        self,
        loader: DataLoader,
        split: str = "test",
    ) -> dict[str, Any]:
        """
        Run inference **and** compute metrics in one call.

        This is the primary entry point for evaluation.

        Args
        ----
        loader : DataLoader for the split to evaluate.
        split  : Name label used in logs/titles ('test', 'val', …).

        Returns
        -------
        results : Complete evaluation dict — see module docstring for keys.
        """
        logger.info(f"\n{'─' * 55}")
        logger.info(f"  EVALUATION  [{split.upper()}]")
        logger.info(f"{'─' * 55}")

        y_true, y_prob, paths = self.run_inference(loader, split=split)
        results = self.compute_metrics(y_true, y_prob)
        results["split"] = split
        results["paths"] = paths
        return results

    # ── Visualization ─────────────────────────────────────────────────────────

    def visualize(
        self,
        results: dict[str, Any],
        output_dir: str | None = None,
    ) -> dict[str, str]:
        """
        Generate all 8 Matplotlib visualizations and save to *output_dir*.

        Plot list
        ---------
        1. metrics_dashboard.png
        2. roc_curve.png
        3. pr_curve.png
        4. confusion_matrix.png
        5. classification_report.png
        6. threshold_analysis.png
        7. probability_distribution.png
        8. full_report.png (single-page composite)

        Returns
        -------
        Dict mapping plot name → absolute file path.
        """
        out = output_dir or os.path.join(
            self.default_output_dir, results.get("split", "eval")
        )
        logger.info(f"\n  Generating visualizations → {os.path.abspath(out)}/")

        saved = generate_all_plots(
            results=results,
            output_dir=out,
            model_name=self.model_name,
            split=results.get("split", "eval"),
        )
        logger.info(f"  {len(saved)} plots saved.")
        return saved

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_metrics(
        self,
        results: dict[str, Any],
        output_dir: str,
        prefix: str = "",
    ) -> str:
        """
        Serialize scalar metrics (JSON-safe values only) to disk.

        Numpy arrays (roc_fpr, confusion_matrix, etc.) and string
        reports are intentionally excluded from the JSON. They can be
        regenerated from the saved y_true / y_prob arrays.

        Returns
        -------
        Absolute path to the saved JSON file.
        """
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{prefix}metrics.json" if prefix else "metrics.json"
        path = os.path.join(output_dir, fname)

        safe: dict[str, Any] = {}
        for k, v in results.items():
            if isinstance(v, (float, np.floating)):
                safe[k] = float(v)
            elif isinstance(v, (int, np.integer)):
                safe[k] = int(v)
            elif isinstance(v, str):
                safe[k] = v

        with open(path, "w") as f:
            json.dump(safe, f, indent=2)

        logger.info(f"  Metrics JSON saved: {path}")
        return path

    def save_arrays(
        self,
        results: dict[str, Any],
        output_dir: str,
    ) -> str:
        """
        Persist raw prediction arrays (y_true, y_prob, y_pred) as a
        compressed NumPy archive (.npz) so plots can be regenerated
        offline without re-running inference.

        Returns
        -------
        Absolute path to the saved .npz file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "predictions.npz")
        np.savez_compressed(
            path,
            y_true=results["y_true"],
            y_prob=results["y_prob"],
            y_pred=results["y_pred"],
        )
        logger.info(f"  Prediction arrays saved: {path}")
        return path

    def save_report(
        self,
        results: dict[str, Any],
        output_dir: str,
    ) -> str:
        """
        Save the sklearn classification_report as a plain-text .txt file.

        Returns
        -------
        Absolute path to the saved text file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "classification_report.txt")
        with open(path, "w") as f:
            split = results.get("split", "eval").upper()
            f.write(f"Classification Report [{split}]\n")
            f.write("=" * 55 + "\n\n")
            f.write(results["classification_report_str"])
            f.write("\n\n")
            # Additional scalar summary
            f.write("Scalar Summary\n")
            f.write("-" * 40 + "\n")
            for key in [
                "accuracy",
                "balanced_acc",
                "precision",
                "recall",
                "f1",
                "f1_macro",
                "f1_weighted",
                "auc_roc",
                "auprc",
                "sensitivity",
                "specificity",
                "npv",
                "mcc",
                "optimal_threshold",
                "applied_threshold",
                "youden_j",
            ]:
                val = results.get(key, 0.0)
                f.write(f"  {key:<25} {val:.6f}\n")
            f.write("\n")
            f.write(
                f"  TP={results['tp']}  TN={results['tn']}  "
                f"FP={results['fp']}  FN={results['fn']}\n"
            )

        logger.info(f"  Text report saved: {path}")
        return path

    # ── All-in-one: save everything ───────────────────────────────────────────

    def save(
        self,
        results: dict[str, Any],
        output_dir: str | None = None,
        prefix: str = "",
    ) -> dict[str, str]:
        """
        Save metrics JSON + text report + prediction arrays + all plots.

        This is the one-liner call at the end of evaluation.

        Args
        ----
        results    : Dict returned by ``run()``.
        output_dir : Root directory.  Defaults to config ``evaluation.output_dir``.
        prefix     : Optional string prepended to the metrics JSON filename.

        Returns
        -------
        Dict mapping artifact name → absolute path.
        """
        out = output_dir or os.path.join(
            self.default_output_dir, results.get("split", "eval")
        )
        os.makedirs(out, exist_ok=True)

        saved: dict[str, str] = {}
        saved["metrics_json"] = self.save_metrics(results, out, prefix)
        saved["classification_report"] = self.save_report(results, out)
        saved["predictions_npz"] = self.save_arrays(results, out)

        # ── 8 standard evaluation plots ───────────────────────────────────
        plot_paths = self.visualize(results, output_dir=out)
        saved.update(plot_paths)

        # ── Confidence estimation + plots ─────────────────────────────────
        conf_dir = os.path.join(out, "confidence")
        try:
            eval_cfg = self.cfg.get("evaluation", {})
            mc_passes = eval_cfg.get("mc_dropout_passes", 0)  # 0=fast/no MC
            threshold = results.get("applied_threshold", 0.50)

            estimator = ConfidenceEstimator(
                threshold=float(threshold),
                mc_dropout_passes=mc_passes,
                use_amp=self.use_amp,
            )
            # Batch estimation from pre-collected probabilities (no re-inference)
            conf_results = estimator.estimate_batch(results["y_prob"])
            conf_summary = ConfidenceEstimator.batch_summary(conf_results)

            # Persist confidence summary JSON
            os.makedirs(conf_dir, exist_ok=True)
            conf_json_path = os.path.join(conf_dir, "confidence_summary.json")
            import json as _json

            with open(conf_json_path, "w") as cf:
                _json.dump(conf_summary, cf, indent=2)
            saved["confidence_summary_json"] = conf_json_path
            logger.info(f"  Confidence summary saved: {conf_json_path}")

            # Print confidence explanation + summary
            print(ConfidenceEstimator.explain())
            print(f"\n  Confidence Summary [{results.get('split', 'eval').upper()}]")
            print(f"  Mean Confidence : {conf_summary['mean_confidence']:.4f}")
            print(
                f"  LOW risk        : {conf_summary['n_low']} "
                f"({conf_summary['risk_fractions']['LOW'] * 100:.1f}%)"  # type: ignore
            )
            print(
                f"  MEDIUM risk     : {conf_summary['n_medium']} "
                f"({conf_summary['risk_fractions']['MEDIUM'] * 100:.1f}%)"  # type: ignore
            )
            print(
                f"  HIGH risk       : {conf_summary['n_high']} "
                f"({conf_summary['risk_fractions']['HIGH'] * 100:.1f}%)"  # type: ignore
            )

            # Generate all confidence visualizations
            conf_plot_paths = generate_confidence_plots(
                results=conf_results,
                y_true=results["y_true"],
                output_dir=conf_dir,
            )
            saved.update({f"conf_{k}": v for k, v in conf_plot_paths.items()})

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  Confidence estimation skipped: {exc}")

        logger.info(
            f"\n  Evaluation artifacts saved → {os.path.abspath(out)}/\n"
            f"  Total files: {len(saved)}"
        )
        return saved

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log_scalar_summary(self, results: dict[str, Any]) -> None:
        """Print a rich tabular metric summary to the console."""
        split = results.get("split", "eval").upper()

        print(f"\n{'─' * 60}")
        print(f"  EVALUATION RESULTS  [{split}]")
        print(f"{'─' * 60}")

        # ── Section 1: Core metrics ───────────────────────────────────────
        rows_core = [
            ("Accuracy", results["accuracy"]),
            ("Balanced Accuracy", results["balanced_acc"]),
            ("AUC-ROC", results["auc_roc"]),
            ("AUPRC", results["auprc"]),
            ("F1 (binary)", results["f1"]),
            ("F1 Macro", results["f1_macro"]),
            ("F1 Weighted", results["f1_weighted"]),
        ]
        print(f"\n  {'Core Metrics':─<50}")
        for name, val in rows_core:
            bar = "█" * int(val * 24) + "░" * (24 - int(val * 24))
            print(f"  {name:<22} {val:.4f}  {bar}")

        # ── Section 2: Clinical metrics ───────────────────────────────────
        rows_clin = [
            ("Precision", results["precision"]),
            ("Recall", results["recall"]),
            ("Sensitivity", results["sensitivity"]),
            ("Specificity", results["specificity"]),
            ("NPV", results["npv"]),
            ("MCC", results["mcc"]),
            ("Youden's J", results["youden_j"]),
        ]
        print(f"\n  {'Clinical Metrics':─<50}")
        for name, val in rows_clin:
            bar = "█" * int(max(0.0, val) * 24) + "░" * (24 - int(max(0.0, val) * 24))
            print(f"  {name:<22} {val:.4f}  {bar}")

        # ── Section 3: Confusion matrix ───────────────────────────────────
        print(f"\n  {'Confusion Matrix':─<50}")
        print("               Predicted Healthy   Predicted ALL+")
        print(f"  True Healthy    TN={results['tn']:<8}    FP={results['fp']}")
        print(f"  True ALL+       FN={results['fn']:<8}    TP={results['tp']}")

        # ── Section 4: Threshold ──────────────────────────────────────────
        print(f"\n  {'Threshold':─<50}")
        print(f"  Optimal (Youden's J)  {results['optimal_threshold']:.4f}")
        print(f"  Applied               {results['applied_threshold']:.4f}")

        # ── Section 5: Classification report ─────────────────────────────
        print(f"\n  {'Classification Report':─<50}")
        print(results["classification_report_str"])
