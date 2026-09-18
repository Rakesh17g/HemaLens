"""
Evaluation Visualizer
======================
Generates a suite of publication-quality dark-mode plots covering
every requested metric. All plots share a consistent design language.

Plots produced:
  1. metrics_dashboard.png  — scorecard grid with all scalar metrics
  2. roc_curve.png          — ROC curve with AUC + Youden's J point
  3. pr_curve.png           — Precision-Recall curve with AUPRC
  4. confusion_matrix.png   — annotated heatmap
  5. classification_report.png — styled table of per-class metrics
  6. threshold_analysis.png — how F1/Sens/Spec vary with threshold
  7. probability_dist.png   — predicted probability histogram by class
  8. evaluation_report.png  — full single-page summary (all-in-one)
"""

import os
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# Design System
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg": "#0B1120",
    "card": "#131D2E",
    "surface": "#1A2744",
    "border": "#243354",
    "accent1": "#6366F1",  # indigo  — positive / AUC curves
    "accent2": "#22D3EE",  # cyan    — secondary curves
    "accent3": "#F59E0B",  # amber   — threshold markers
    "positive": "#EF4444",  # red     — ALL+ class
    "negative": "#22C55E",  # green   — Healthy class
    "text": "#F1F5F9",
    "subtext": "#94A3B8",
    "grid": "#1E3A5F",
}


def _setup_fig(fig: plt.Figure, title: str = "") -> None:
    fig.patch.set_facecolor(PALETTE["bg"])
    if title:
        fig.suptitle(
            title, color=PALETTE["text"], fontsize=13, fontweight="bold", y=0.98
        )


def _setup_ax(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = True,
) -> None:
    ax.set_facecolor(PALETTE["card"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["subtext"])
    ax.yaxis.label.set_color(PALETTE["subtext"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=9, fontweight="bold", pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if grid:
        ax.grid(color=PALETTE["grid"], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str, dpi: int = 130) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"    Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metrics Dashboard
# ─────────────────────────────────────────────────────────────────────────────


def plot_metrics_dashboard(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    A 4×3 grid of metric scorecards — the at-a-glance summary page.
    Each card shows the metric name, value as a large number, and a
    color-coded arc gauge.
    """
    metrics = [
        ("Accuracy", results["accuracy"], "#6366F1"),
        ("Balanced Accuracy", results["balanced_acc"], "#818CF8"),
        ("AUC-ROC", results["auc_roc"], "#22D3EE"),
        ("AUPRC", results["auprc"], "#06B6D4"),
        ("F1 Score", results["f1"], "#F59E0B"),
        ("F1 Macro", results["f1_macro"], "#FBBF24"),
        ("Precision", results["precision"], "#22C55E"),
        ("Recall/Sensitivity", results["sensitivity"], "#EF4444"),
        ("Specificity", results["specificity"], "#34D399"),
        ("MCC", results["mcc"], "#A78BFA"),
        ("Youden's J", results["youden_j"], "#FB923C"),
        ("NPV", results["npv"], "#38BDF8"),
    ]

    cols, rows = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9))
    _setup_fig(fig, "ALL Detection — Evaluation Metrics Dashboard")
    axes = axes.flatten()

    for ax, (name, val, color) in zip(axes, metrics):
        ax.set_facecolor(PALETTE["surface"])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)

        # Arc gauge (donut-style)
        val_clipped = max(0.0, min(1.0, float(val)))
        theta = np.linspace(0, 2 * np.pi * val_clipped, 120)
        theta_bg = np.linspace(0, 2 * np.pi, 120)
        r = 0.38

        ax.plot(
            r * np.cos(theta_bg) + 0.5,
            r * np.sin(theta_bg) + 0.55,
            color=PALETTE["border"],
            linewidth=6,
            solid_capstyle="round",
        )
        if val_clipped > 0.01:
            ax.plot(
                r * np.cos(theta) + 0.5,
                r * np.sin(theta) + 0.55,
                color=color,
                linewidth=6,
                solid_capstyle="round",
            )

        # Value text
        ax.text(
            0.5,
            0.52,
            f"{val:.4f}",
            ha="center",
            va="center",
            color=color,
            fontsize=16,
            fontweight="bold",
            transform=ax.transAxes,
        )

        # Metric name
        ax.text(
            0.5,
            0.18,
            name,
            ha="center",
            va="center",
            color=PALETTE["subtext"],
            fontsize=8.5,
            transform=ax.transAxes,
            wrap=True,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Threshold info strip
    fig.text(
        0.5,
        0.01,
        f"Applied Threshold: {results['applied_threshold']:.3f}   "
        f"Youden-J Optimal: {results['optimal_threshold']:.3f}   "
        f"TP={results['tp']}  TN={results['tn']}  "
        f"FP={results['fp']}  FN={results['fn']}",
        ha="center",
        va="bottom",
        color=PALETTE["subtext"],
        fontsize=8,
    )

    plt.tight_layout(pad=1.2, rect=[0, 0.03, 1, 0.96])  # type: ignore
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROC Curve
# ─────────────────────────────────────────────────────────────────────────────


def plot_roc_curve(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    ROC curve with:
    - Shaded area under the curve
    - Youden's J operating point highlighted
    - Random classifier diagonal
    - AUC annotation
    """
    fpr = results["roc_fpr"]
    tpr = results["roc_tpr"]
    auc = results["auc_roc"]
    opt_t = results["optimal_threshold"]
    # Find the point on the curve at the optimal threshold
    thresh = results["roc_thresholds"]
    idx = np.argmin(np.abs(thresh - opt_t))

    fig, ax = plt.subplots(figsize=(7, 6))
    _setup_fig(fig, "ROC Curve — Receiver Operating Characteristic")
    _setup_ax(
        ax,
        xlabel="False Positive Rate (1 − Specificity)",
        ylabel="True Positive Rate (Sensitivity)",
    )

    # Shaded area
    ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["accent1"])

    # ROC line
    ax.plot(
        fpr,
        tpr,
        color=PALETTE["accent1"],
        linewidth=2.5,
        label=f"EfficientNet-B0  (AUC = {auc:.4f})",
    )

    # Diagonal (random)
    ax.plot(
        [0, 1],
        [0, 1],
        color=PALETTE["subtext"],
        linewidth=1,
        linestyle="--",
        label="Random Classifier (AUC = 0.50)",
    )

    # Youden's J operating point
    ax.scatter(
        fpr[idx],
        tpr[idx],
        s=120,
        color=PALETTE["accent3"],
        zorder=5,
        label=f"Youden's J Point  (t={opt_t:.3f})",
    )
    ax.annotate(
        f" Sens={tpr[idx]:.3f}\n Spec={1 - fpr[idx]:.3f}",
        xy=(fpr[idx], tpr[idx]),
        xytext=(fpr[idx] + 0.05, tpr[idx] - 0.08),
        color=PALETTE["accent3"],
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": PALETTE["accent3"], "lw": 1.0},
    )

    # Formatting
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(
        loc="lower right",
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
    )

    # AUC badge
    ax.text(
        0.97,
        0.08,
        f"AUC\n{auc:.4f}",
        ha="right",
        va="bottom",
        color=PALETTE["accent1"],
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────


def plot_pr_curve(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    Precision-Recall curve with AUPRC, baseline (prevalence), and
    the operating point at the applied threshold.
    """
    prec = results["pr_precision"]
    rec = results["pr_recall"]
    auprc = results["auprc"]
    prevalence = results["y_true"].mean()

    # Operating point
    op_prec = results["precision"]
    op_rec = results["recall"]

    fig, ax = plt.subplots(figsize=(7, 6))
    _setup_fig(fig, "Precision-Recall Curve")
    _setup_ax(ax, xlabel="Recall (Sensitivity)", ylabel="Precision (PPV)")

    ax.fill_between(rec, prec, alpha=0.12, color=PALETTE["accent2"])
    ax.plot(
        rec,
        prec,
        color=PALETTE["accent2"],
        linewidth=2.5,
        label=f"EfficientNet-B0  (AUPRC = {auprc:.4f})",
    )
    ax.axhline(
        prevalence,
        color=PALETTE["subtext"],
        linewidth=1,
        linestyle="--",
        label=f"Chance (prevalence = {prevalence:.3f})",
    )
    ax.scatter(
        op_rec,
        op_prec,
        s=120,
        color=PALETTE["accent3"],
        zorder=5,
        label=f"Operating Point  (t={results['applied_threshold']:.3f})",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend(
        loc="lower left",
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
    )
    ax.text(
        0.97,
        0.97,
        f"AUPRC\n{auprc:.4f}",
        ha="right",
        va="top",
        color=PALETTE["accent2"],
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────


def plot_confusion_matrix(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    Annotated confusion matrix heatmap with counts, percentages,
    and per-cell interpretation labels.
    """
    cm = results["confusion_matrix"]  # (2, 2)
    names = results.get("class_names", ["Negative", "Positive"])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _setup_fig(fig, "Confusion Matrix")

    # Custom colormap: dark purple → bright indigo
    cmap = LinearSegmentedColormap.from_list(
        "medical", [PALETTE["card"], PALETTE["accent1"]]
    )
    im = ax.imshow(cm, cmap=cmap, aspect="auto", vmin=0, vmax=total)

    # Annotations
    cell_labels = [
        ["TN\n(Correct\nNegative)", "FP\n(False\nAlarm)"],
        ["FN\n(Missed\nCancer!)", "TP\n(Correct\nPositive)"],
    ]
    label_colors = [
        [PALETTE["negative"], PALETTE["accent3"]],
        [PALETTE["positive"], PALETTE["negative"]],
    ]

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100 * count / total
            txt_col = PALETTE["text"]
            ax.text(
                j,
                i - 0.18,
                str(count),
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color=txt_col,
            )
            ax.text(
                j,
                i + 0.10,
                f"{pct:.1f}% of total",
                ha="center",
                va="center",
                fontsize=8.5,
                color=txt_col,
            )
            ax.text(
                j,
                i + 0.35,
                cell_labels[i][j],
                ha="center",
                va="center",
                fontsize=7.5,
                color=label_colors[i][j],
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(names, color=PALETTE["text"], fontsize=10)
    ax.set_yticklabels(
        names, color=PALETTE["text"], fontsize=10, rotation=90, va="center"
    )
    ax.set_xlabel("Predicted Label", color=PALETTE["subtext"], fontsize=9)
    ax.set_ylabel("True Label", color=PALETTE["subtext"], fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.tick_params(colors=PALETTE["subtext"])
    cbar.set_label("Count", color=PALETTE["subtext"], fontsize=8)

    # Summary strip
    fig.text(
        0.5,
        0.01,
        f"Sensitivity: {results['sensitivity']:.4f}   "
        f"Specificity: {results['specificity']:.4f}   "
        f"Total samples: {total}",
        ha="center",
        color=PALETTE["subtext"],
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])  # type: ignore
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Classification Report Table
# ─────────────────────────────────────────────────────────────────────────────


def plot_classification_report(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    Render sklearn's classification_report as a styled matplotlib table.
    """
    report_dict = results["classification_report_dict"]
    class_names = results.get("class_names", ["Healthy (0)", "ALL+ (1)"])

    rows = []

    for cls_name in class_names:
        r = report_dict.get(cls_name, {})
        rows.append(
            [
                cls_name,
                f"{r.get('precision', 0):.4f}",
                f"{r.get('recall', 0):.4f}",
                f"{r.get('f1-score', 0):.4f}",
                str(int(r.get("support", 0))),
            ]
        )
    # Accuracy row
    rows.append(["", "", "", "", ""])
    rows.append(
        [
            "Accuracy",
            "",
            "",
            f"{results['accuracy']:.4f}",
            str(results["tp"] + results["tn"] + results["fp"] + results["fn"]),
        ]
    )
    for avg_key in ["macro avg", "weighted avg"]:
        r = report_dict.get(avg_key, {})
        rows.append(
            [
                avg_key.title(),
                f"{r.get('precision', 0):.4f}",
                f"{r.get('recall', 0):.4f}",
                f"{r.get('f1-score', 0):.4f}",
                str(int(r.get("support", 0))),
            ]
        )

    fig, ax = plt.subplots(figsize=(9, 4))
    _setup_fig(fig, "Classification Report")
    ax.axis("off")

    header_row = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    all_rows = [header_row] + rows
    n_rows = len(all_rows)

    # Draw rows
    col_widths = [0.30, 0.175, 0.175, 0.175, 0.175]
    x_starts = [0.02 + sum(col_widths[:i]) for i in range(5)]
    row_h = 0.85 / n_rows
    y_start = 0.90

    for r_idx, row in enumerate(all_rows):
        y = y_start - r_idx * row_h
        is_header = r_idx == 0
        is_empty = all(c == "" for c in row)
        is_class_row = r_idx in (1, 2)

        bg_col = (
            PALETTE["surface"]
            if is_header
            else (PALETTE["card"] if r_idx % 2 == 0 else PALETTE["bg"])
        )
        if not is_empty:
            rect = mpatches.FancyBboxPatch(
                (0.01, y - row_h * 0.85),
                0.98,
                row_h * 0.9,
                transform=ax.transAxes,
                boxstyle="round,pad=0.005",
                facecolor=bg_col,
                edgecolor=PALETTE["border"],
                linewidth=0.5,
            )
            ax.add_patch(rect)

        for c_idx, (cell, xw) in enumerate(zip(row, x_starts)):
            fw = "bold" if is_header else "normal"
            color = PALETTE["text"] if is_header else PALETTE["subtext"]

            if is_class_row and c_idx == 0:
                color = PALETTE["positive"] if r_idx == 2 else PALETTE["negative"]

            ax.text(
                xw + col_widths[c_idx] / 2,
                y - row_h * 0.35,
                cell,
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
                fontweight=fw,
                transform=ax.transAxes,
            )

    plt.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Threshold Analysis
# ─────────────────────────────────────────────────────────────────────────────


def plot_threshold_analysis(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    Shows how F1, Sensitivity, Specificity, and Precision change
    as the decision threshold varies from 0 to 1.
    Highlights the Youden's J optimal point.
    """
    y_true = results["y_true"]
    y_prob = results["y_prob"]
    opt_t = results["optimal_threshold"]

    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, sens, spec, prec = [], [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = (
            [0, 0, 0, 0]
            if y_pred.sum() == 0 or y_pred.sum() == len(y_pred)
            else __import__("sklearn.metrics", fromlist=["confusion_matrix"])
            .confusion_matrix(y_true, y_pred, labels=[0, 1])
            .ravel()
        )
        f1s.append(
            __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                y_true, y_pred, zero_division=0
            )
        )
        sens.append(tp / (tp + fn + 1e-8))
        spec.append(tn / (tn + fp + 1e-8))
        prec.append(tp / (tp + fp + 1e-8))

    fig, ax = plt.subplots(figsize=(9, 5))
    _setup_fig(fig, "Metric Sensitivity to Decision Threshold")
    _setup_ax(ax, xlabel="Decision Threshold", ylabel="Metric Value")

    lines = [
        (f1s, PALETTE["accent1"], "F1 Score"),
        (sens, PALETTE["positive"], "Sensitivity (Recall)"),
        (spec, PALETTE["negative"], "Specificity"),
        (prec, PALETTE["accent3"], "Precision"),
    ]
    for vals, color, label in lines:
        ax.plot(thresholds, vals, color=color, linewidth=2, label=label)

    # Optimal threshold line
    ax.axvline(
        opt_t, color=PALETTE["accent3"], linewidth=1.5, linestyle="--", alpha=0.8
    )
    ax.text(
        opt_t + 0.01,
        0.97,
        f"Optimal t={opt_t:.3f}",
        color=PALETTE["accent3"],
        fontsize=8,
        va="top",
    )

    # Applied threshold marker
    applied_t = results["applied_threshold"]
    if abs(applied_t - opt_t) > 0.01:
        ax.axvline(
            applied_t, color=PALETTE["accent2"], linewidth=1.0, linestyle=":", alpha=0.7
        )
        ax.text(
            applied_t + 0.01,
            0.88,
            f"Applied t={applied_t:.3f}",
            color=PALETTE["accent2"],
            fontsize=7.5,
            va="top",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(
        loc="lower center",
        ncol=4,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
    )

    plt.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Probability Distribution
# ─────────────────────────────────────────────────────────────────────────────


def plot_probability_distribution(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    Histogram of predicted probabilities, split by class.
    Well-calibrated models show clearly separated distributions.
    """
    y_true = results["y_true"]
    y_prob = results["y_prob"]
    t = results["applied_threshold"]

    fig, ax = plt.subplots(figsize=(9, 5))
    _setup_fig(fig, "Predicted Probability Distribution by Class")
    _setup_ax(ax, xlabel="Predicted Probability (P[ALL+])", ylabel="Number of Samples")

    bins = np.linspace(0, 1, 41)
    pos_idx = y_true == 1
    neg_idx = y_true == 0

    ax.hist(
        y_prob[neg_idx],
        bins=bins,  # type: ignore
        color=PALETTE["negative"],
        alpha=0.75,
        label=f"Healthy (0)  n={neg_idx.sum()}",
        edgecolor=PALETTE["bg"],
        linewidth=0.4,
    )
    ax.hist(
        y_prob[pos_idx],
        bins=bins,  # type: ignore
        color=PALETTE["positive"],
        alpha=0.75,
        label=f"ALL+ (1)     n={pos_idx.sum()}",
        edgecolor=PALETTE["bg"],
        linewidth=0.4,
    )

    ax.axvline(
        t,
        color=PALETTE["accent3"],
        linewidth=2,
        linestyle="--",
        label=f"Threshold  t={t:.3f}",
    )

    # Median markers
    for prob_arr, color, side in [
        (y_prob[neg_idx], PALETTE["negative"], "bottom"),
        (y_prob[pos_idx], PALETTE["positive"], "top"),
    ]:
        if len(prob_arr):
            med = np.median(prob_arr)
            ax.axvline(med, color=color, linewidth=1.0, linestyle=":", alpha=0.7)

    ax.legend(
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=9,
    )
    plt.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full Evaluation Report (Single-Page)
# ─────────────────────────────────────────────────────────────────────────────


def plot_evaluation_report(
    results: dict[str, Any],
    output_path: str,
    model_name: str = "EfficientNet-B0",
    split: str = "Test",
) -> None:
    """
    Single-page PDF-style report combining:
    ROC + PR + Confusion Matrix + Key Metrics + Threshold Analysis
    """
    fig = plt.figure(figsize=(16, 11))
    _setup_fig(fig, f"{model_name} — Full Evaluation Report  [{split} Set]")

    gs = gridspec.GridSpec(
        2,
        3,
        hspace=0.42,
        wspace=0.32,
        top=0.91,
        bottom=0.06,
        left=0.06,
        right=0.97,
    )

    # ── ROC Curve ────────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[0, 0])
    _setup_ax(ax_roc, "ROC Curve", "FPR", "TPR")
    fpr = results["roc_fpr"]
    tpr = results["roc_tpr"]
    auc = results["auc_roc"]
    opt = results["optimal_threshold"]
    thresh = results["roc_thresholds"]
    idx = np.argmin(np.abs(thresh - opt))
    ax_roc.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["accent1"])
    ax_roc.plot(fpr, tpr, color=PALETTE["accent1"], lw=2, label=f"AUC={auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], "--", color=PALETTE["subtext"], lw=0.8)
    ax_roc.scatter(fpr[idx], tpr[idx], s=80, color=PALETTE["accent3"], zorder=5)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.legend(
        loc="lower right",
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=7.5,
    )

    # ── PR Curve ─────────────────────────────────────────────────────────
    ax_pr = fig.add_subplot(gs[0, 1])
    _setup_ax(ax_pr, "Precision-Recall Curve", "Recall", "Precision")
    prec = results["pr_precision"]
    rec = results["pr_recall"]
    auprc = results["auprc"]
    prev = results["y_true"].mean()
    ax_pr.fill_between(rec, prec, alpha=0.12, color=PALETTE["accent2"])
    ax_pr.plot(rec, prec, color=PALETTE["accent2"], lw=2, label=f"AUPRC={auprc:.4f}")
    ax_pr.axhline(prev, color=PALETTE["subtext"], lw=0.8, linestyle="--")
    ax_pr.scatter(
        results["recall"],
        results["precision"],
        s=80,
        color=PALETTE["accent3"],
        zorder=5,
    )
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.legend(
        loc="lower left",
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=7.5,
    )

    # ── Confusion Matrix ──────────────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[0, 2])
    _setup_ax(ax_cm, "Confusion Matrix", "Predicted", "True")
    cm = results["confusion_matrix"]
    cmap = LinearSegmentedColormap.from_list(
        "med", [PALETTE["card"], PALETTE["accent1"]]
    )
    ax_cm.imshow(cm, cmap=cmap, aspect="auto")
    names = results.get("class_names", ["Neg", "Pos"])
    labels_2x2 = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax_cm.text(
                j,
                i,
                f"{labels_2x2[i][j]}\n{cm[i, j]}",
                ha="center",
                va="center",
                color=PALETTE["text"],
                fontsize=11,
                fontweight="bold",
            )
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(names, color=PALETTE["text"], fontsize=8)
    ax_cm.set_yticklabels(
        names, color=PALETTE["text"], fontsize=8, rotation=90, va="center"
    )

    # ── Metrics Scorecard ─────────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[1, 0])
    ax_sc.axis("off")
    ax_sc.set_facecolor(PALETTE["card"])
    ax_sc.set_title(
        "Key Metrics", color=PALETTE["text"], fontsize=9, fontweight="bold", pad=6
    )
    scorecard = [
        ("Accuracy", results["accuracy"], PALETTE["accent1"]),
        ("AUC-ROC", results["auc_roc"], PALETTE["accent1"]),
        ("Sensitivity", results["sensitivity"], PALETTE["positive"]),
        ("Specificity", results["specificity"], PALETTE["negative"]),
        ("F1 Score", results["f1"], PALETTE["accent3"]),
        ("Precision", results["precision"], PALETTE["accent2"]),
        ("MCC", results["mcc"], "#A78BFA"),
        ("AUPRC", results["auprc"], PALETTE["accent2"]),
    ]
    for i, (name, val, color) in enumerate(scorecard):
        y = 0.92 - i * 0.115
        bw = max(0.0, min(val, 1.0)) * 0.55
        ax_sc.barh(
            [y],
            [bw],
            height=0.08,
            left=0.35,
            color=color,
            alpha=0.7,
            transform=ax_sc.transAxes,
        )
        ax_sc.text(
            0.02,
            y,
            name,
            color=PALETTE["subtext"],
            fontsize=8,
            va="center",
            transform=ax_sc.transAxes,
        )
        ax_sc.text(
            0.93,
            y,
            f"{val:.4f}",
            color=color,
            fontsize=8.5,
            fontweight="bold",
            va="center",
            ha="right",
            transform=ax_sc.transAxes,
        )
    ax_sc.set_facecolor(PALETTE["card"])

    # ── Threshold Analysis ────────────────────────────────────────────────
    ax_th = fig.add_subplot(gs[1, 1:])
    _setup_ax(ax_th, "Metric vs. Decision Threshold", "Threshold", "Value")
    y_true_t = results["y_true"]
    y_prob_t = results["y_prob"]
    thresholds = np.linspace(0.01, 0.99, 150)

    def _f1(t):
        yp = (y_prob_t >= t).astype(int)
        from sklearn.metrics import f1_score as _fs

        return _fs(y_true_t, yp, zero_division=0)

    def _sens(t):
        yp = (y_prob_t >= t).astype(int)
        tp = ((yp == 1) & (y_true_t == 1)).sum()
        fn = ((yp == 0) & (y_true_t == 1)).sum()
        return tp / (tp + fn + 1e-8)

    def _spec(t):
        yp = (y_prob_t >= t).astype(int)
        tn = ((yp == 0) & (y_true_t == 0)).sum()
        fp = ((yp == 1) & (y_true_t == 0)).sum()
        return tn / (tn + fp + 1e-8)

    f1_vals = [_f1(t) for t in thresholds]
    sens_vals = [_sens(t) for t in thresholds]
    spec_vals = [_spec(t) for t in thresholds]

    ax_th.plot(thresholds, f1_vals, color=PALETTE["accent1"], lw=2, label="F1 Score")
    ax_th.plot(
        thresholds, sens_vals, color=PALETTE["positive"], lw=2, label="Sensitivity"
    )
    ax_th.plot(
        thresholds, spec_vals, color=PALETTE["negative"], lw=2, label="Specificity"
    )
    ax_th.axvline(
        opt,
        color=PALETTE["accent3"],
        lw=1.5,
        linestyle="--",
        label=f"Youden's J (t={opt:.3f})",
    )
    ax_th.set_xlim(0, 1)
    ax_th.set_ylim(0, 1.05)
    ax_th.legend(
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=8,
        loc="lower center",
        ncol=4,
    )

    # Footer
    fig.text(
        0.5,
        0.01,
        f"Threshold={results['applied_threshold']:.3f} | "
        f"TP={results['tp']}  TN={results['tn']}  "
        f"FP={results['fp']}  FN={results['fn']}  | "
        f"N={len(results['y_true'])}",
        ha="center",
        color=PALETTE["subtext"],
        fontsize=8,
    )

    _save(fig, output_path, dpi=140)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def generate_all_plots(
    results: dict[str, Any],
    output_dir: str,
    model_name: str = "EfficientNet-B0",
    split: str = "test",
) -> dict[str, str]:
    """
    Generate all 8 evaluation plots and return a dict of {name: path}.

    Args:
        results:    Full dict from ModelEvaluator.evaluate().
        output_dir: Directory to save all plots.
        model_name: Label for titles.
        split:      Split name for titles.
    Returns:
        Dict mapping plot name to file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Generating evaluation plots → {output_dir}/")

    saved = {}

    plots = [  # type: ignore
        ("metrics_dashboard", plot_metrics_dashboard, {}),
        ("roc_curve", plot_roc_curve, {}),
        ("pr_curve", plot_pr_curve, {}),
        ("confusion_matrix", plot_confusion_matrix, {}),
        ("classification_report", plot_classification_report, {}),
        ("threshold_analysis", plot_threshold_analysis, {}),
        ("probability_distribution", plot_probability_distribution, {}),
    ]

    for name, fn, kwargs in plots:
        path = os.path.join(output_dir, f"{name}.png")
        fn(results, path, **kwargs)
        saved[name] = path

    # Full report last
    report_path = os.path.join(output_dir, "full_report.png")
    plot_evaluation_report(
        results, report_path, model_name=model_name, split=split.capitalize()
    )
    saved["full_report"] = report_path

    print(f"\n  All {len(saved)} plots saved to: {os.path.abspath(output_dir)}/")
    return saved
