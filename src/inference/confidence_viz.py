"""
Confidence Visualization
=========================
Matplotlib plots that display confidence alongside predictions.

Plots
-----
1. confidence_gauge()         — Single-image radial gauge (probability + confidence)
2. confidence_vs_probability() — Scatter plot comparing p vs c for a batch
3. risk_distribution()        — Pie/bar chart of risk level distribution
4. confidence_histogram()     — Distribution of confidence scores by class
5. confidence_reliability()   — Calibration-style: mean confidence vs mean accuracy
6. plot_confidence_report()   — Single-page composite for one sample
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
from matplotlib import gridspec

# Re-use the design system from the main visualizer
PALETTE = {
    "bg": "#0B1120",
    "card": "#131D2E",
    "surface": "#1A2744",
    "border": "#243354",
    "accent1": "#6366F1",  # indigo
    "accent2": "#22D3EE",  # cyan
    "accent3": "#F59E0B",  # amber
    "positive": "#EF4444",  # red  — ALL+
    "negative": "#22C55E",  # green — Healthy
    "low": "#22C55E",  # green — LOW risk
    "medium": "#F59E0B",  # amber — MEDIUM risk
    "high": "#EF4444",  # red   — HIGH risk
    "text": "#F1F5F9",
    "subtext": "#94A3B8",
    "grid": "#1E3A5F",
}

RISK_COLORS = {
    "LOW": PALETTE["low"],
    "MEDIUM": PALETTE["medium"],
    "HIGH": PALETTE["high"],
}


def _fig_base(title: str = "") -> plt.Figure:
    fig = plt.figure()
    fig.patch.set_facecolor(PALETTE["bg"])
    if title:
        fig.suptitle(
            title, color=PALETTE["text"], fontsize=12, fontweight="bold", y=0.97
        )
    return fig


def _ax_base(
    ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "", grid: bool = True
) -> None:
    ax.set_facecolor(PALETTE["card"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["subtext"])
    ax.yaxis.label.set_color(PALETTE["subtext"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=9, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if grid:
        ax.grid(color=PALETTE["grid"], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str, dpi: int = 130) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"    Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Single-image gauge
# ─────────────────────────────────────────────────────────────────────────────


def plot_confidence_gauge(
    result: Any,  # ConfidenceResult
    output_path: str,
) -> None:
    """
    Dual radial gauge showing prediction probability + confidence score
    for a single sample.  The risk-level colour band fills the outer ring.
    """
    prob = result.probability
    conf = result.confidence
    risk = result.risk_level
    pred = result.prediction
    r_col = RISK_COLORS[risk]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "Confidence Estimation — Single Sample",
        color=PALETTE["text"],
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    def _gauge(ax, value, color, title, subtitle):
        ax.set_facecolor(PALETTE["surface"])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Background arc
        theta_bg = np.linspace(-np.pi * 1.1, np.pi * 0.1, 200)
        r = 0.37
        cx, cy = 0.5, 0.48
        ax.plot(
            cx + r * np.cos(theta_bg),
            cy + r * np.sin(theta_bg),
            color=PALETTE["border"],
            linewidth=10,
            solid_capstyle="round",
        )

        # Value arc (from left bottom → around top → right bottom)
        sweep = value * (np.pi * 1.2)  # 0 → 1.2π for value 0 → 1
        theta_v = np.linspace(-np.pi * 1.1, -np.pi * 1.1 + sweep, 200)
        if value > 0.005:
            ax.plot(
                cx + r * np.cos(theta_v),
                cy + r * np.sin(theta_v),
                color=color,
                linewidth=10,
                solid_capstyle="round",
            )

        # Needle dot
        angle = -np.pi * 1.1 + sweep
        ax.scatter(
            [cx + r * np.cos(angle)],
            [cy + r * np.sin(angle)],
            s=140,
            color=color,
            zorder=6,
            edgecolors=PALETTE["bg"],
            linewidths=2,
        )

        # Value text
        ax.text(
            cx,
            cy,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=color,
            fontsize=26,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            cx,
            cy - 0.14,
            f"{value * 100:.1f}%",
            ha="center",
            va="center",
            color=PALETTE["subtext"],
            fontsize=11,
            transform=ax.transAxes,
        )

        # Title and subtitle
        ax.text(
            cx,
            0.90,
            title,
            ha="center",
            va="top",
            color=PALETTE["text"],
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            cx,
            0.10,
            subtitle,
            ha="center",
            va="bottom",
            color=PALETTE["subtext"],
            fontsize=8.5,
            transform=ax.transAxes,
            wrap=True,
        )

    # Left: Probability gauge
    _gauge(
        axes[0],
        prob,
        color=PALETTE["positive"] if pred == "ALL+" else PALETTE["negative"],
        title="Prediction Probability",
        subtitle=f"Class → {pred}  (threshold={result.threshold:.2f})",
    )

    # Right: Confidence gauge
    _gauge(
        axes[1],
        conf,
        color=r_col,
        title="Confidence Score",
        subtitle=f"Risk Level → {risk}",
    )

    # Risk badge
    risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
    fig.text(
        0.5,
        0.01,
        f"{risk_emoji} {risk} RISK  |  "
        f"Boundary={result.boundary_score:.3f}  "
        f"Entropy={result.entropy_score:.3f}  "
        + (
            f"MC-Dropout={result.mcdrop_score:.3f}  (T={result.mcdrop_passes})"
            if result.mcdrop_passes > 0
            else "MC-Dropout=disabled"
        ),
        ha="center",
        color=r_col,
        fontsize=9,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # type: ignore
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confidence vs Probability scatter (batch)
# ─────────────────────────────────────────────────────────────────────────────


def plot_confidence_vs_probability(
    results: list[Any],  # List[ConfidenceResult]
    output_path: str,
) -> None:
    """
    Scatter plot: x = probability, y = confidence.
    Points are coloured by risk level.  Quadrant annotations explain
    the four regimes (high/low probability × high/low confidence).
    """
    probs = np.array([r.probability for r in results])
    confs = np.array([r.confidence for r in results])
    risks = [r.risk_level for r in results]
    [r.prediction for r in results]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    _ax_base(
        ax,
        "Confidence vs Prediction Probability",
        "Prediction Probability  p",
        "Confidence Score  c",
    )

    # Plot per risk level
    for risk, col in RISK_COLORS.items():
        mask = [r == risk for r in risks]
        if not any(mask):
            continue
        ax.scatter(
            probs[mask],
            confs[mask],
            c=col,
            s=55,
            alpha=0.75,
            edgecolors=PALETTE["bg"],
            linewidths=0.4,
            label=f"{risk} risk (n={sum(mask)})",
            zorder=3,
        )

    # Threshold lines
    ax.axvline(
        0.5, color=PALETTE["accent3"], lw=1.2, linestyle="--", alpha=0.5, label="p=0.50"
    )
    ax.axhline(
        0.80,
        color=PALETTE["low"],
        lw=1.0,
        linestyle=":",
        alpha=0.6,
        label="c=0.80 (LOW/MEDIUM)",
    )
    ax.axhline(
        0.55,
        color=PALETTE["medium"],
        lw=1.0,
        linestyle=":",
        alpha=0.6,
        label="c=0.55 (MEDIUM/HIGH)",
    )

    # Quadrant labels
    quad_texts = [
        (0.15, 0.90, "Healthy\n(confident)"),
        (0.78, 0.90, "ALL+\n(confident)"),
        (0.78, 0.30, "ALL+\n(uncertain)"),
        (0.15, 0.30, "Healthy\n(uncertain)"),
    ]
    for tx, ty, label in quad_texts:
        ax.text(
            tx,
            ty,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=PALETTE["subtext"],
            fontsize=7.5,
            alpha=0.7,
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": PALETTE["surface"],
                "ec": PALETTE["border"],
                "alpha": 0.7,
            },
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(
        loc="lower center",
        ncol=3,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
    )

    plt.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Risk distribution
# ─────────────────────────────────────────────────────────────────────────────


def plot_risk_distribution(
    results: list[Any],  # List[ConfidenceResult]
    output_path: str,
) -> None:
    """
    Side-by-side bar chart + pie chart of risk level distribution.
    Broken down by predicted class (ALL+ vs Healthy).
    """
    risks = [r.risk_level for r in results]
    preds = [r.prediction for r in results]

    levels = ["LOW", "MEDIUM", "HIGH"]
    colors = [RISK_COLORS[l] for l in levels]

    # By class
    cls_counts: dict[str, dict[str, int]] = {}
    for pred in ["ALL+", "Healthy"]:
        cls_counts[pred] = {
            lv: sum(1 for r, p in zip(risks, preds) if r == lv and p == pred)
            for lv in levels
        }

    total_counts = [risks.count(lv) for lv in levels]

    fig = plt.figure(figsize=(13, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "Risk Level Distribution",
        color=PALETTE["text"],
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        1, 3, wspace=0.38, left=0.06, right=0.97, top=0.88, bottom=0.12
    )

    # ── Bar chart: total ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    _ax_base(ax1, "Overall Distribution", "Risk Level", "Count", grid=False)
    bars = ax1.bar(
        levels,
        total_counts,
        color=colors,
        edgecolor=PALETTE["bg"],
        linewidth=0.8,
        width=0.55,
    )
    for bar, count in zip(bars, total_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{count}\n({100 * count / max(len(results), 1):.1f}%)",
            ha="center",
            va="bottom",
            color=PALETTE["text"],
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels(levels, color=PALETTE["text"], fontsize=9)

    # ── Grouped bar: by class ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    _ax_base(ax2, "By Predicted Class", "Risk Level", "Count", grid=False)
    x = np.arange(len(levels))
    bw = 0.34
    for i, (pred_cls, mc) in enumerate(
        [("ALL+", PALETTE["positive"]), ("Healthy", PALETTE["negative"])]
    ):
        vals = [cls_counts[pred_cls][lv] for lv in levels]
        ax2.bar(
            x + (i - 0.5) * bw,
            vals,
            bw,
            color=mc,
            alpha=0.80,
            edgecolor=PALETTE["bg"],
            linewidth=0.6,
            label=pred_cls,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels, color=PALETTE["text"], fontsize=9)
    ax2.legend(
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
    )

    # ── Pie chart ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(PALETTE["card"])
    ax3.set_title(
        "Risk Share", color=PALETTE["text"], fontsize=9, fontweight="bold", pad=6
    )
    non_zero = [
        (c, lv, col) for c, lv, col in zip(total_counts, levels, colors) if c > 0
    ]
    if non_zero:
        counts_nz, labels_nz, colors_nz = zip(*non_zero)
        _wedges, texts, autotexts = ax3.pie(
            counts_nz,
            labels=labels_nz,
            colors=colors_nz,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.78,
            wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 1.5},
        )
        for t in texts:
            t.set_color(PALETTE["text"])
            t.set_fontsize(9)
        for at in autotexts:
            at.set_color(PALETTE["bg"])
            at.set_fontsize(8)
            at.set_fontweight("bold")
    ax3.axis("equal")

    plt.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Confidence histogram by class
# ─────────────────────────────────────────────────────────────────────────────


def plot_confidence_histogram(
    results: list[Any],  # List[ConfidenceResult]
    output_path: str,
) -> None:
    """
    Overlapping histograms of confidence scores, split by predicted class.
    Vertical lines mark the LOW/MEDIUM/HIGH risk boundaries.
    """
    confs_pos = [r.confidence for r in results if r.prediction == "ALL+"]
    confs_neg = [r.confidence for r in results if r.prediction == "Healthy"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    _ax_base(
        ax,
        "Confidence Score Distribution by Predicted Class",
        "Confidence Score  c",
        "Count",
    )

    bins = np.linspace(0, 1, 31)
    if confs_neg:
        ax.hist(
            confs_neg,
            bins=bins,  # type: ignore
            color=PALETTE["negative"],
            alpha=0.72,
            label=f"Healthy  (n={len(confs_neg)})",
            edgecolor=PALETTE["bg"],
            linewidth=0.4,
        )
    if confs_pos:
        ax.hist(
            confs_pos,
            bins=bins,  # type: ignore
            color=PALETTE["positive"],
            alpha=0.72,
            label=f"ALL+     (n={len(confs_pos)})",
            edgecolor=PALETTE["bg"],
            linewidth=0.4,
        )

    # Risk boundary lines
    for boundary, label, col in [
        (0.80, "LOW / MEDIUM boundary", PALETTE["low"]),
        (0.55, "MEDIUM / HIGH boundary", PALETTE["medium"]),
    ]:
        ax.axvline(boundary, color=col, linewidth=2, linestyle="--", alpha=0.85)
        ax.text(
            boundary + 0.01,
            ax.get_ylim()[1] * 0.97,
            label,
            color=col,
            fontsize=7.5,
            va="top",
        )

    # Risk zone shading
    ax.axvspan(0, 0.55, alpha=0.07, color=PALETTE["high"], label="HIGH risk zone")
    ax.axvspan(
        0.55, 0.80, alpha=0.07, color=PALETTE["medium"], label="MEDIUM risk zone"
    )
    ax.axvspan(0.80, 1.00, alpha=0.07, color=PALETTE["low"], label="LOW risk zone")

    ax.set_xlim(0, 1)
    ax.legend(
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
        ncol=2,
        loc="upper left",
    )

    plt.tight_layout()
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Confidence reliability (calibration-style)
# ─────────────────────────────────────────────────────────────────────────────


def plot_confidence_reliability(
    results: list[Any],  # List[ConfidenceResult]
    y_true: np.ndarray,  # (N,) ground truth labels {0,1}
    output_path: str,
    n_bins: int = 10,
) -> None:
    """
    Reliability diagram: mean confidence score vs fraction correct per bin.

    A well-calibrated confidence estimator should have the points fall
    along the diagonal (high confidence → high accuracy).

    Args
    ----
    results : List of ConfidenceResult (in the same order as y_true).
    y_true  : Ground-truth binary labels.
    n_bins  : Number of confidence bins.
    """
    confs = np.array([r.confidence for r in results])
    preds = np.array([1 if r.prediction == "ALL+" else 0 for r in results])
    correct = (preds == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_conf_per_bin: list[float] = []
    mean_acc_per_bin: list[float] = []
    bin_counts = []

    for lo, hi in itertools.pairwise(bin_edges):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            mean_conf_per_bin.append(None)  # type: ignore
            mean_acc_per_bin.append(None)  # type: ignore
            bin_counts.append(0)
        else:
            mean_conf_per_bin.append(float(confs[mask].mean()))
            mean_acc_per_bin.append(float(correct[mask].mean()))
            bin_counts.append(int(mask.sum()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "Confidence Reliability Diagram",
        color=PALETTE["text"],
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # ── Left: reliability diagram ─────────────────────────────────────────
    _ax_base(
        ax1,
        "Mean Confidence vs Accuracy per Bin",
        "Mean Confidence in Bin",
        "Fraction Correct",
    )

    valid_x = [x for x in mean_conf_per_bin if x is not None]
    valid_y = [y for y in mean_acc_per_bin if y is not None]

    # Perfect calibration diagonal
    ax1.plot(
        [0, 1],
        [0, 1],
        "--",
        color=PALETTE["subtext"],
        linewidth=1.2,
        label="Perfect calibration",
        alpha=0.6,
    )
    # Gap shading
    for xv, yv in zip(valid_x, valid_y):
        ax1.fill_between([xv], [xv], [yv], color=PALETTE["accent3"], alpha=0.25)
    # Reliability plot
    ax1.plot(
        valid_x,
        valid_y,
        "o-",
        color=PALETTE["accent1"],
        linewidth=2,
        markersize=7,
        label="Model reliability",
        zorder=4,
    )

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["border"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
    )

    # ── Right: bin count bar chart ────────────────────────────────────────
    _ax_base(ax2, "Samples per Confidence Bin", "Confidence Bin", "Count")
    bar_colors = [
        PALETTE["low"]
        if c >= 0.80
        else PALETTE["medium"]
        if c >= 0.55
        else PALETTE["high"]
        for c in bin_centers
    ]
    ax2.bar(
        bin_centers,
        bin_counts,
        width=0.09,
        color=bar_colors,
        edgecolor=PALETTE["bg"],
        linewidth=0.5,
    )
    ax2.set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Single-sample full report (composite)
# ─────────────────────────────────────────────────────────────────────────────


def plot_confidence_report(
    result: Any,  # ConfidenceResult
    output_path: str,
    image_array: np.ndarray | None = None,  # (H,W,3) uint8 — optional
) -> None:
    """
    Single-page composite: gauge + breakdown bars + explanation text.
    Optionally includes the input image in the top-left panel.

    Args
    ----
    result       : ConfidenceResult for the sample.
    output_path  : Where to save the PNG.
    image_array  : Optional (H, W, 3) uint8 image for display.
    """
    risk = result.risk_level
    r_col = RISK_COLORS[risk]
    prob = result.probability
    conf = result.confidence
    pred = result.prediction
    emoji_map = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}

    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        f"ALL Detection — Confidence Report  [{pred}]",
        color=PALETTE["text"],
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 4, hspace=0.50, wspace=0.40, top=0.92, bottom=0.08, left=0.05, right=0.97
    )

    # ── Panel 0: Image (if provided) or blank ─────────────────────────────
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.set_facecolor(PALETTE["surface"])
    for spine in ax_img.spines.values():
        spine.set_edgecolor(r_col)
        spine.set_linewidth(2.5)
    if image_array is not None:
        ax_img.imshow(image_array, aspect="auto")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
    else:
        ax_img.text(
            0.5,
            0.5,
            "Cell Image\n(not provided)",
            ha="center",
            va="center",
            color=PALETTE["subtext"],
            fontsize=10,
            transform=ax_img.transAxes,
        )
        ax_img.axis("off")
    ax_img.set_title(
        f"{pred}  |  {emoji_map[risk]} {risk} RISK",
        color=r_col,
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    # ── Panel 1: Probability gauge ────────────────────────────────────────
    ax_prob = fig.add_subplot(gs[0, 1])
    ax_prob.set_facecolor(PALETTE["surface"])
    for spine in ax_prob.spines.values():
        spine.set_edgecolor(
            PALETTE["positive"] if pred == "ALL+" else PALETTE["negative"]
        )
        spine.set_linewidth(1.8)
    p_col = PALETTE["positive"] if pred == "ALL+" else PALETTE["negative"]
    theta = np.linspace(0, 2 * np.pi * prob, 200)
    theta_bg = np.linspace(0, 2 * np.pi, 200)
    r = 0.35
    ax_prob.plot(
        r * np.cos(theta_bg) + 0.5,
        r * np.sin(theta_bg) + 0.52,
        color=PALETTE["border"],
        linewidth=8,
        solid_capstyle="round",
    )
    if prob > 0.005:
        ax_prob.plot(
            r * np.cos(theta) + 0.5,
            r * np.sin(theta) + 0.52,
            color=p_col,
            linewidth=8,
            solid_capstyle="round",
        )
    ax_prob.text(
        0.5,
        0.50,
        f"{prob:.3f}",
        ha="center",
        va="center",
        color=p_col,
        fontsize=20,
        fontweight="bold",
        transform=ax_prob.transAxes,
    )
    ax_prob.text(
        0.5,
        0.18,
        "Prediction Probability",
        ha="center",
        color=PALETTE["subtext"],
        fontsize=8,
        transform=ax_prob.transAxes,
    )
    ax_prob.set_xlim(0, 1)
    ax_prob.set_ylim(0, 1)
    ax_prob.axis("off")

    # ── Panel 2: Confidence gauge ─────────────────────────────────────────
    ax_conf = fig.add_subplot(gs[0, 2])
    ax_conf.set_facecolor(PALETTE["surface"])
    for spine in ax_conf.spines.values():
        spine.set_edgecolor(r_col)
        spine.set_linewidth(1.8)
    theta_c = np.linspace(0, 2 * np.pi * conf, 200)
    ax_conf.plot(
        r * np.cos(theta_bg) + 0.5,
        r * np.sin(theta_bg) + 0.52,
        color=PALETTE["border"],
        linewidth=8,
        solid_capstyle="round",
    )
    if conf > 0.005:
        ax_conf.plot(
            r * np.cos(theta_c) + 0.5,
            r * np.sin(theta_c) + 0.52,
            color=r_col,
            linewidth=8,
            solid_capstyle="round",
        )
    ax_conf.text(
        0.5,
        0.50,
        f"{conf:.3f}",
        ha="center",
        va="center",
        color=r_col,
        fontsize=20,
        fontweight="bold",
        transform=ax_conf.transAxes,
    )
    ax_conf.text(
        0.5,
        0.18,
        "Confidence Score",
        ha="center",
        color=PALETTE["subtext"],
        fontsize=8,
        transform=ax_conf.transAxes,
    )
    ax_conf.set_xlim(0, 1)
    ax_conf.set_ylim(0, 1)
    ax_conf.axis("off")

    # ── Panel 3: Risk level card ──────────────────────────────────────────
    ax_risk = fig.add_subplot(gs[0, 3])
    ax_risk.set_facecolor(PALETTE["surface"])
    for spine in ax_risk.spines.values():
        spine.set_edgecolor(r_col)
        spine.set_linewidth(2.5)
    ax_risk.axis("off")
    ax_risk.text(
        0.5,
        0.72,
        emoji_map[risk],
        ha="center",
        va="center",
        fontsize=42,
        transform=ax_risk.transAxes,
    )
    ax_risk.text(
        0.5,
        0.48,
        risk + " RISK",
        ha="center",
        va="center",
        color=r_col,
        fontsize=16,
        fontweight="bold",
        transform=ax_risk.transAxes,
    )
    ax_risk.text(
        0.5,
        0.25,
        result.clinical_note,
        ha="center",
        va="center",
        color=PALETTE["subtext"],
        fontsize=7.5,
        transform=ax_risk.transAxes,
        wrap=True,
    )

    # ── Bottom row: confidence breakdown bars ─────────────────────────────
    ax_bd = fig.add_subplot(gs[1, 1:])
    _ax_base(
        ax_bd,
        "Confidence Component Breakdown",
        "Score (0 = uncertain, 1 = certain)",
        "",
        grid=False,
    )

    components = [
        (
            "Boundary Distance",
            result.boundary_score,
            PALETTE["accent1"],
            f"|p−t|/max_dist  =  |{prob:.3f}−{result.threshold:.2f}|  →  {result.boundary_score:.3f}",
        ),
        (
            "Entropy Score",
            result.entropy_score,
            PALETTE["accent2"],
            f"1 − H(p)/ln2  |  H={result.entropy_raw:.4f}  →  {result.entropy_score:.3f}",
        ),
    ]
    if result.mcdrop_passes > 0:
        components.append(
            (
                f"MC Dropout (T={result.mcdrop_passes})",
                result.mcdrop_score,
                PALETTE["accent3"],
                f"1 − σ/0.25  |  σ={result.mcdrop_std:.4f}  mean={result.mcdrop_mean:.3f}  →  {result.mcdrop_score:.3f}",
            )
        )
    components.append(
        (
            "Combined Confidence",
            conf,
            r_col,
            f"Weighted sum → {conf:.3f}  ({conf * 100:.1f}%)  [{risk} RISK]",
        )
    )

    ys = np.arange(len(components)) * 1.2
    ys[-1]
    for y, (name, val, col, annot) in zip(ys, components):
        ax_bd.barh(
            [y],
            [val],
            height=0.7,
            color=col,
            alpha=0.85,
            edgecolor=PALETTE["bg"],
            linewidth=0.5,
        )
        ax_bd.text(
            -0.01,
            y,
            name,
            ha="right",
            va="center",
            color=PALETTE["text"],
            fontsize=8.5,
            fontweight="bold",
        )
        ax_bd.text(
            val + 0.01,
            y,
            annot,
            ha="left",
            va="center",
            color=PALETTE["subtext"],
            fontsize=7.5,
        )

    ax_bd.set_xlim(-0.05, 1.35)
    ax_bd.set_yticks([])
    ax_bd.set_xlim(0, 1.45)
    ax_bd.axvline(1.0, color=PALETTE["border"], lw=0.8, linestyle=":")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # type: ignore
    _save(fig, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def generate_confidence_plots(
    results: list[Any],  # List[ConfidenceResult]
    y_true: np.ndarray,  # (N,) ground truth
    output_dir: str,
    single_result: Any | None = None,  # ConfidenceResult for gauge/report
    image_array: np.ndarray | None = None,
) -> dict[str, str]:
    """
    Generate all confidence visualization plots.

    Args
    ----
    results       : Full list of ConfidenceResults for batch plots.
    y_true        : Ground-truth labels (same order as results).
    output_dir    : Directory to save all PNGs.
    single_result : A single ConfidenceResult for the gauge + report plots.
                    Defaults to the result with median confidence.
    image_array   : Optional image for the single-sample report panel.

    Returns
    -------
    Dict mapping plot name → absolute path.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Generating confidence plots → {os.path.abspath(output_dir)}/")

    # Pick a representative single result if not provided
    if single_result is None and results:
        confs = [r.confidence for r in results]
        median_idx = int(np.argsort(confs)[len(confs) // 2])
        single_result = results[median_idx]

    saved: dict[str, str] = {}

    if single_result:
        p = os.path.join(output_dir, "confidence_gauge.png")
        plot_confidence_gauge(single_result, p)
        saved["confidence_gauge"] = p

        p = os.path.join(output_dir, "confidence_report.png")
        plot_confidence_report(single_result, p, image_array=image_array)
        saved["confidence_report"] = p

    if results:
        p = os.path.join(output_dir, "confidence_vs_probability.png")
        plot_confidence_vs_probability(results, p)
        saved["confidence_vs_probability"] = p

        p = os.path.join(output_dir, "risk_distribution.png")
        plot_risk_distribution(results, p)
        saved["risk_distribution"] = p

        p = os.path.join(output_dir, "confidence_histogram.png")
        plot_confidence_histogram(results, p)
        saved["confidence_histogram"] = p

        if len(y_true) == len(results):
            p = os.path.join(output_dir, "confidence_reliability.png")
            plot_confidence_reliability(results, y_true, p)
            saved["confidence_reliability"] = p

    print(f"\n  {len(saved)} confidence plots saved.")
    return saved
