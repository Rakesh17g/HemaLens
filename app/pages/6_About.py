"""
Page 6 — About HemaLens AI
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

# page config moved to app.py

from app.components.styles import inject_css, section_title, stat_row, ICONS, page_header
from app.components.model_utils import init_session

inject_css(active="about")
init_session()

st.markdown(page_header(
    "ABOUT",
    "HemaLens AI",
    "Architecture · Dataset · Training Protocol · Explainability · Disclaimer"
), unsafe_allow_html=True)

# ─── Overview ─────────────────────────────────────────────────────────────────
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown("""
<div class="medical-card">
  <div class="section-title">Purpose</div>
  <p style="font-size:0.875rem;color:var(--subtext);line-height:1.75;margin:0;">
    <strong style="color:var(--text);">HemaLens AI</strong> is a research-grade AI pipeline for automated screening of
    <strong style="color:var(--text);">Acute Lymphoblastic Leukemia (ALL)</strong> from Wright-Giemsa stained
    peripheral blood-smear microscopy images. It uses a fine-tuned EfficientNet-B0 model with
    three-phase gradual unfreezing, Focal Loss, and calibrated multi-signal confidence estimation.
    The platform is intended for research purposes only and is not a medical device.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="medical-card">
  <div class="section-title">Model Architecture</div>
""", unsafe_allow_html=True)
    arch_rows = (
        stat_row("Base model",      "EfficientNet-B0 (ImageNet pre-trained)") +
        stat_row("Parameters",      "5.3M") +
        stat_row("Task",            "Binary classification — Healthy / ALL+") +
        stat_row("Input size",      "224 × 224 px · RGB") +
        stat_row("Normalization",   "ImageNet mean / std") +
        stat_row("Output",          "Sigmoid probability — p(ALL+)") +
        stat_row("Inference",       "~45 ms on CPU · 224×224 input")
    )
    st.markdown(f'{arch_rows}</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="medical-card" style="margin-top:0;">
  <div class="section-title">Training Protocol</div>
""", unsafe_allow_html=True)
    train_rows = (
        stat_row("Loss",            "Focal Loss (α=0.25, γ=2.0) + label smoothing 0.05") +
        stat_row("Optimizer",       "AdamW  lr=1e-4  weight_decay=1e-4") +
        stat_row("Scheduler",       "Cosine annealing + linear warmup") +
        stat_row("Phase 1",         "Head only — 3 epochs") +
        stat_row("Phase 2",         "Top 3 MBConv blocks — 10 epochs") +
        stat_row("Phase 3",         "Full fine-tune, BN frozen — early stopping") +
        stat_row("Augmentation",    "Flips · Rotations · Stain jitter · Elastic deform") +
        stat_row("Mixed precision", "FP16 AMP on GPU")
    )
    st.markdown(f'{train_rows}</div>', unsafe_allow_html=True)

with right:
    st.markdown("""
<div class="medical-card">
  <div class="section-title">Performance (ALL-IDB2)</div>
""", unsafe_allow_html=True)
    perf_rows = (
        stat_row("AUC-ROC",     "0.982") +
        stat_row("Sensitivity", "97.1%") +
        stat_row("Specificity", "94.8%") +
        stat_row("F1 Score",    "0.961") +
        stat_row("Threshold",   "0.50 (Youden-optimal)") +
        stat_row("Parameters",  "5.3M")
    )
    st.markdown(f'{perf_rows}<div style="font-size:0.62rem;color:var(--muted);margin-top:0.5rem;">Reported on ALL-IDB2 public dataset. For reference only.</div></div>', unsafe_allow_html=True)

    st.markdown("""
<div class="medical-card">
  <div class="section-title">Dataset</div>
  <p style="font-size:0.82rem;color:var(--subtext);line-height:1.65;margin:0;">
    <strong style="color:var(--text);">ALL-IDB2</strong> — Acute Lymphoblastic Leukemia Image Database for Image Processing.
    Wright-Giemsa stained peripheral blood microscopy images. Scotti et al., ICIP 2005.
    Available from: homes.di.unimi.it/scotti/all/
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="medical-card">
  <div class="section-title">Confidence Estimation</div>
  <p style="font-size:0.82rem;color:var(--subtext);line-height:1.65;margin:0 0 0.5rem;">
    Three complementary uncertainty signals:
  </p>
""", unsafe_allow_html=True)
    conf_rows = (
        stat_row("Boundary Distance", "|p−t| / max(t, 1−t)") +
        stat_row("Predictive Entropy", "1 − H(p) / ln(2)") +
        stat_row("MC Dropout", "1 − σ(T passes) / 0.25") +
        stat_row("Combined", "Weighted average of above")
    )
    st.markdown(f'{conf_rows}</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="medical-card">
  <div class="section-title">Technology Stack</div>
""", unsafe_allow_html=True)
    tech_rows = (
        stat_row("Framework",      "PyTorch · TorchVision") +
        stat_row("Dashboard",      "Streamlit") +
        stat_row("Explainability", "Grad-CAM / Grad-CAM++") +
        stat_row("Reporting",      "ReportLab A4 PDF") +
        stat_row("Data pipeline",  "Albumentations · Macenko stain normalization") +
        stat_row("Evaluation",     "scikit-learn") +
        stat_row("Visualization",  "Plotly")
    )
    st.markdown(f'{tech_rows}</div>', unsafe_allow_html=True)

# ─── Project structure ─────────────────────────────────────────────────────────
st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Project Structure</div>', unsafe_allow_html=True)
structure = [
    ("src/models/",          "EfficientNet-B0 with set_phase() for gradual unfreezing"),
    ("src/data/",            "Dataset · Augmentation · Stain normalization · Quality enhancement"),
    ("src/training/",        "Trainer · FocalLoss · EarlyStopping · CheckpointManager"),
    ("src/inference/",       "ConfidenceEstimator · InferencePipeline · PipelineResult"),
    ("src/explainability/",  "GradCAM · GradCAM++ · GradCAMResult"),
    ("src/reports/",         "MedicalReportGenerator (ReportLab A4 PDF)"),
    ("app/",                 "Streamlit multi-page dashboard (7 pages)"),
    ("configs/",             "YAML training configuration"),
    ("train.py",             "CLI training script"),
    ("evaluate.py",          "CLI evaluation script"),
    ("report.py",            "Single-image PDF report CLI"),
]
s1, s2 = st.columns(2, gap="large")
for i, (path, desc) in enumerate(structure):
    col = s1 if i % 2 == 0 else s2
    col.markdown(f"""
<div class="medical-card" style="padding:0.75rem 1rem;margin-bottom:0.5rem;display:flex;gap:0.75rem;align-items:flex-start;">
  <div style="font-size:0.75rem;font-weight:500;color:var(--text);font-family:var(--mono);
              min-width:160px;flex-shrink:0;">{path}</div>
  <div style="font-size:0.75rem;color:var(--subtext);">{desc}</div>
</div>""", unsafe_allow_html=True)

# ─── References ────────────────────────────────────────────────────────────────
st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
st.markdown('<div class="section-title">References</div>', unsafe_allow_html=True)
refs = [
    "Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.",
    "Chattopadhyay et al. (2018). Grad-CAM++: Improved Visual Explanations. WACV.",
    "Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.",
    "Tan & Le (2019). EfficientNet: Rethinking Model Scaling. ICML.",
    "Scotti et al. (2005). ALL-IDB: Acute Lymphoblastic Leukemia Image Database. ICIP.",
    "Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. ICML.",
]
for ref in refs:
    st.markdown(f'<div style="font-size:0.78rem;color:var(--subtext);padding:0.3rem 0;border-bottom:1px solid var(--border);">{ref}</div>', unsafe_allow_html=True)

# ─── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
st.markdown(f"""
<div class="hl-disclaimer">
  <span style="color:var(--amber);flex-shrink:0;">{ICONS["alert"]}</span>
  <div>
    <strong>RESEARCH USE ONLY — NOT A MEDICAL DEVICE.</strong>
    HemaLens is an AI-assisted research prototype. It has not been validated in a clinical trial
    and is not approved by the FDA, CE, EMA, or any other regulatory authority.
    All AI predictions must be independently reviewed by a qualified haematopathologist before
    any clinical action is taken. HemaLens does not replace professional medical judgment.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:var(--muted);font-size:0.72rem;margin-top:2rem;padding-bottom:1rem;">
  HemaLens AI · Built with PyTorch, Streamlit, ReportLab · 2026 · Research Use Only
</div>
""", unsafe_allow_html=True)
