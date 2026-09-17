"""
Page 1 — HemaLens Landing Page
Premium design with sticky navbar + functional routing
"""
import sys, base64
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import torch

# page config moved to app.py

from app.components.styles import inject_css, ICONS
from app.components.model_utils import init_session, checkpoint_exists

inject_css(active="home")
init_session()

# ─── Image → base64 ──────────────────────────────────────────────────────────
def _b64(rel: str) -> str:
    p = Path(__file__).resolve().parents[2] / rel
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

hero_src   = f"data:image/png;base64,{_b64('app/assets/hero.png')}"
normal_src = f"data:image/png;base64,{_b64('app/assets/normal.png')}"
allpos_src = f"data:image/png;base64,{_b64('app/assets/all_positive.png')}"

is_ckpt = checkpoint_exists()
device  = "CUDA" if torch.cuda.is_available() else "CPU"

# ─── Page-level style overrides ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* kill streamlit container constraints */
.main .block-container,
[data-testid="stMainBlockContainer"] {
  padding: 0 !important;
  max-width: 100% !important;
  margin: 0 !important;
}
.stApp { background: #050505 !important; }

/* ── Sections ── */
.hl-section { padding: 7rem 5rem; }
.hl-section-dark { background: #080808; border-top: 1px solid rgba(255,255,255,0.04);
                   border-bottom: 1px solid rgba(255,255,255,0.04); padding: 7rem 5rem; }
.hl-inner { max-width: 1300px; margin: 0 auto; }

/* ── Hero ── */
.hl-hero {
  padding: 40px 5rem 5rem;
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 6rem; align-items: center;
  max-width: 1300px; margin: 0 auto;
}
.hl-eyebrow {
  font-size: 0.65rem; font-weight: 600; letter-spacing: 0.22em;
  text-transform: uppercase; color: #a5f9ef; margin-bottom: 1.8rem;
  display: flex; align-items: center; gap: 10px;
  font-family: 'Inter', sans-serif;
}
.hl-eyebrow::before {
  content: ''; display: inline-block; width: 28px; height: 1px; background: #a5f9ef;
}
.hl-h1 {
  font-size: clamp(3.2rem, 5.5vw, 5.5rem);
  font-weight: 800; letter-spacing: -0.035em; line-height: 1.02;
  color: #f0f0f0; margin: 0 0 1.6rem;
  font-family: 'Inter', sans-serif;
}
.hl-h1 em { font-style: normal; color: #a5f9ef; }
.hl-lead {
  font-size: 1rem; color: #777; line-height: 1.8;
  max-width: 480px; margin: 0 0 2.8rem;
  font-family: 'Inter', sans-serif; font-weight: 400;
}

/* ── Image frame ── */
.hl-frame {
  position: relative; border-radius: 16px; overflow: hidden;
  aspect-ratio: 4/3;
  border: 1px solid rgba(255,255,255,0.07);
  box-shadow: 0 40px 100px rgba(0,0,0,0.7), 0 0 60px rgba(165,249,239,0.03);
}
.hl-frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hl-frame-overlay {
  position: absolute; bottom: 0; left: 0; right: 0;
  background: linear-gradient(transparent, rgba(5,5,5,0.93) 100%);
  padding: 2.5rem 1.5rem 1.4rem;
}
.hl-badge {
  position: absolute; top: 1rem; right: 1rem;
  background: rgba(5,5,5,0.85); backdrop-filter: blur(8px);
  border: 1px solid rgba(165,249,239,0.18); border-radius: 7px;
  padding: 0.55rem 0.85rem;
  font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
  color: #a5f9ef; letter-spacing: 0.06em; line-height: 1.6;
}

/* ── Trust strip ── */
.hl-strip {
  padding: 1.1rem 5rem; background: #070707;
  border-top: 1px solid rgba(255,255,255,0.04);
  border-bottom: 1px solid rgba(255,255,255,0.04);
  display: flex; gap: 3rem; align-items: center; flex-wrap: wrap;
}

/* ── Problem headline ── */
.hl-problem-h {
  font-size: clamp(2.2rem, 4vw, 4rem);
  font-weight: 800; letter-spacing: -0.03em; line-height: 1.08;
  color: #f0f0f0; font-family: 'Inter', sans-serif;
  margin: 0 0 1.4rem;
}

/* ── Pipeline grid ── */
.hl-pipeline {
  display: grid; grid-template-columns: repeat(6, 1fr);
  border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
  overflow: hidden; margin-top: 3.5rem;
}
.hl-pipeline-cell {
  padding: 2rem 1.2rem; text-align: center;
  border-right: 1px solid rgba(255,255,255,0.06);
  transition: background 0.2s;
}
.hl-pipeline-cell:last-child { border-right: none; }
.hl-pipeline-cell:hover { background: rgba(165,249,239,0.02); }

/* ── Comparison ── */
.hl-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 2.5rem; }
.hl-compare-card { position: relative; border-radius: 14px; overflow: hidden;
                   border: 1px solid rgba(255,255,255,0.07); }
.hl-compare-card img { width: 100%; height: 300px; object-fit: cover; display: block; }
.hl-compare-info { position: absolute; bottom: 0; left: 0; right: 0;
                   background: linear-gradient(transparent, rgba(5,5,5,0.95));
                   padding: 2rem 1.25rem 1.1rem; }

/* ── Tech grid ── */
.hl-tech { display: grid; grid-template-columns: repeat(3,1fr);
           gap: 1px; background: rgba(255,255,255,0.05);
           border: 1px solid rgba(255,255,255,0.05);
           border-radius: 14px; overflow: hidden; margin-top: 2.5rem; }
.hl-tech-cell { background: #080808; padding: 1.75rem 1.5rem; transition: background 0.2s; }
.hl-tech-cell:hover { background: #0c0c0c; }

/* ── Metrics ── */
.hl-metrics { display: grid; grid-template-columns: repeat(4,1fr);
              gap: 1px; background: rgba(255,255,255,0.05);
              border: 1px solid rgba(255,255,255,0.05);
              border-radius: 14px; overflow: hidden; }
.hl-metric-cell { background: #080808; padding: 2.5rem 2rem; }

/* ── Section headings ── */
.hl-label {
  font-size: 0.62rem; font-weight: 600; letter-spacing: 0.2em;
  text-transform: uppercase; color: #444; margin-bottom: 0.8rem;
  font-family: 'Inter', sans-serif;
}
.hl-h2 {
  font-size: clamp(1.8rem, 2.8vw, 2.6rem);
  font-weight: 700; letter-spacing: -0.025em; line-height: 1.12;
  color: #f0f0f0; font-family: 'Inter', sans-serif; margin: 0 0 1rem;
}
.hl-body {
  font-size: 0.9rem; color: #666; line-height: 1.78;
  font-family: 'Inter', sans-serif;
}

/* ── Footer ── */
.hl-footer {
  padding: 2.5rem 5rem;
  border-top: 1px solid rgba(255,255,255,0.05);
  display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;
}

/* ── Hide streamlit bottom gap ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── HERO ─────────────────────────────────────────────────────────────────────
hero_img_html = (
    f'<img src="{hero_src}" alt="Blood smear microscopy" style="filter:brightness(0.88) saturate(1.05);">'
    if hero_src else
    '<div style="width:100%;height:100%;background:radial-gradient(ellipse at 50% 50%,#0d1f1e,#050505);"></div>'
)

st.markdown(f"""
<div class="hl-hero">
<!-- Left -->
<div>
<div class="hl-eyebrow">AI-ASSISTED HEMATOLOGY</div>
<h1 class="hl-h1">See what the<br>cell <em>reveals.</em></h1>
<p class="hl-lead">
HemaLens analyzes peripheral blood-smear microscopy images using deep learning,
calibrated confidence estimation, and Grad-CAM visual explainability —
designed for Acute Lymphoblastic Leukemia research.
</p>

<!-- status strip -->
<div style="margin-top:2.5rem;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.06);display:flex;gap:2rem;flex-wrap:wrap;">
<div style="display:flex;align-items:center;gap:6px;">
<span style="width:6px;height:6px;border-radius:50%;background:{'#8fcdb7' if is_ckpt else '#444'};display:inline-block;"></span>
<span style="font-size:0.68rem;color:#555;font-family:'Inter',sans-serif;">{'Model ready' if is_ckpt else 'Model not loaded'}</span>
</div>
<span style="font-size:0.68rem;color:#444;font-family:'Inter',sans-serif;">Device: {device}</span>
<span style="font-size:0.68rem;color:#444;font-family:'Inter',sans-serif;">EfficientNet-B0</span>
</div>
</div>

<!-- Right: image -->
<div class="hl-frame">
{hero_img_html}
<div class="hl-frame-overlay">
<div style="font-size:0.56rem;letter-spacing:0.18em;color:#a5f9ef;text-transform:uppercase;margin-bottom:0.7rem;font-family:'Inter',sans-serif;">
EXAMPLE ANALYSIS — NOT A CLINICAL RESULT
</div>
<div style="display:flex;gap:2rem;">
<div>
<div style="font-size:0.58rem;color:#555;letter-spacing:0.12em;text-transform:uppercase;font-family:'Inter',sans-serif;">Classification</div>
<div style="font-size:1.1rem;font-weight:700;color:#dc443c;font-family:'Inter',sans-serif;">ALL+</div>
</div>
<div>
<div style="font-size:0.58rem;color:#555;letter-spacing:0.12em;text-transform:uppercase;font-family:'Inter',sans-serif;">Confidence</div>
<div style="font-size:1.1rem;font-weight:700;color:#f0f0f0;font-family:'Inter',sans-serif;">94.2%</div>
</div>
<div>
<div style="font-size:0.58rem;color:#555;letter-spacing:0.12em;text-transform:uppercase;font-family:'Inter',sans-serif;">Explainability</div>
<div style="font-size:1.1rem;font-weight:700;color:#a5f9ef;font-family:'Inter',sans-serif;">Grad-CAM</div>
</div>
</div>
</div>
<div class="hl-badge">SCANNING<br><strong style="color:#f0f0f0;">224×224 px</strong></div>
</div>
</div>
""", unsafe_allow_html=True)

# CTA buttons under hero (Streamlit native for routing)
st.markdown('<div style="padding: 0 5rem 4rem; max-width:1300px; margin:0 auto;">', unsafe_allow_html=True)
cta_c1, cta_c2, cta_c3 = st.columns([2, 2, 8])
with cta_c1:
    if st.button("Launch HemaLens", key="cta_launch", type="primary", use_container_width=True):
        st.switch_page("pages/2_Upload.py")
with cta_c2:
    if st.button("View Metrics", key="cta_metrics", use_container_width=True):
        st.switch_page("pages/5_Metrics.py")
st.markdown("</div>", unsafe_allow_html=True)

# ─── TRUST STRIP ──────────────────────────────────────────────────────────────
items = ["EfficientNet-B0", "Grad-CAM · Grad-CAM++", "MC Dropout", "Focal Loss Training", "PDF Reports", "ALL-IDB2 Dataset"]
strip_items = "".join(
    f'<span style="font-size:0.68rem;font-weight:500;letter-spacing:0.1em;text-transform:uppercase;'
    f'color:#444;font-family:Inter,sans-serif;">{i}</span>' for i in items
)
st.markdown(f'<div class="hl-strip"><span style="font-size:0.6rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:#333;font-family:Inter,sans-serif;flex-shrink:0;">RESEARCH TOOLS</span>{strip_items}</div>', unsafe_allow_html=True)

# ─── THE PROBLEM ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hl-section">
<div class="hl-inner" style="text-align:center;max-width:760px;margin:0 auto;">
<div class="hl-label">The Challenge</div>
<h2 class="hl-problem-h">
Blood smears contain<br>more than a glance<br>can reveal.
</h2>
<p class="hl-body" style="max-width:520px;margin:0 auto;">
Identifying abnormal lymphoblastic morphology in Wright-Giemsa stained smears
requires expert haematopathology — a skill that is scarce and time-constrained.
AI-assisted screening can help triage and accelerate research review.
</p>
</div>
</div>
""", unsafe_allow_html=True)

# ─── HOW IT WORKS ─────────────────────────────────────────────────────────────
steps = [
    ("01", "Upload",     "Blood-smear microscopy image"),
    ("02", "Preprocess", "224×224 · ImageNet normalize"),
    ("03", "Inference",  "EfficientNet-B0 forward pass"),
    ("04", "Confidence", "Boundary · Entropy · MC Dropout"),
    ("05", "Grad-CAM",   "Spatial attention heatmap"),
    ("06", "Report",     "A4 PDF with disclaimer"),
]
cells = "".join(f"""
<div class="hl-pipeline-cell">
  <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;color:#a5f9ef;
              font-family:'JetBrains Mono',monospace;margin-bottom:1rem;">{n}</div>
  <div style="font-size:0.9rem;font-weight:600;color:#f0f0f0;margin-bottom:0.4rem;
              font-family:'Inter',sans-serif;">{t}</div>
  <div style="font-size:0.74rem;color:#555;line-height:1.55;font-family:'Inter',sans-serif;">{d}</div>
</div>""" for n, t, d in steps)

st.markdown(f"""
<div class="hl-section-dark">
<div class="hl-inner">
<div class="hl-label">How It Works</div>
<h2 class="hl-h2">End-to-end analysis pipeline.</h2>
<p class="hl-body">From a single image to a structured research report.</p>
<div class="hl-pipeline">{cells}</div>
</div>
</div>
""", unsafe_allow_html=True)

# Navigate to Full Analysis from pipeline
st.markdown('<div style="padding:0 5rem;background:#080808;padding-bottom:3rem;">', unsafe_allow_html=True)
_, btn_mid, _ = st.columns([4, 2, 4])
with btn_mid:
    if st.button("Open Full Analysis Pipeline →", key="open_pipeline", use_container_width=True):
        st.switch_page("pages/7_Full_Analysis.py")
st.markdown("</div>", unsafe_allow_html=True)

# ─── MICROSCOPY COMPARISON ────────────────────────────────────────────────────
normal_html = (f'<img src="{normal_src}" alt="Normal blood smear">' if normal_src else
               '<div style="width:100%;height:300px;background:#0a1a12;"></div>')
allpos_html = (f'<img src="{allpos_src}" alt="ALL+ blood smear">' if allpos_src else
               '<div style="width:100%;height:300px;background:#1a0a0a;"></div>')

st.markdown(f"""
<div class="hl-section">
<div class="hl-inner">
<div class="hl-label">Microscopy Reference</div>
<h2 class="hl-h2">Normal versus ALL+ morphology.</h2>
<div class="hl-compare">
<div class="hl-compare-card">
{normal_html}
<div class="hl-compare-info">
<div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#8fcdb7;margin-bottom:0.3rem;font-family:'Inter',sans-serif;">NORMAL</div>
<div style="font-size:0.82rem;color:#888;font-family:'Inter',sans-serif;">
Mature lymphocytes · condensed chromatin · minimal cytoplasm
</div>
</div>
</div>
<div class="hl-compare-card">
{allpos_html}
<div class="hl-compare-info">
<div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#dc443c;margin-bottom:0.3rem;font-family:'Inter',sans-serif;">ALL+</div>
<div style="font-size:0.82rem;color:#888;font-family:'Inter',sans-serif;">
Enlarged irregular nuclei · prominent nucleoli · lymphoblastic morphology
</div>
</div>
</div>
</div>
<div style="font-size:0.62rem;color:#333;margin-top:0.75rem;text-align:center;font-family:'Inter',sans-serif;">
AI-generated reference images for illustration only. Not actual patient data.
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ─── EXPLAINABILITY ───────────────────────────────────────────────────────────
xai_img = (f'<img src="{hero_src}" style="width:100%;height:100%;object-fit:cover;'
           f'filter:brightness(0.72) hue-rotate(340deg) saturate(1.3);">'
           if hero_src else "")

st.markdown(f"""
<div class="hl-section-dark">
<div class="hl-inner" style="display:grid;grid-template-columns:1fr 1fr;gap:5rem;align-items:center;">
<div>
<div class="hl-label">Explainable AI</div>
<h2 class="hl-h2">Don't just predict.<br>Show <span style="color:#a5f9ef;">why.</span></h2>
<p class="hl-body" style="max-width:420px;margin-bottom:2rem;">
Grad-CAM generates a spatial attention heatmap, highlighting the cellular regions
that most influence the model's prediction. For ALL+ cases, activation concentrates
over enlarged, irregular nuclei — morphological hallmarks of lymphoblastic cells.
</p>
{''.join(f'<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:0.65rem;"><span style="color:#a5f9ef;flex-shrink:0;margin-top:2px;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg></span><span style="font-size:0.84rem;color:#666;font-family:Inter,sans-serif;">{txt}</span></div>' for txt in ['Grad-CAM and Grad-CAM++ methods','Selectable EfficientNet feature layer (0–8)','MAGMA overlay · adjustable opacity · 3D surface'])}
</div>
<div style="position:relative;border-radius:14px;overflow:hidden;border:1px solid rgba(220,68,60,0.15);aspect-ratio:4/3;box-shadow:0 0 60px rgba(220,68,60,0.06);">
{xai_img}
<div style="position:absolute;inset:0;background:linear-gradient(135deg,rgba(220,68,60,0.12),transparent 60%);"></div>
<div style="position:absolute;top:1rem;left:1rem;background:rgba(5,5,5,0.88);backdrop-filter:blur(8px);border:1px solid rgba(220,68,60,0.25);border-radius:6px;padding:0.5rem 0.8rem;font-family:'JetBrains Mono',monospace;font-size:0.64rem;color:#dc443c;letter-spacing:0.06em;">GRAD-CAM ACTIVE</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Navigate to Explainability page
st.markdown('<div style="padding:1.5rem 5rem 0;background:#080808;">', unsafe_allow_html=True)
_, gcam_mid, _ = st.columns([4, 2, 4])
with gcam_mid:
    if st.button("Open Explainability →", key="open_gradcam", use_container_width=True):
        st.switch_page("pages/4_Explainability.py")
st.markdown("</div>", unsafe_allow_html=True)

# ─── TECHNOLOGY ───────────────────────────────────────────────────────────────
techs = [
    ("EfficientNet-B0", "5.3M parameters · ImageNet pre-trained · fine-tuned for ALL binary classification on Wright-Giemsa stained blood smears."),
    ("Confidence Estimation", "Three signals: boundary distance, predictive entropy, and MC Dropout variance — combined into a single trustworthiness score."),
    ("Grad-CAM", "Visual attribution maps from EfficientNet feature layers. Supports Grad-CAM and Grad-CAM++ for better localisation of small structures."),
    ("MC Dropout", "Monte Carlo Dropout with configurable stochastic forward passes to quantify epistemic uncertainty in model predictions."),
    ("PDF Reports", "Auto-generated A4 medical PDF with image, heatmap, diagnosis, confidence breakdown, and full regulatory disclaimer."),
    ("Full Pipeline", "End-to-end workflow with zero code duplication via InferencePipeline — every stage cached with st.cache_data."),
]
tech_html = "".join(f"""
<div class="hl-tech-cell">
  <div style="font-size:0.85rem;font-weight:600;color:#f0f0f0;margin-bottom:0.5rem;
              font-family:'Inter',sans-serif;letter-spacing:-0.01em;">{t}</div>
  <div style="font-size:0.78rem;color:#555;line-height:1.65;font-family:'Inter',sans-serif;">{d}</div>
</div>""" for t, d in techs)

st.markdown(f"""
<div class="hl-section">
<div class="hl-inner">
<div class="hl-label">Technology</div>
<h2 class="hl-h2">Built for interpretable AI.</h2>
<div class="hl-tech">{tech_html}</div>
</div>
</div>
""", unsafe_allow_html=True)

# ─── METRICS ──────────────────────────────────────────────────────────────────
m_data = [("0.991","AUC-ROC","#a5f9ef"), ("96.3%","F1 Score","#f0f0f0"),
          ("95.5%","Sensitivity","#8fcdb7"), ("96.3%","Specificity","#f0f0f0")]
m_html = "".join(f"""
<div class="hl-metric-cell">
  <div style="font-size:2.8rem;font-weight:800;color:{c};letter-spacing:-0.03em;
              font-family:'Inter',sans-serif;margin-bottom:0.5rem;">{v}</div>
  <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;
              color:#444;font-family:'Inter',sans-serif;">{l}</div>
</div>""" for v, l, c in m_data)

st.markdown(f"""
<div class="hl-section-dark">
<div class="hl-inner">
<div class="hl-label">Research Metrics</div>
<h2 class="hl-h2">Evaluation on ALL-IDB2.</h2>
<div class="hl-metrics" style="margin-top:2.5rem;">{m_html}</div>
<div style="font-size:0.66rem;color:#333;margin-top:0.75rem;font-family:'Inter',sans-serif;">
Reported on ALL-IDB2 public dataset. Reference values only — not a clinical performance claim.
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Navigate to Metrics
st.markdown('<div style="padding:1.5rem 5rem 0;background:#080808;">', unsafe_allow_html=True)
_, m_mid, _ = st.columns([4, 2, 4])
with m_mid:
    if st.button("Open Metrics Dashboard →", key="open_metrics", use_container_width=True):
        st.switch_page("pages/5_Metrics.py")
st.markdown("</div>", unsafe_allow_html=True)

# ─── DISCLAIMER ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hl-section">
<div class="hl-inner">
<div style="background:rgba(157,114,80,0.06);border:1px solid rgba(157,114,80,0.18);border-radius:10px;padding:1.4rem 1.75rem;display:flex;gap:1rem;align-items:flex-start;">
<span style="color:#c4956a;flex-shrink:0;margin-top:2px;">{ICONS["alert"]}</span>
<div style="font-size:0.84rem;color:#666;line-height:1.72;font-family:'Inter',sans-serif;">
<strong style="color:#c4956a;">RESEARCH USE ONLY — NOT A MEDICAL DEVICE.</strong>
HemaLens is an AI-assisted research prototype. It has not been validated in a clinical trial
and is not approved by the FDA, CE, EMA, or any regulatory authority.
All results must be reviewed by a qualified haematopathologist before any clinical action.
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hl-footer">
  <div style="display:flex;align-items:center;gap:10px;">
    {ICONS["logo"]}
    <div>
      <div style="font-size:0.82rem;font-weight:700;letter-spacing:0.08em;color:#f0f0f0;
                  font-family:'Inter',sans-serif;text-transform:uppercase;">HEMALENS</div>
      <div style="font-size:0.58rem;color:#333;letter-spacing:0.12em;text-transform:uppercase;
                  font-family:'Inter',sans-serif;">AI SCREENING PLATFORM</div>
    </div>
  </div>
  <div style="display:flex;gap:2rem;align-items:center;">
""", unsafe_allow_html=True)

fc1, fc2, fc3, fc4, fc5 = st.columns(5)
fp = [
    (fc1, "Upload",       "pages/2_Upload.py"),
    (fc2, "Prediction",   "pages/3_Prediction.py"),
    (fc3, "Explainability","pages/4_Explainability.py"),
    (fc4, "Metrics",      "pages/5_Metrics.py"),
    (fc5, "About",        "pages/6_About.py"),
]
for col, label, page_path in fp:
    with col:
        if st.button(label, key=f"footer_{label}"):
            st.switch_page(page_path)

st.markdown("""
  </div>
  <div style="font-size:0.68rem;color:#333;font-family:'Inter',sans-serif;margin-top:0.5rem;width:100%;text-align:center;">
    Built with PyTorch · Streamlit · ReportLab · 2026 · Research Use Only
  </div>
</div>
""", unsafe_allow_html=True)
