"""
Page 3 — AI Prediction
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
import streamlit as st

from app.components.model_utils import (
    checkpoint_exists,
    has_image,
    init_session,
    run_prediction,
)

# page config moved to app.py
from app.components.styles import (
    ICONS,
    confidence_bar,
    inject_css,
    page_header,
    stat_row,
)

inject_css(active="prediction")
init_session()

ckpt_ok = checkpoint_exists()
# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    page_header(
        "DIAGNOSIS",
        "AI Prediction",
        "EfficientNet-B0 classification with calibrated multi-signal confidence estimation.",
    ),
    unsafe_allow_html=True,
)

if not has_image():
    st.markdown(
        f"""
<div style="background:rgba(157,114,80,0.07);border:1px solid rgba(157,114,80,0.2);
            border-radius:8px;padding:0.8rem 1rem;font-size:0.8rem;color:#c4956a;
            display:flex;align-items:center;gap:0.6rem;">
  {ICONS["alert"]} No image loaded. Upload a blood-smear image first.
</div>""",
        unsafe_allow_html=True,
    )
    st.stop()

if not checkpoint_exists():
    st.markdown(
        f"""
<div style="background:rgba(220,68,60,0.08);border:1px solid rgba(220,68,60,0.2);
            border-radius:8px;padding:0.8rem 1rem;font-size:0.8rem;color:#dc443c;
            display:flex;align-items:center;gap:0.6rem;">
  {ICONS["alert"]} Model checkpoint not found. Check the path in the sidebar.
</div>""",
        unsafe_allow_html=True,
    )
    st.stop()

# ─── Run prediction ───────────────────────────────────────────────────────────
with st.spinner("Running inference…"):
    result = run_prediction(
        image_bytes=st.session_state["uploaded_bytes"],
        checkpoint=st.session_state["checkpoint_path"],
        threshold=st.session_state["threshold"],
        mc_passes=st.session_state["mc_passes"],
    )
    st.session_state["prediction_result"] = result

if "error" in result:
    st.error(f"Inference failed: {result['error']}")
    st.stop()

prob = result["probability"]
pred = result["prediction"]
conf = result["confidence"]
risk = result["risk_level"]
thresh = result["threshold"]
note = result["clinical_note"]
b_sc = result["boundary_score"]
e_sc = result["entropy_score"]
mc_sc = result["mcdrop_score"]
mc_p = result["mcdrop_passes"]
orig = result["original_rgb"]

is_pos = "ALL" in pred.upper()
pred_color = "#dc443c" if is_pos else "#8fcdb7"
risk_color = {"LOW": "#8fcdb7", "MEDIUM": "#c4956a", "HIGH": "#dc443c"}.get(
    risk, "#888888"
)

# ─── Image + diagnosis side by side ──────────────────────────────────────────
img_col, diag_col = st.columns([1, 1.4], gap="large")

with img_col:
    st.markdown(
        '<div class="section-title">Uploaded Image</div>', unsafe_allow_html=True
    )
    st.image(orig, use_container_width=True, clamp=True)
    st.markdown(
        f'<div class="hl-img-caption">{st.session_state.get("uploaded_filename", "image")} · 224×224px · RGB</div>',
        unsafe_allow_html=True,
    )

with diag_col:
    st.markdown(
        '<div class="section-title">Analysis Results</div>', unsafe_allow_html=True
    )

    # Main KPI grid
    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, color, sub in [
        (k1, "PREDICTION", pred, pred_color, ""),
        (
            k2,
            "PROBABILITY",
            f"{prob * 100:.1f}%",
            pred_color,
            f"threshold {thresh:.2f}",
        ),
        (k3, "CONFIDENCE", f"{conf * 100:.1f}%", "#a5f9ef", "model certainty"),
        (k4, "RISK", risk, risk_color, ""),
    ]:
        col.markdown(
            f"""
<div class="hl-kpi hl-pulse">
  <div class="hl-kpi-label">{label}</div>
  <div class="hl-kpi-value" style="font-size:1.3rem;color:{color};">{val}</div>
  {"" if not sub else f'<div class="hl-kpi-sub">{sub}</div>'}
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence breakdown
    st.markdown(
        '<div class="section-title">Confidence Breakdown</div>', unsafe_allow_html=True
    )
    breakdown_html = (
        confidence_bar(b_sc, "Boundary Distance")
        + confidence_bar(e_sc, "Predictive Entropy")
        + (
            confidence_bar(mc_sc, f"MC Dropout (T={mc_p})")
            if mc_p > 0
            else '<div style="font-size:0.75rem;color:var(--muted);margin:0.4rem 0;">MC Dropout disabled (set passes > 0)</div>'
        )
        + confidence_bar(conf, "Combined Confidence")
    )
    st.markdown(
        f'<div class="medical-card">{breakdown_html}</div>', unsafe_allow_html=True
    )

    # Clinical guidance
    bg_map = {
        "LOW": "rgba(143,205,183,0.06)",
        "MEDIUM": "rgba(157,114,80,0.07)",
        "HIGH": "rgba(220,68,60,0.07)",
    }
    bdr_map = {
        "LOW": "rgba(143,205,183,0.2)",
        "MEDIUM": "rgba(157,114,80,0.2)",
        "HIGH": "rgba(220,68,60,0.2)",
    }
    st.markdown(
        f"""
<div style="background:{bg_map.get(risk, "rgba(255,255,255,0.03)")};
            border:1px solid {bdr_map.get(risk, "rgba(255,255,255,0.07)")};
            border-radius:8px;padding:0.9rem 1rem;margin-top:0.75rem;">
  <div style="font-size:0.62rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;
              color:{risk_color};margin-bottom:0.5rem;">
    Clinical Guidance — {risk} RISK
  </div>
  <div style="font-size:0.82rem;color:var(--subtext);line-height:1.6;">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)

# ─── Charts row ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown(
        '<div class="section-title">Confidence Radar</div>', unsafe_allow_html=True
    )
    components = ["Boundary", "Entropy", "MC Dropout", "Combined"]
    values = [b_sc, e_sc, mc_sc if mc_p > 0 else 0.0, conf]
    radar = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=components + [components[0]],
            fill="toself",
            line={"color": "#a5f9ef", "width": 1.5},
            fillcolor="rgba(165,249,239,0.08)",
            name="Confidence",
        )
    )
    radar.update_layout(
        polar={
            "bgcolor": "#0e0e0e",
            "radialaxis": {
                "range": [0, 1],
                "tickfont": {"size": 8, "color": "#555"},
                "gridcolor": "rgba(255,255,255,0.06)",
                "linecolor": "rgba(255,255,255,0.06)",
            },
            "angularaxis": {
                "tickfont": {"size": 10, "color": "#888"},
                "gridcolor": "rgba(255,255,255,0.06)",
                "linecolor": "rgba(255,255,255,0.06)",
            },
        },
        paper_bgcolor="#050505",
        font={"color": "#888", "family": "Inter"},
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        showlegend=False,
        height=240,
    )
    st.plotly_chart(radar, use_container_width=True)

    # Formula notes
    notes_html = (
        stat_row("Boundary", f"{b_sc:.4f}  ·  |p−t| / max(t,1−t)")
        + stat_row("Entropy", f"{e_sc:.4f}  ·  1 − H(p)/ln(2)")
        + (stat_row("MC Dropout", f"{mc_sc:.4f}  ·  1 − σ/0.25") if mc_p > 0 else "")
        + stat_row(
            "Weights", "0.35 · 0.35 · 0.30" if mc_p > 0 else "0.50 · 0.50 (MC off)"
        )
    )
    st.markdown(f'<div class="medical-card">{notes_html}</div>', unsafe_allow_html=True)

with right_col:
    st.markdown(
        '<div class="section-title">Probability Gauge</div>', unsafe_allow_html=True
    )
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 26, "color": pred_color}},
            gauge={
                "axis": {
                    "range": [0, 100], "tickcolor": "#555", "tickfont": {"color": "#555"}
                },
                "bar": {"color": pred_color},
                "bgcolor": "#141414",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, thresh * 100], "color": "rgba(143,205,183,0.05)"},
                    {"range": [thresh * 100, 100], "color": "rgba(220,68,60,0.05)"},
                ],
                "threshold": {
                    "line": {"color": "#c4956a", "width": 2},
                    "thickness": 0.8,
                    "value": thresh * 100,
                },
            },
        )
    )
    gauge.update_layout(
        paper_bgcolor="#0e0e0e",
        font={"color": "#888", "family": "Inter"},
        margin={"l": 20, "r": 20, "t": 20, "b": 10},
        height=210,
    )
    st.plotly_chart(gauge, use_container_width=True)

    with st.expander("Why is Probability different from Confidence?"):
        st.markdown("""
**Probability** is the raw sigmoid output from the model — how similar the input is to ALL+ patterns.

**Confidence** is a post-hoc trustworthiness score combining:
- **Boundary Distance**: How far the probability is from the decision threshold
- **Predictive Entropy**: Sharpness of the probability distribution
- **MC Dropout**: Consistency across stochastic forward passes

A high probability with low confidence may indicate an out-of-distribution input.
""")

st.markdown(
    f"""
<div class="hl-disclaimer" style="margin-top:1rem;">
  <span style="color:var(--amber);flex-shrink:0;">{ICONS["alert"]}</span>
  <div>
    <strong>RESEARCH USE ONLY.</strong> This AI prediction must be confirmed by a qualified pathologist before any clinical decision.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
