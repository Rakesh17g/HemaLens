"""
Page 4 — Grad-CAM Explainability
"""
import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# page config moved to app.py

from app.components.styles import inject_css, section_title, stat_row, ICONS, page_header
from app.components.model_utils import init_session, has_image, run_gradcam, checkpoint_exists

inject_css(active="gradcam")
init_session()

ckpt_ok = checkpoint_exists()
c1, c2 = st.columns(2)
overlay_alpha = c1.slider("Heatmap opacity", 0.1, 1.0, value=0.55, step=0.05)
colormap_name = c2.selectbox("Colormap", ["Magma", "Hot", "Viridis", "Plasma"], index=0)
# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(page_header(
    "EXPLAINABILITY",
    "Grad-CAM Visualization",
    "Gradient-weighted Class Activation Maps reveal which spatial regions of the blood-smear drove the model's prediction."
), unsafe_allow_html=True)

if not has_image():
    st.markdown(f"""
<div style="background:rgba(157,114,80,0.07);border:1px solid rgba(157,114,80,0.2);
            border-radius:8px;padding:0.8rem 1rem;font-size:0.8rem;color:#c4956a;
            display:flex;align-items:center;gap:0.6rem;">
  {ICONS["alert"]} No image loaded. Upload a blood-smear image first.
</div>""", unsafe_allow_html=True)
    st.stop()

if not checkpoint_exists():
    st.markdown(f"""
<div style="background:rgba(220,68,60,0.08);border:1px solid rgba(220,68,60,0.2);
            border-radius:8px;padding:0.8rem 1rem;font-size:0.8rem;color:#dc443c;
            display:flex;align-items:center;gap:0.6rem;">
  {ICONS["alert"]} Model checkpoint not found.
</div>""", unsafe_allow_html=True)
    st.stop()

# ─── Compute Grad-CAM ─────────────────────────────────────────────────────────
with st.spinner("Computing Grad-CAM…"):
    cam = run_gradcam(
        image_bytes=st.session_state["uploaded_bytes"],
        checkpoint=st.session_state["checkpoint_path"],
        layer=st.session_state["gradcam_layer"],
        method=st.session_state["gradcam_method"],
    )

if "error" in cam:
    st.error(f"Grad-CAM failed: {cam['error']}")
    st.stop()

original = cam["original"]
heatmap  = cam["heatmap"]

import cv2
orig_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
cm_uint8 = (heatmap * 255).astype(np.uint8)
cmap_choice = {
    "Magma": cv2.COLORMAP_MAGMA, "Hot": cv2.COLORMAP_HOT,
    "Viridis": cv2.COLORMAP_VIRIDIS, "Plasma": cv2.COLORMAP_PLASMA,
}[colormap_name]
colormap_bgr = cv2.applyColorMap(cm_uint8, cmap_choice)
overlay_bgr  = cv2.addWeighted(orig_bgr, 1 - overlay_alpha, colormap_bgr, overlay_alpha, 0)
overlay_rgb  = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
colormap_rgb = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)

# ─── View tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Side-by-Side", "Heatmap", "Overlay", "Activation Profile"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.image(original,     caption="Original Cell", use_container_width=True, clamp=True)
    c2.image(colormap_rgb, caption=f"Grad-CAM ({cam['method'].upper()})", use_container_width=True, clamp=True)
    c3.image(overlay_rgb,  caption=f"Overlay  α={overlay_alpha:.2f}", use_container_width=True, clamp=True)
    st.markdown(f"""
<div class="hl-img-caption">
  High-activation regions indicate strongest model attention ·
  Layer: features[{st.session_state['gradcam_layer']}] · Method: {cam['method'].upper()}
</div>""", unsafe_allow_html=True)

with tab2:
    col_hm, col_cb = st.columns([2, 1])
    with col_hm:
        fig = px.imshow(colormap_rgb, title=f"Saliency Map — {cam['method'].upper()}")
        fig.update_layout(
            paper_bgcolor="#050505", plot_bgcolor="#050505",
            font=dict(color="#888", family="Inter"),
            coloraxis_showscale=False,
            margin=dict(l=5, r=5, t=35, b=5),
            title_font=dict(size=12, color="#888"),
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_cb:
        st.markdown('<div class="section-title">Activation Info</div>', unsafe_allow_html=True)
        hw   = heatmap
        info = (
            stat_row("Max activation",  f"{hw.max():.4f}") +
            stat_row("Mean activation", f"{hw.mean():.4f}") +
            stat_row("Std deviation",   f"{hw.std():.4f}") +
            stat_row("High-act pixels", f"{(hw > 0.7).sum()} / {hw.size}") +
            stat_row("Coverage > 0.5",  f"{(hw > 0.5).mean()*100:.1f}%") +
            stat_row("Layer",           f"features[{st.session_state['gradcam_layer']}]") +
            stat_row("Method",          cam['method'].upper())
        )
        st.markdown(f'<div class="medical-card">{info}</div>', unsafe_allow_html=True)
        cbar_y = np.linspace(1, 0, 256).reshape(256, 1)
        cbar_bgr = cv2.applyColorMap((cbar_y * 255).astype(np.uint8), cmap_choice)
        cbar_rgb = cv2.cvtColor(cbar_bgr, cv2.COLOR_BGR2RGB)
        st.image(np.hstack([cbar_rgb] * 30), caption="Low → High", width=60)

with tab3:
    st.image(overlay_rgb, caption=f"Grad-CAM Overlay · α={overlay_alpha:.2f}",
             use_container_width=True, clamp=True)

with tab4:
    st.markdown('<div class="section-title">Activation Profile</div>', unsafe_allow_html=True)
    row_mean = heatmap.mean(axis=1)
    col_mean = heatmap.mean(axis=0)
    H, W = heatmap.shape
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=np.arange(H), y=row_mean, name="Row mean",
        line=dict(color="#a5f9ef", width=1.5), fill="tozeroy",
        fillcolor="rgba(165,249,239,0.08)",
    ))
    fig2.add_trace(go.Scatter(
        x=np.arange(W), y=col_mean, name="Column mean",
        line=dict(color="#8fcdb7", width=1.5), fill="tozeroy",
        fillcolor="rgba(143,205,183,0.08)",
    ))
    fig2.update_layout(
        paper_bgcolor="#0e0e0e", plot_bgcolor="#050505",
        font=dict(color="#888", family="Inter"),
        legend=dict(bgcolor="#0e0e0e", bordercolor="rgba(255,255,255,0.07)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Pixel position"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Mean Activation"),
        margin=dict(l=10, r=10, t=10, b=10), height=260,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:1rem;">Activation Surface (3D)</div>', unsafe_allow_html=True)
    fig3 = go.Figure(go.Surface(
        z=heatmap, colorscale="magma", showscale=True,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True))),
    ))
    fig3.update_layout(
        paper_bgcolor="#050505",
        scene=dict(
            bgcolor="#050505",
            xaxis=dict(backgroundcolor="#0e0e0e", gridcolor="rgba(255,255,255,0.05)", showbackground=True, title="X"),
            yaxis=dict(backgroundcolor="#0e0e0e", gridcolor="rgba(255,255,255,0.05)", showbackground=True, title="Y"),
            zaxis=dict(backgroundcolor="#0e0e0e", gridcolor="rgba(255,255,255,0.05)", showbackground=True, title="Activation"),
        ),
        font=dict(color="#888"), margin=dict(l=0, r=0, t=0, b=0), height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
st.markdown("""
<div class="medical-card" style="font-size:0.82rem;color:var(--subtext);line-height:1.7;">
  <strong style="color:var(--text);font-weight:500;">Interpretation guide.</strong>
  High-activation regions (bright/warm) are where the model found the most discriminative features.
  For ALL+, expect activation concentrated over enlarged, irregular nuclei with prominent nucleoli —
  hallmarks of lymphoblastic cells. For healthy, activation may be diffuse or minimal.
</div>
""", unsafe_allow_html=True)
