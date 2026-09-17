"""
Page 2 — Upload Microscopy Image
"""
import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import streamlit as st
from PIL import Image

# page config moved to app.py

from app.components.styles import inject_css, section_title, stat_row, fancy_hr, ICONS, page_header
from app.components.model_utils import init_session, preprocess_image

inject_css(active="upload")
init_session()

st.markdown('<div class="section-title">Upload Options</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
target_size = c1.selectbox("Resize to", [224, 256, 320], index=0)
show_channels = c2.checkbox("Show RGB channels", value=False)
# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(page_header(
    "UPLOAD",
    "Upload Microscopy Image",
    "Load a peripheral blood-smear image for AI-assisted screening. "
    "Wright-Giemsa stained images at ≥ 40× magnification yield best results."
), unsafe_allow_html=True)

# ─── Upload zone ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop microscopy image here or click to browse",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    help="Single blood-smear cell image.",
    label_visibility="collapsed",
)

if uploaded is not None:
    raw_bytes = uploaded.read()
    st.session_state["uploaded_bytes"]    = raw_bytes
    st.session_state["uploaded_filename"] = uploaded.name
    st.session_state["prediction_result"] = None
    st.session_state["gradcam_result"]    = None
    st.session_state["pipeline_result"]   = None

    pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    tensor, orig_rgb = preprocess_image(pil, target_size=target_size)

    st.markdown("<br>", unsafe_allow_html=True)
    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown(f'<div class="section-title">Image Preview</div>', unsafe_allow_html=True)
        st.image(orig_rgb, caption="", use_container_width=True, clamp=True)
        st.markdown(
            f'<div class="hl-img-caption">{uploaded.name} · resized to {target_size}×{target_size}px · RGB</div>',
            unsafe_allow_html=True,
        )
        if show_channels:
            r, g, b = orig_rgb[:, :, 0], orig_rgb[:, :, 1], orig_rgb[:, :, 2]
            ch_cols = st.columns(3)
            for ch_col, channel, name, idx in zip(ch_cols, [r, g, b], ["Red", "Green", "Blue"], [0, 1, 2]):
                arr = np.zeros((*channel.shape, 3), dtype=np.uint8)
                arr[:, :, idx] = channel
                ch_col.image(arr, caption=name, use_container_width=True)

    with col_info:
        pil_orig = Image.open(io.BytesIO(raw_bytes))
        w0, h0   = pil_orig.size
        file_kb  = len(raw_bytes) / 1024
        mode     = pil_orig.mode

        st.markdown(f'<div class="section-title">Image Information</div>', unsafe_allow_html=True)
        info_html = (
            stat_row("Filename",      uploaded.name) +
            stat_row("Original size", f"{w0} × {h0} px") +
            stat_row("File size",     f"{file_kb:.1f} KB") +
            stat_row("Color mode",    mode) +
            stat_row("Resized to",    f"{target_size} × {target_size} px") +
            stat_row("Channels",      "3 (RGB)")
        )
        st.markdown(f'<div class="medical-card">{info_html}</div>', unsafe_allow_html=True)

        arr      = orig_rgb.astype(np.float32)
        st.markdown(f'<div class="section-title" style="margin-top:1rem;">Pixel Statistics</div>', unsafe_allow_html=True)
        pix_html = (
            stat_row("Mean R,G,B", f"({arr[:,:,0].mean():.1f}, {arr[:,:,1].mean():.1f}, {arr[:,:,2].mean():.1f})") +
            stat_row("Std R,G,B",  f"({arr[:,:,0].std():.1f}, {arr[:,:,1].std():.1f}, {arr[:,:,2].std():.1f})") +
            stat_row("Min pixel",  f"{arr.min():.0f}") +
            stat_row("Max pixel",  f"{arr.max():.0f}") +
            stat_row("Brightness", f"{arr.mean():.1f} / 255")
        )
        st.markdown(f'<div class="medical-card">{pix_html}</div>', unsafe_allow_html=True)

        # Channel histogram
        import plotly.graph_objects as go
        fig = go.Figure()
        COLORS = {"R": "#dc443c", "G": "#8fcdb7", "B": "#a5f9ef"}
        for i, (ch, col) in enumerate(COLORS.items()):
            counts, bins = np.histogram(orig_rgb[:, :, i].ravel(), bins=64, range=(0, 256))
            fig.add_trace(go.Scatter(
                x=bins[:-1], y=counts, mode="lines",
                name=f"{ch} channel", line=dict(color=col, width=1.5),
                fill="tozeroy", fillcolor=col.replace("#", "rgba(") + f",0.08)".replace("rgba(", "rgba(")
                if False else f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.08)",
            ))
        fig.update_layout(
            paper_bgcolor="#0e0e0e", plot_bgcolor="#050505",
            font=dict(color="#888888", family="Inter"),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="#0e0e0e", bordercolor="rgba(255,255,255,0.07)", borderwidth=1,
                        font=dict(size=10)),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Pixel value", title_font_size=10),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Count", title_font_size=10),
            height=180,
        )
        st.markdown(f'<div class="section-title" style="margin-top:1rem;">Channel Histogram</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="background:rgba(143,205,183,0.06);border:1px solid rgba(143,205,183,0.2);
            border-radius:8px;padding:0.75rem 1rem;font-size:0.8rem;color:#8fcdb7;
            display:flex;align-items:center;gap:0.6rem;">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>
  <span><strong>{uploaded.name}</strong> uploaded successfully. Navigate to <strong>Prediction</strong> to run analysis.</span>
</div>
""", unsafe_allow_html=True)

elif st.session_state["uploaded_bytes"] is not None:
    st.markdown(f"""
<div style="background:rgba(165,249,239,0.05);border:1px solid rgba(165,249,239,0.15);
            border-radius:8px;padding:0.75rem 1rem;font-size:0.8rem;color:var(--subtext);">
  Image already loaded: <strong style="color:var(--text);">{st.session_state.get('uploaded_filename','image')}</strong>.
  Upload a new file to replace it, or navigate to Prediction.
</div>
""", unsafe_allow_html=True)
    pil = Image.open(io.BytesIO(st.session_state["uploaded_bytes"]))
    st.image(pil, caption=st.session_state.get("uploaded_filename", ""), width=320)
else:
    st.markdown(f"""
<div style="text-align:center;padding:5rem 2rem;color:var(--muted);">
  <div style="display:flex;justify-content:center;margin-bottom:1.2rem;opacity:0.3;">
    {ICONS["image-plus"]}
  </div>
  <div style="font-size:0.95rem;font-weight:500;color:var(--subtext);margin-bottom:0.4rem;">
    No image uploaded yet
  </div>
  <div style="font-size:0.8rem;max-width:320px;margin:0 auto;line-height:1.6;">
    Drag and drop a blood-smear cell image above, or click to browse.
    Supported: PNG · JPG · BMP · TIFF.
  </div>
</div>
""", unsafe_allow_html=True)
