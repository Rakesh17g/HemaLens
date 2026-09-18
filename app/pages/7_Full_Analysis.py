"""
Page 7 — Full Analysis Pipeline
=================================
Upload → Preprocess → Predict → Grad-CAM → Confidence → PDF Export
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.components.model_utils import (
    checkpoint_exists,
    generate_pdf_bytes,
    get_device,
    init_session,
    run_full_pipeline,
)

# page config moved to app.py
from app.components.styles import (
    ICONS,
    confidence_bar,
    inject_css,
    page_header,
    stat_row,
)

inject_css(active="analysis")
init_session()

device = get_device().upper()
ckpt_ok = checkpoint_exists()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    page_header(
        "FULL ANALYSIS",
        "End-to-End Pipeline",
        "Upload → Preprocess → Inference → Confidence → Grad-CAM → PDF Report — in one workflow.",
    ),
    unsafe_allow_html=True,
)

# ─── Workflow progress indicator ──────────────────────────────────────────────
steps_html = "".join(
    [
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px;flex:1;">'
        f'  <div style="font-size:0.58rem;font-weight:700;letter-spacing:0.12em;color:#a5f9ef;font-family:var(--mono);">{num}</div>'
        f'  <div style="font-size:0.72rem;font-weight:500;color:var(--subtext);">{label}</div>'
        f"</div>"
        for num, label in [
            ("01", "Upload"),
            ("02", "Preprocess"),
            ("03", "Inference"),
            ("04", "Confidence"),
            ("05", "Grad-CAM"),
            ("06", "Report"),
        ]
    ]
)
st.markdown(
    f"""
<div style="display:flex;align-items:center;padding:1rem 0 1.5rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem;">
  {steps_html}
</div>""",
    unsafe_allow_html=True,
)

# ─── Step 1: Upload ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-title">01 — Upload Cell Image</div>', unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Blood-smear microscopy image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    label_visibility="collapsed",
)

if uploaded is not None:
    raw_bytes = uploaded.read()
    if raw_bytes != st.session_state.get("uploaded_bytes"):
        st.session_state["uploaded_bytes"] = raw_bytes
        st.session_state["uploaded_filename"] = uploaded.name
        st.session_state["pipeline_result"] = None
elif st.session_state.get("uploaded_bytes") is not None:
    raw_bytes = st.session_state["uploaded_bytes"]
    st.markdown(
        f"""
<div style="background:rgba(165,249,239,0.04);border:1px solid rgba(165,249,239,0.12);
            border-radius:8px;padding:0.65rem 1rem;font-size:0.78rem;color:var(--subtext);margin-bottom:0.75rem;">
  Using image from session: <strong style="color:var(--text);">{st.session_state.get("uploaded_filename", "image")}</strong>.
  Upload a new file to replace it.
</div>""",
        unsafe_allow_html=True,
    )
else:
    raw_bytes = None  # type: ignore
    st.markdown(
        f"""
<div style="text-align:center;padding:4rem 2rem;color:var(--muted);">
  <div style="display:flex;justify-content:center;margin-bottom:1rem;opacity:0.25;">{ICONS["image-plus"]}</div>
  <div style="font-size:0.9rem;font-weight:500;color:var(--subtext);">No image uploaded</div>
  <div style="font-size:0.78rem;margin-top:0.3rem;">Upload a peripheral blood-smear image to begin the analysis pipeline.</div>
</div>""",
        unsafe_allow_html=True,
    )

if raw_bytes:
    from PIL import Image as _PIL

    thumb = _PIL.open(io.BytesIO(raw_bytes)).convert("RGB")
    prev_col, btn_col = st.columns([1, 3], gap="large")
    with prev_col:
        st.image(
            thumb,
            caption=st.session_state.get("uploaded_filename", ""),
            use_container_width=True,
        )
    with btn_col:
        config_html = (
            stat_row("Filename", st.session_state.get("uploaded_filename", "?"))
            + stat_row("Device", device)
            + stat_row("Threshold", str(st.session_state["threshold"]))
            + stat_row("MC Passes", str(st.session_state["mc_passes"]))
            + stat_row(
                "Grad-CAM",
                f"{st.session_state['gradcam_method'].upper()} · layer {st.session_state['gradcam_layer']}",
            )
        )
        st.markdown(
            f'<div class="medical-card">{config_html}</div>', unsafe_allow_html=True
        )
        run_btn = st.button(
            "Run Full Analysis", use_container_width=True, type="primary"
        )

    # ── Pipeline execution ────────────────────────────────────────────────────
    if run_btn or st.session_state.get("pipeline_result"):
        if run_btn or st.session_state["pipeline_result"] is None:
            if not checkpoint_exists():
                st.error("Checkpoint not found. Update the path in the sidebar.")
                st.stop()

            with st.status("Running analysis pipeline…", expanded=True) as status:
                st.write("Preprocessing image…")
                st.write("Running EfficientNet-B0 inference…")
                st.write("Estimating confidence…")
                st.write("Computing Grad-CAM…")

                result = run_full_pipeline(
                    image_bytes=st.session_state["uploaded_bytes"],
                    checkpoint=st.session_state["checkpoint_path"],
                    threshold=st.session_state["threshold"],
                    mc_passes=st.session_state["mc_passes"],
                    gradcam_method=st.session_state["gradcam_method"],
                    gradcam_layer=st.session_state["gradcam_layer"],
                    filename=st.session_state.get("uploaded_filename", "unknown"),
                )

                if "error" in result:
                    status.update(label="Pipeline failed", state="error")
                    st.error(result["error"])
                    st.stop()

                st.session_state["pipeline_result"] = result
                status.update(label="Analysis complete", state="complete")

        result = st.session_state["pipeline_result"]

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">02 — Diagnostic Results</div>',
            unsafe_allow_html=True,
        )

        prob = result["probability"]
        pred = result["prediction"]
        conf = result["confidence"]
        risk = result["risk_level"]
        note = result["clinical_note"]
        b_sc = result["boundary_score"]
        e_sc = result["entropy_score"]
        mc_sc = result["mcdrop_score"]
        mc_p = result["mcdrop_passes"]

        is_pos = "ALL" in pred.upper()
        pred_color = "#dc443c" if is_pos else "#8fcdb7"
        risk_color = {"LOW": "#8fcdb7", "MEDIUM": "#c4956a", "HIGH": "#dc443c"}.get(
            risk, "#888"
        )

        k1, k2, k3, k4 = st.columns(4)
        for col, label, val, color, sub in [
            (k1, "PREDICTION", pred, pred_color, ""),
            (
                k2,
                "PROBABILITY",
                f"{prob * 100:.1f}%",
                pred_color,
                f"threshold {result['threshold']:.2f}",
            ),
            (k3, "CONFIDENCE", f"{conf * 100:.1f}%", "#a5f9ef", "model certainty"),
            (
                k4,
                "RISK",
                risk,
                risk_color,
                {
                    "LOW": "Trustworthy",
                    "MEDIUM": "Review advised",
                    "HIGH": "Human review required",
                }.get(risk, ""),
            ),
        ]:
            col.markdown(
                f"""
<div class="hl-kpi hl-pulse">
  <div class="hl-kpi-label">{label}</div>
  <div class="hl-kpi-value" style="font-size:1.5rem;color:{color};">{val}</div>
  {"" if not sub else f'<div class="hl-kpi-sub">{sub}</div>'}
</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Grad-CAM ─────────────────────────────────────────────────────────
        st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">03 — Grad-CAM Attention Map</div>',
            unsafe_allow_html=True,
        )

        original = result["original_rgb"]
        heatmap = result["heatmap"]
        overlay = result["overlay"]

        img_c1, img_c2, img_c3 = st.columns(3)
        img_c1.image(original, caption="Original", use_container_width=True, clamp=True)
        img_c2.image(
            result["colormap"],
            caption=f"Attention ({result['gradcam_method'].upper()})",
            use_container_width=True,
            clamp=True,
        )
        img_c3.image(overlay, caption="Overlay", use_container_width=True, clamp=True)

        hm_stats = (
            stat_row("Max activation", f"{heatmap.max():.4f}")
            + stat_row("Mean activation", f"{heatmap.mean():.4f}")
            + stat_row("Coverage > 0.5", f"{(heatmap > 0.5).mean() * 100:.1f}%")
            + stat_row("Layer", f"features[{result['gradcam_layer']}]")
            + stat_row("Method", result["gradcam_method"].upper())
        )
        st.markdown(
            f'<div class="medical-card">{hm_stats}</div>', unsafe_allow_html=True
        )

        # ── Confidence breakdown ──────────────────────────────────────────────
        st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">04 — Confidence Breakdown</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.2, 0.8], gap="large")

        with left:
            bars = (
                confidence_bar(b_sc, "Boundary Distance")
                + confidence_bar(e_sc, "Predictive Entropy")
                + (
                    confidence_bar(mc_sc, f"MC Dropout (T={mc_p})")
                    if mc_p > 0
                    else '<div style="color:var(--muted);font-size:0.75rem;margin:0.4rem 0;">MC Dropout disabled</div>'
                )
                + confidence_bar(conf, "Combined Confidence")
            )
            detail = (
                stat_row("Boundary", f"{b_sc:.4f}  |p−t| / max(t,1−t)")
                + stat_row("Entropy", f"{e_sc:.4f}  1 − H(p)/ln(2)")
                + (stat_row("MC Drop", f"{mc_sc:.4f}  1 − σ/0.25") if mc_p > 0 else "")
                + stat_row(
                    "Weights",
                    " · ".join(f"{v:.2f}" for v in result["weights"].values()),
                )
            )
            st.markdown(
                f'<div class="medical-card">{bars}</div>'
                f'<div class="medical-card">{detail}</div>',
                unsafe_allow_html=True,
            )

        with right:
            components = ["Boundary", "Entropy", "MC Dropout", "Combined"]
            vals = [b_sc, e_sc, mc_sc if mc_p > 0 else 0.0, conf]
            radar = go.Figure(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=components + [components[0]],
                    fill="toself",
                    line={"color": "#a5f9ef", "width": 1.5},
                    fillcolor="rgba(165,249,239,0.08)",
                )
            )
            radar.update_layout(
                polar={
                    "bgcolor": "#0e0e0e",
                    "radialaxis": {
                        "range": [0, 1],
                        "tickfont": {"size": 8, "color": "#555"},
                        "gridcolor": "rgba(255,255,255,0.05)",
                        "linecolor": "rgba(255,255,255,0.05)",
                    },
                    "angularaxis": {
                        "tickfont": {"size": 10, "color": "#888"},
                        "gridcolor": "rgba(255,255,255,0.05)",
                        "linecolor": "rgba(255,255,255,0.05)",
                    },
                },
                paper_bgcolor="#050505",
                font={"color": "#888", "family": "Inter"},
                margin={"l": 30, "r": 30, "t": 20, "b": 20},
                showlegend=False,
                height=260,
            )
            st.plotly_chart(radar, use_container_width=True)

        # Clinical guidance banner
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
            border-radius:8px;padding:0.9rem 1.1rem;margin:0.75rem 0;">
  <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
              color:{risk_color};margin-bottom:0.4rem;">Clinical Guidance — {risk} RISK</div>
  <div style="font-size:0.82rem;color:var(--subtext);line-height:1.6;">{note}</div>
</div>""",
            unsafe_allow_html=True,
        )

        # ── PDF Export ────────────────────────────────────────────────────────
        st.markdown('<hr class="fancy-hr">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">05 — Export Report</div>',
            unsafe_allow_html=True,
        )

        pdf_col, info_col = st.columns([1, 2], gap="large")

        with pdf_col:
            st.markdown(
                f"""
<div class="medical-card" style="text-align:center;padding:2rem 1.5rem;">
  <div style="display:flex;justify-content:center;margin-bottom:1rem;opacity:0.4;">
    {ICONS["file-text"].replace('width="16" height="16"', 'width="36" height="36"')}
  </div>
  <div style="font-weight:600;color:var(--text);margin-bottom:0.4rem;font-size:0.9rem;">
    Medical PDF Report
  </div>
  <div style="color:var(--subtext);font-size:0.76rem;line-height:1.6;">
    A4 format · Original image · Heatmap · Diagnosis · Confidence · Disclaimer
  </div>
</div>""",
                unsafe_allow_html=True,
            )

            with st.spinner("Generating PDF…"):
                try:
                    pdf_bytes = generate_pdf_bytes(
                        pipeline_result_dict={
                            k: v
                            for k, v in result.items()
                            if not isinstance(v, np.ndarray)
                        }
                        | {
                            "original_rgb": result["original_rgb"],
                            "overlay": result["overlay"],
                        },
                        patient_id=st.session_state.get("patient_id", "ANON"),
                        sample_id=st.session_state.get("sample_id", "N/A"),
                        institution=st.session_state.get("institution", "HemaLens AI"),
                        analyst=st.session_state.get(
                            "analyst", "AI Diagnostic Assistant"
                        ),
                        model_version=st.session_state.get(
                            "model_version", "EfficientNet-B0 v1.0"
                        ),
                    )

                    if pdf_bytes:
                        fname = "hemalens_medical_report.pdf"
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name=fname,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                        st.markdown(
                            f'<div style="font-size:0.7rem;color:var(--muted);text-align:center;margin-top:0.4rem;">'
                            f"Ready · {len(pdf_bytes) // 1024} KB</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("Internal Error: PDF generation returned empty bytes.")
                except Exception as e:  # noqa: BLE001
                    import traceback

                    st.error(
                        f"PDF generation failed! Error: {e!r}\n\nTraceback:\n{traceback.format_exc()}"
                    )

        with info_col:
            st.markdown(
                '<div class="section-title">Report Contents</div>',
                unsafe_allow_html=True,
            )
            contents = [
                ("Original Image", "224×224 blood-smear (Wright-Giemsa)"),
                (
                    "Grad-CAM Heatmap",
                    f"MAGMA overlay · {result['gradcam_method'].upper()} · layer {result['gradcam_layer']}",
                ),
                (
                    "Diagnosis Panel",
                    f"{pred} · {prob * 100:.1f}% · {conf * 100:.1f}% confidence",
                ),
                ("Confidence Scores", "Boundary · Entropy · MC Dropout"),
                (
                    "Model Details",
                    f"{st.session_state.get('model_version', 'EfficientNet-B0 v1.0')}",
                ),
                ("Clinical Guidance", f"{risk} risk — {note[:55]}…"),
                ("Regulatory Disclaimer", "HIPAA · GDPR · Research Use Only"),
            ]
            rows = "".join(
                f'<div class="stat-row">'
                f'<span class="stat-key">{label}</span>'
                f'<span class="stat-value" style="color:var(--subtext);font-size:0.74rem;">{desc}</span>'
                f"</div>"
                for label, desc in contents
            )
            st.markdown(
                f'<div class="medical-card">{rows}</div>', unsafe_allow_html=True
            )

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown(
            f"""
<div class="hl-disclaimer" style="margin-top:1.5rem;">
  <span style="color:var(--amber);flex-shrink:0;">{ICONS["alert"]}</span>
  <div>
    <strong>RESEARCH USE ONLY — NOT A MEDICAL DEVICE.</strong>
    All AI predictions must be confirmed by a qualified pathologist.
    This report is not approved by the FDA, CE, or any regulatory authority.
  </div>
</div>""",
            unsafe_allow_html=True,
        )
