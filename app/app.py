"""
HemaLens AI — Main Entry Point
= Routing via st.navigation() =
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(
    page_title="HemaLens AI — Leukemia Screening",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 28 28'><circle cx='14' cy='14' r='13' stroke='%23a5f9ef' stroke-width='1.2' fill='none'/><circle cx='14' cy='14' r='3' fill='%23a5f9ef'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "HemaLens AI — end-to-end medical AI dashboard.",
    },
)

# 1. Initialize session
from app.components.model_utils import init_session

init_session()

# 2. Inject global CSS
from app.components.styles import inject_css

inject_css()

# 3. Define the actual navigation system (Replaces native file-based sidebar routing)
pages = {
    "Navigation": [
        st.Page("pages/1_Home.py", title="Home", icon=":material/home:", default=True),
        st.Page("pages/2_Upload.py", title="Upload", icon=":material/upload:"),
        st.Page("pages/3_Prediction.py", title="Prediction", icon=":material/biotech:"),
        st.Page(
            "pages/4_Explainability.py",
            title="Explainability",
            icon=":material/target:",
        ),
        st.Page("pages/5_Metrics.py", title="Metrics", icon=":material/analytics:"),
        st.Page(
            "pages/7_Full_Analysis.py",
            title="Full Analysis",
            icon=":material/account_tree:",
        ),
        st.Page("pages/6_About.py", title="About", icon=":material/info:"),
    ]
}

# Hide native navigation so we can sequence it ourselves!
pg = st.navigation(pages, position="hidden")

# 4. Settings & Custom Navigation in Sidebar
with st.sidebar:
    from app.components.styles import ICONS

    # --- Brand Logo ---
    st.markdown(
        f"""
<div class="hl-sidebar-brand" style="display:flex;align-items:center;gap:10px;padding:12px 18px 8px;">
  {ICONS["logo"]}
  <div>
    <div class="hl-brand-name" style="font-size:0.9rem;font-weight:700;letter-spacing:0.06em;color:#f0f0f0;line-height:1.1;">HEMALENS</div>
    <div class="hl-brand-sub" style="font-size:0.58rem;letter-spacing:0.14em;color:#888;text-transform:uppercase;">AI Screening</div>
  </div>
</div>
    """,
        unsafe_allow_html=True,
    )

    # --- Custom Navigation Menu ---
    st.markdown(
        '<div class="hl-nav-section" style="margin-left:0.5rem;font-size:0.58rem;color:#777;letter-spacing:0.14em;text-transform:uppercase;padding:12px 0 8px;">Navigation</div>',
        unsafe_allow_html=True,
    )

    st.page_link("pages/1_Home.py", label="Home", icon=":material/home:")
    st.page_link("pages/2_Upload.py", label="Upload", icon=":material/upload:")
    st.page_link("pages/3_Prediction.py", label="Prediction", icon=":material/biotech:")
    st.page_link(
        "pages/4_Explainability.py", label="Explainability", icon=":material/target:"
    )
    st.page_link("pages/5_Metrics.py", label="Metrics", icon=":material/analytics:")
    st.page_link(
        "pages/7_Full_Analysis.py",
        label="Full Analysis",
        icon=":material/account_tree:",
    )
    st.page_link("pages/6_About.py", label="About", icon=":material/info:")

    # --- Model Config ---
    st.markdown(
        '<div class="hl-nav-section" style="margin-left:0.5rem;font-size:0.58rem;color:#777;letter-spacing:0.14em;text-transform:uppercase;border-top:1px solid rgba(255,255,255,0.07);padding-top:16px;margin-top:12px;">Model Config</div>',
        unsafe_allow_html=True,
    )

    import os
    from pathlib import Path
    ckpt_dir = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
    available_ckpts = []
    if ckpt_dir.exists():
        available_ckpts = [f"models/checkpoints/{f.name}" for f in ckpt_dir.iterdir() if f.suffix in ('.pth', '.pt')]
    
    # Fallback if the user's session state got corrupted or they typed an invalid path
    current_ckpt = st.session_state.get("checkpoint_path", "")
    if current_ckpt not in available_ckpts and available_ckpts:
        st.session_state["checkpoint_path"] = available_ckpts[0]

    if available_ckpts:
        st.selectbox(
            "Checkpoint", 
            options=available_ckpts,
            key="checkpoint_path", 
            label_visibility="collapsed"
        )
    else:
        st.text_input(
            "Checkpoint",
            key="checkpoint_path",
            label_visibility="collapsed",
            placeholder="Path to .pth file",
        )
    st.slider(
        "Decision threshold",
        0.20,
        0.80,
        key="threshold",
        step=0.01,
    )
    st.selectbox(
        "MC Dropout passes",
        [0, 10, 20, 50],
        key="mc_passes",
    )

    st.markdown('<hr class="hl-nav-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="hl-nav-section" style="margin-left:0.5rem;font-size:0.65rem;color:#888;letter-spacing:0.1em;text-transform:uppercase;">Report Metadata</div>',
        unsafe_allow_html=True,
    )
    st.text_input(
        "Patient ID",
        key="patient_id",
        label_visibility="collapsed",
    )
    st.text_input(
        "Sample ID",
        key="sample_id",
        label_visibility="collapsed",
    )
    st.text_input(
        "Analyst",
        key="analyst",
        label_visibility="collapsed",
    )
    st.text_input(
        "Institution",
        key="institution",
        label_visibility="collapsed",
    )

# 5. Run the selected page
pg.run()
