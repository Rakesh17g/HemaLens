"""
Hugging Face Spaces entrypoint
================================
Hugging Face Spaces expects the Streamlit app to be launched from a file
called ``app.py`` at the repository root.

This file launches the real multi-page Streamlit app from ``app/app.py``
by inserting its directory into sys.path so ``from app.components ...``
resolves correctly even when invoked from the repo root.

NOTE:  On HF Spaces the model checkpoint must either be:
  (a) included in the repo (small models only), or
  (b) downloaded from HF Hub at startup (recommended for large files).

To use (b), set the environment variable:
    HF_MODEL_REPO  = "your-username/all-detection-model"
    HF_MODEL_FILE  = "efficientnet_b0_best.pth"

The app will auto-download on first run and cache in models/checkpoints/.
"""

import os
import sys
from pathlib import Path

# ── Project root must be on sys.path so ``src.*`` imports work ─────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Optional: download model checkpoint from HF Hub ──────────────────────────
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
HF_MODEL_FILE = os.environ.get("HF_MODEL_FILE", "efficientnet_b0_best.pth")
CKPT_DIR      = ROOT / "models" / "checkpoints"
CKPT_PATH     = CKPT_DIR / HF_MODEL_FILE

if HF_MODEL_REPO and not CKPT_PATH.exists():
    try:
        from huggingface_hub import hf_hub_download
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id   = HF_MODEL_REPO,
            filename  = HF_MODEL_FILE,
            local_dir = str(CKPT_DIR),
        )
        print(f"✔  Checkpoint downloaded: {downloaded}")
    except Exception as e:
        print(f"⚠️  Could not download checkpoint: {e}")
        print("    The app will still start; upload a checkpoint via the sidebar.")

# ── Launch the real Streamlit app ─────────────────────────────────────────────
# We run the actual app/app.py as __main__ so all Streamlit page registrations
# and st.set_page_config() fire in the correct context.
# Running via streamlit CLI:  streamlit run app.py
# Running via HF Spaces:      same — HF executes: streamlit run app.py

# Streamlit will discover app/pages/* automatically because Streamlit's
# page discovery uses the scriptfile's directory, not CWD.
# Since this root app.py just delegates, we redirect it to app/app.py.

import runpy
runpy.run_path(str(ROOT / "app" / "app.py"), run_name="__main__")
