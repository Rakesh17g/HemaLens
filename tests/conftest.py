"""
Shared pytest fixtures for the ALL Detection test suite.

All heavy objects (model, image tensor, PIL image) are built at module
scope so they are created once per test session, not once per test.

Fixtures
--------
device          : torch.device (cpu for CI)
model           : EfficientNetB0 (pretrained=False, eval mode)
rand_tensor     : (1,3,224,224) random normalised tensor
rand_pil        : 224×224 RGB PIL Image (random noise)
rand_bytes      : raw PNG bytes of rand_pil (for load-from-bytes paths)
tmp_checkpoint  : tmp_path-scoped .pth file (saves / loads model state)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── constants shared across all test modules ──────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 2


# ─────────────────────────────────────────────────────────────────────────────
# Session-scope: built once, shared by the full test run
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")  # CPU only — no GPU required for unit tests


@pytest.fixture(scope="session")
def model(device):
    """Untrained EfficientNetB0 in eval mode."""
    from src.models.efficientnet import build_model

    m = build_model(pretrained=False, device=device)
    m.eval()
    return m


@pytest.fixture(scope="session")
def rand_tensor() -> torch.Tensor:
    """(1, 3, 224, 224) float tensor — values in ImageNet-normalised range."""
    torch.manual_seed(42)
    return torch.randn(1, 3, IMG_SIZE, IMG_SIZE)


@pytest.fixture(scope="session")
def rand_pil() -> Image.Image:
    """224×224 RGB PIL image with random pixel values."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture(scope="session")
def rand_bytes(rand_pil) -> bytes:
    """PNG bytes of rand_pil — mimics st.file_uploader output."""
    buf = io.BytesIO()
    rand_pil.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Function-scope: unique per test (checkpoints, temp dirs)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_checkpoint(tmp_path, model) -> Path:
    """Write model state to a .pth file and return the path."""
    ckpt = tmp_path / "test_model.pth"
    torch.save({"model_state_dict": model.state_dict()}, str(ckpt))
    return ckpt


@pytest.fixture()
def tmp_checkpoint_flat(tmp_path, model) -> Path:
    """Flat checkpoint (state_dict only, no wrapper dict)."""
    ckpt = tmp_path / "flat_model.pth"
    torch.save(model.state_dict(), str(ckpt))
    return ckpt
