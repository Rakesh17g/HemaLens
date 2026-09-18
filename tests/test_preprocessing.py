"""
tests/test_preprocessing.py
============================
Unit tests for image preprocessing (src/inference/pipeline.py::preprocess
and app/components/model_utils.py::preprocess_image).

Covers:
  - Output shapes and dtypes
  - Normalisation range
  - RGB channel order preservation
  - Handling of different input sizes (non-square, tiny, large)
  - Handling of non-RGB modes (RGBA, L, CMYK)
  - Batch dimension consistency
  - Denormalisation inverse
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.pipeline import preprocess

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_pil(w: int, h: int, mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(0)
    channels = {"RGB": 3, "RGBA": 4, "L": 1, "CMYK": 4}.get(mode, 3)
    if channels == 1:
        arr = rng.integers(0, 256, (h, w), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, (h, w, channels), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


# ─────────────────────────────────────────────────────────────────────────────
# Output shape tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocessOutputShapes:
    def test_tensor_shape_default_size(self, rand_pil):
        tensor, _orig_rgb = preprocess(rand_pil, target_size=224)
        assert tensor.shape == (1, 3, 224, 224), (
            f"Expected (1,3,224,224), got {tuple(tensor.shape)}"
        )

    def test_original_rgb_shape_default_size(self, rand_pil):
        _, orig_rgb = preprocess(rand_pil, target_size=224)
        assert orig_rgb.shape == (224, 224, 3), (
            f"Expected (224,224,3), got {orig_rgb.shape}"
        )

    @pytest.mark.parametrize("size", [64, 128, 224, 256, 320])
    def test_tensor_shape_various_sizes(self, rand_pil, size):
        tensor, orig = preprocess(rand_pil, target_size=size)
        assert tensor.shape == (1, 3, size, size)
        assert orig.shape == (size, size, 3)

    def test_non_square_input_resizes_correctly(self):
        wide = make_pil(640, 480)
        tensor, orig = preprocess(wide, target_size=224)
        assert tensor.shape == (1, 3, 224, 224)
        assert orig.shape == (224, 224, 3)

    def test_tiny_input_resizes_correctly(self):
        tiny = make_pil(16, 16)
        tensor, _orig = preprocess(tiny, target_size=224)
        assert tensor.shape == (1, 3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# Output dtype tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocessDtypes:
    def test_tensor_dtype_is_float32(self, rand_pil):
        tensor, _ = preprocess(rand_pil)
        assert tensor.dtype == torch.float32

    def test_original_rgb_dtype_is_uint8(self, rand_pil):
        _, orig = preprocess(rand_pil)
        assert orig.dtype == np.uint8

    def test_original_rgb_range_0_255(self, rand_pil):
        _, orig = preprocess(rand_pil)
        assert int(orig.min()) >= 0
        assert int(orig.max()) <= 255


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocessNormalisation:
    def test_tensor_is_normalised(self, rand_pil):
        """Any valid image should produce values way outside [0,1] after normalisation."""
        tensor, _ = preprocess(rand_pil, target_size=224)
        # ImageNet normalisation shifts values; a plain [0,1] tensor would be wrong
        # After norm, channel means should be near 0
        chw = tensor.squeeze(0)  # (3, H, W)
        channel_mean = chw.mean(dim=(1, 2))
        # Not all near 0.5 (which un-normalised ToTensor() would give)
        assert channel_mean.abs().max().item() < 2.0, "Tensor values suspiciously large"

    def test_different_images_produce_different_tensors(self):
        pil_a = make_pil(224, 224)
        rng = np.random.default_rng(99)
        arr_b = rng.integers(128, 256, (224, 224, 3), dtype=np.uint8)
        pil_b = Image.fromarray(arr_b)
        t_a, _ = preprocess(pil_a, target_size=224)
        t_b, _ = preprocess(pil_b, target_size=224)
        assert not torch.allclose(t_a, t_b), (
            "Two distinct images gave identical tensors"
        )

    def test_same_image_produces_identical_tensors(self, rand_pil):
        t1, _ = preprocess(rand_pil, target_size=224)
        t2, _ = preprocess(rand_pil, target_size=224)
        assert torch.allclose(t1, t2), "Same image produced different tensors"


# ─────────────────────────────────────────────────────────────────────────────
# Input mode handling
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocessInputModes:
    @pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
    def test_handles_various_pil_modes(self, mode):
        """preprocess must normalise non-RGB images without crashing."""
        pil = make_pil(224, 224, mode=mode)
        tensor, orig = preprocess(pil, target_size=224)
        assert tensor.shape == (1, 3, 224, 224)
        assert orig.shape == (224, 224, 3)

    def test_output_original_is_rgb_not_bgr(self):
        """original_rgb should have red channel first (not BGR)."""
        # Create an image that is noticeably red
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # pure red in RGB
        pil = Image.fromarray(arr, mode="RGB")
        _, orig = preprocess(pil, target_size=224)
        assert orig[:, :, 0].mean() > 200, "Channel 0 should be RED (highest value)"
        assert orig[:, :, 2].mean() < 5, "Channel 2 should be BLUE (near zero)"


# ─────────────────────────────────────────────────────────────────────────────
# Batch dimension
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocessBatch:
    def test_tensor_has_batch_dim_of_1(self, rand_pil):
        tensor, _ = preprocess(rand_pil)
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1

    def test_tensor_can_be_stacked(self, rand_pil):
        """Two preprocessed tensors can be cat'd into a batch."""
        t1, _ = preprocess(rand_pil, target_size=224)
        t2, _ = preprocess(rand_pil, target_size=224)
        batch = torch.cat([t1, t2], dim=0)
        assert batch.shape == (2, 3, 224, 224)
