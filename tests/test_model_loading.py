"""
tests/test_model_loading.py
============================
Unit tests for model construction and checkpoint loading.

Covers:
  - build_model() produces correct architecture
  - Phase management (1/2/3) trainable parameter counts
  - Forward pass shape correctness
  - Sigmoid output in [0, 1]
  - load_model(): wrapping dict checkpoint ("model_state_dict")
  - load_model(): flat checkpoint (raw state_dict)
  - load_model(): missing checkpoint path → returns None gracefully
  - load_model(): corrupted checkpoint → raises descriptive error
  - Model eval mode is set correctly
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildModel:

    def test_model_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_model_has_features_attr(self, model):
        assert hasattr(model, "features"), "Model must have a .features attribute for GradCAM"

    def test_model_has_classifier_head(self, model):
        assert hasattr(model, "classifier") or hasattr(model, "head"), \
            "Model must expose a classifier/head attribute"

    def test_total_parameter_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        # EfficientNet-B0 ≈ 5.3M; allow ±20%
        assert 4_000_000 < total < 7_000_000, \
            f"Expected ~5.3M params, got {total:,}"

    def test_output_shape_single_image(self, model, rand_tensor, device):
        x   = rand_tensor.to(device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1), f"Expected (1,1), got {tuple(out.shape)}"

    @pytest.mark.parametrize("batch", [1, 2, 4])
    def test_output_shape_various_batches(self, model, device, batch):
        x   = torch.randn(batch, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, 1)

    def test_sigmoid_output_in_unit_interval(self, model, rand_tensor, device):
        x = rand_tensor.to(device)
        with torch.no_grad():
            logit = model(x)
            prob  = torch.sigmoid(logit)
        assert 0.0 <= float(prob.item()) <= 1.0

    def test_model_starts_in_eval_mode(self, model):
        assert not model.training, "build_model() should return a model in eval mode"


# ─────────────────────────────────────────────────────────────────────────────
# Phase management
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseManagement:

    def test_phase_1_freezes_backbone(self, device):
        from src.models.efficientnet import build_model
        m = build_model(pretrained=False, device=device)
        m.set_phase(1)
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert trainable < total * 0.20, \
            f"Phase 1 should have <20% trainable; got {100*trainable/total:.1f}%"

    def test_phase_2_unfreezes_more_than_phase_1(self, device):
        from src.models.efficientnet import build_model
        m = build_model(pretrained=False, device=device)
        m.set_phase(1)
        tr1 = sum(p.numel() for p in m.parameters() if p.requires_grad)
        m.set_phase(2)
        tr2 = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert tr2 > tr1, "Phase 2 must unfreeze more params than Phase 1"

    def test_phase_3_unfreezes_nearly_all(self, device):
        from src.models.efficientnet import build_model
        m = build_model(pretrained=False, device=device)
        m.set_phase(3)
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert trainable >= total * 0.95, \
            f"Phase 3 should have ≥95% trainable; got {100*trainable/total:.1f}%"

    def test_phases_are_monotonically_increasing(self, device):
        from src.models.efficientnet import build_model
        m   = build_model(pretrained=False, device=device)
        tr  = []
        for phase in [1, 2, 3]:
            m.set_phase(phase)
            tr.append(sum(p.numel() for p in m.parameters() if p.requires_grad))
        assert tr[0] <= tr[1] <= tr[2], \
            f"Trainable params should be non-decreasing: {tr}"


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointLoading:

    def test_load_wrapped_checkpoint(self, tmp_checkpoint, device):
        """load_model() handles {'model_state_dict': ...} checkpoints."""
        from app.components.model_utils import load_model
        loaded_model, loaded_device = load_model(str(tmp_checkpoint), str(device))
        assert loaded_model is not None
        assert isinstance(loaded_model, nn.Module)

    def test_load_flat_checkpoint(self, tmp_checkpoint_flat, device):
        """load_model() handles raw state_dict checkpoints."""
        from app.components.model_utils import load_model
        loaded_model, _ = load_model(str(tmp_checkpoint_flat), str(device))
        assert loaded_model is not None

    def test_missing_checkpoint_returns_none(self, device):
        """load_model() should return (None, device) — not raise — on missing path."""
        from app.components.model_utils import load_model
        model_out, device_out = load_model("/non/existent/path.pth", str(device))
        assert model_out is None

    def test_loaded_model_is_in_eval_mode(self, tmp_checkpoint, device):
        from app.components.model_utils import load_model
        m, _ = load_model(str(tmp_checkpoint), str(device))
        assert not m.training, "Loaded model must be in eval mode"

    def test_loaded_model_produces_same_output(self, model, tmp_checkpoint, rand_tensor, device):
        """Checkpoint round-trip preserves weights exactly."""
        from app.components.model_utils import load_model
        loaded, _ = load_model(str(tmp_checkpoint), str(device))
        x = rand_tensor.to(device)
        with torch.no_grad():
            orig_out   = model(x)
            loaded_out = loaded(x)
        assert torch.allclose(orig_out, loaded_out, atol=1e-5), \
            "Checkpoint round-trip changed model output"

    def test_corrupted_checkpoint_raises(self, tmp_path, device):
        """A file that is not a valid .pth should raise an error."""
        bad = tmp_path / "bad.pth"
        bad.write_bytes(b"not a pickle file at all !@#$")
        from app.components.model_utils import load_model
        with pytest.raises(Exception):
            load_model(str(bad), str(device))
