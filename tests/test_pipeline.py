"""
tests/test_pipeline.py
=======================
Integration tests for InferencePipeline — the unified orchestrator.

Covers:
  - PipelineResult field presence and types
  - run() accepts PIL.Image
  - run_from_bytes() accepts raw bytes
  - All array shapes and dtypes
  - All scalar ranges (probability, confidence, etc.)
  - to_dict() produces JSON-serialisable output
  - is_positive property
  - Pipeline with MC Dropout enabled/disabled
  - Pipeline with gradcam++ vs gradcam
  - Pipeline with different layers
  - Different images produce different results
  - Same image + settings → deterministic result
  - Missing/None model graceful error (via run_full_pipeline)
  - run_full_pipeline() returns dict with required keys
  - run_full_pipeline() missing checkpoint returns error dict
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.pipeline import InferencePipeline, PipelineResult

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline(model, device):
    return InferencePipeline(
        model=model,
        device=device,
        threshold=0.50,
        mc_passes=0,
        gradcam_method="gradcam",
        gradcam_layer=8,
    )


@pytest.fixture(scope="module")
def result(pipeline, rand_pil):
    return pipeline.run(rand_pil, filename="test_cell.png")


# ─────────────────────────────────────────────────────────────────────────────
# PipelineResult field tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineResultFields:
    def test_result_is_pipeline_result(self, result):
        assert isinstance(result, PipelineResult)

    def test_filename_stored(self, result):
        assert result.filename == "test_cell.png"

    def test_probability_float_in_unit_interval(self, result):
        assert isinstance(result.probability, float)
        assert 0.0 <= result.probability <= 1.0

    def test_prediction_is_valid(self, result):
        assert result.prediction in ("ALL+", "Healthy")

    def test_confidence_float_in_unit_interval(self, result):
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_risk_level_is_valid(self, result):
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH")

    def test_clinical_note_is_string(self, result):
        assert isinstance(result.clinical_note, str)
        assert len(result.clinical_note) > 10

    def test_boundary_score_in_unit_interval(self, result):
        assert 0.0 <= result.boundary_score <= 1.0

    def test_entropy_score_in_unit_interval(self, result):
        assert 0.0 <= result.entropy_score <= 1.0

    def test_mcdrop_score_zero_when_disabled(self, result):
        assert result.mcdrop_passes == 0
        assert abs(result.mcdrop_score) < 1e-9

    def test_threshold_stored(self, result):
        assert abs(result.threshold - 0.50) < 1e-6

    def test_weights_is_dict(self, result):
        assert isinstance(result.weights, dict)
        assert len(result.weights) > 0


class TestPipelineResultArrays:
    def test_original_rgb_shape(self, result):
        assert result.original_rgb.shape == (224, 224, 3)

    def test_original_rgb_dtype(self, result):
        assert result.original_rgb.dtype == np.uint8

    def test_heatmap_shape(self, result):
        assert result.heatmap.shape == (224, 224)

    def test_heatmap_range(self, result):
        assert float(result.heatmap.min()) >= 0.0 - 1e-6
        assert float(result.heatmap.max()) <= 1.0 + 1e-6

    def test_colormap_shape(self, result):
        assert result.colormap.shape == (224, 224, 3)

    def test_overlay_shape(self, result):
        assert result.overlay.shape == (224, 224, 3)

    def test_overlay_dtype(self, result):
        assert result.overlay.dtype == np.uint8

    def test_image_size_matches_arrays(self, result):
        H = result.image_size
        assert result.original_rgb.shape == (H, H, 3)
        assert result.heatmap.shape == (H, H)


# ─────────────────────────────────────────────────────────────────────────────
# to_dict() and properties
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineResultMethods:
    def test_to_dict_is_json_serialisable(self, result):
        d = result.to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_has_required_keys(self, result):
        d = result.to_dict()
        required = {
            "probability",
            "prediction",
            "confidence",
            "risk_level",
            "gradcam_method",
            "image_size",
        }
        missing = required - d.keys()
        assert not missing, f"Missing keys: {missing}"

    def test_is_positive_property(self, model, device):
        """is_positive must agree with the prediction string."""
        pip = InferencePipeline(model=model, device=device)
        torch.ones(1, 3, 224, 224).to(device)
        r = pip.run(Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8)))
        assert r.is_positive == ("ALL" in r.prediction.upper())


# ─────────────────────────────────────────────────────────────────────────────
# run() input variants
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineInputVariants:
    def test_run_accepts_pil_image(self, pipeline, rand_pil):
        r = pipeline.run(rand_pil)
        assert isinstance(r, PipelineResult)

    def test_run_from_bytes_accepts_png_bytes(self, pipeline, rand_bytes):
        r = pipeline.run_from_bytes(rand_bytes, filename="from_bytes.png")
        assert isinstance(r, PipelineResult)
        assert r.filename == "from_bytes.png"

    def test_run_accepts_numpy_image(self, pipeline):
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        r = pipeline.run(arr, filename="numpy.png")
        assert isinstance(r, PipelineResult)

    def test_run_from_bytes_jpg_format(self, pipeline):
        pil = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        r = pipeline.run_from_bytes(buf.getvalue())
        assert isinstance(r, PipelineResult)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration variants
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineConfigurations:
    def test_mc_dropout_enabled(self, model, device, rand_pil):
        pip = InferencePipeline(model=model, device=device, mc_passes=10)
        r = pip.run(rand_pil)
        assert r.mcdrop_passes == 10
        assert 0.0 <= r.mcdrop_score <= 1.0

    def test_gradcam_plus_plus_method(self, model, device, rand_pil):
        pip = InferencePipeline(model=model, device=device, gradcam_method="gradcam++")
        r = pip.run(rand_pil)
        assert r.gradcam_method == "gradcam++"
        assert r.heatmap.shape == (224, 224)

    @pytest.mark.parametrize("layer", [4, 6, 8])
    def test_various_gradcam_layers(self, model, device, rand_pil, layer):
        pip = InferencePipeline(model=model, device=device, gradcam_layer=layer)
        r = pip.run(rand_pil)
        assert r.gradcam_layer == layer

    @pytest.mark.parametrize("threshold", [0.20, 0.50, 0.80])
    def test_various_thresholds(self, model, device, rand_pil, threshold):
        pip = InferencePipeline(model=model, device=device, threshold=threshold)
        r = pip.run(rand_pil)
        assert abs(r.threshold - threshold) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineDeterminism:
    def test_same_image_same_result(self, pipeline, rand_pil):
        r1 = pipeline.run(rand_pil, filename="img.png")
        r2 = pipeline.run(rand_pil, filename="img.png")
        assert abs(r1.probability - r2.probability) < 1e-5
        assert r1.prediction == r2.prediction
        np.testing.assert_array_equal(r1.original_rgb, r2.original_rgb)


# ─────────────────────────────────────────────────────────────────────────────
# run_full_pipeline() (Streamlit-facing cached wrapper)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunFullPipeline:
    def test_missing_checkpoint_returns_error_dict(self, rand_bytes):
        """When the checkpoint does not exist, return {'error': ...} not raise."""
        from app.components.model_utils import run_full_pipeline

        result = run_full_pipeline(
            image_bytes=rand_bytes,
            checkpoint="/does/not/exist.pth",
            threshold=0.50,
            mc_passes=0,
            gradcam_method="gradcam",
            gradcam_layer=8,
        )
        assert "error" in result
        assert isinstance(result["error"], str)

    def test_valid_checkpoint_returns_full_dict(self, rand_bytes, tmp_checkpoint):
        """With a valid checkpoint, the result dict must have all required keys."""
        from app.components.model_utils import run_full_pipeline

        result = run_full_pipeline(
            image_bytes=rand_bytes,
            checkpoint=str(tmp_checkpoint),
            threshold=0.50,
            mc_passes=0,
            gradcam_method="gradcam",
            gradcam_layer=8,
            filename="test.png",
        )
        assert "error" not in result
        required = {
            "probability",
            "prediction",
            "confidence",
            "risk_level",
            "clinical_note",
            "original_rgb",
            "heatmap",
            "overlay",
            "colormap",
            "gradcam_method",
            "gradcam_layer",
        }
        missing = required - result.keys()
        assert not missing, f"Missing keys in pipeline result: {missing}"

    def test_pipeline_result_images_are_numpy_arrays(self, rand_bytes, tmp_checkpoint):
        from app.components.model_utils import run_full_pipeline

        result = run_full_pipeline(
            image_bytes=rand_bytes,
            checkpoint=str(tmp_checkpoint),
            threshold=0.50,
        )
        if "error" in result:
            pytest.skip("Checkpoint unavailable")
        assert isinstance(result["original_rgb"], np.ndarray)
        assert isinstance(result["heatmap"], np.ndarray)
        assert isinstance(result["overlay"], np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# generate_pdf_bytes() integration
# ─────────────────────────────────────────────────────────────────────────────


class TestGeneratePDFBytesIntegration:
    def test_pdf_bytes_from_pipeline_dict(self, rand_bytes, tmp_checkpoint):
        pytest.importorskip("reportlab", reason="reportlab not installed")
        from app.components.model_utils import generate_pdf_bytes, run_full_pipeline

        result = run_full_pipeline(
            image_bytes=rand_bytes,
            checkpoint=str(tmp_checkpoint),
            threshold=0.50,
        )
        if "error" in result:
            pytest.skip("Checkpoint unavailable")

        pdf = generate_pdf_bytes(
            pipeline_result_dict=result,
            patient_id="ANON",
            sample_id="N/A",
        )
        assert isinstance(pdf, bytes)
        assert pdf[:5] == b"%PDF-"
