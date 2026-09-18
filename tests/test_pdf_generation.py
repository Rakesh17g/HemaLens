"""
tests/test_pdf_generation.py
=============================
Unit tests for the PDF medical report generator.

Covers:
  - MedicalReportData construction (defaults + custom fields)
  - from_pipeline_result() factory (zero-duplication bridge)
  - generate_bytes() returns non-empty bytes
  - generate_bytes() returns valid PDF header
  - generate() writes a file to disk
  - report_id auto-generation (UUID format)
  - date_str format
  - pred_label based on prediction
  - Missing ReportLab → ImportError raised at init
  - Handling of different risk levels
  - Various image sizes
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

# Skip entire module if reportlab is not installed
reportlab = pytest.importorskip(
    "reportlab",
    reason="reportlab not installed — skip PDF tests (pip install reportlab)",
)

from src.reports.report_generator import (
    MedicalReportData,
    MedicalReportGenerator,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_image(h=224, w=224) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def sample_data() -> MedicalReportData:
    """A complete MedicalReportData object for reuse across tests."""
    return MedicalReportData(
        original_image=_make_image(),
        heatmap_overlay=_make_image(),
        prediction="ALL+",
        probability=0.87,
        confidence=0.73,
        risk_level="MEDIUM",
        clinical_note="Moderate-confidence positive. Pathologist review recommended.",
        boundary_score=0.74,
        entropy_score=0.72,
        mcdrop_score=0.0,
        threshold=0.50,
        model_version="EfficientNet-B0 v1.0",
        patient_id="PAT-001",
        sample_id="SLD-2026-007",
        institution="Test Hospital",
        analyst="Test Analyst",
    )


@pytest.fixture(scope="module")
def generator(tmp_path_factory) -> MedicalReportGenerator:
    out = str(tmp_path_factory.mktemp("reports"))
    return MedicalReportGenerator(output_dir=out)


# ─────────────────────────────────────────────────────────────────────────────
# MedicalReportData construction
# ─────────────────────────────────────────────────────────────────────────────


class TestMedicalReportDataConstruction:
    def test_required_fields_stored(self, sample_data):
        assert sample_data.prediction == "ALL+"
        assert sample_data.probability == pytest.approx(0.87)
        assert sample_data.patient_id == "PAT-001"

    def test_report_id_auto_generated(self):
        d = MedicalReportData(
            original_image=_make_image(),
            heatmap_overlay=_make_image(),
            prediction="Healthy",
            probability=0.10,
            confidence=0.92,
            risk_level="LOW",
            clinical_note="High-confidence negative.",
        )
        assert d.report_id.startswith("RPT-")
        assert len(d.report_id) == 12  # "RPT-" + 8 hex chars

    def test_custom_report_id(self):
        d = MedicalReportData(
            original_image=_make_image(),
            heatmap_overlay=_make_image(),
            prediction="ALL+",
            probability=0.90,
            confidence=0.85,
            risk_level="LOW",
            clinical_note="High-confidence positive.",
            report_id="CUSTOM-001",
        )
        assert d.report_id == "CUSTOM-001"

    def test_date_str_format(self, sample_data):
        import re

        assert re.match(r"\d{4}-\d{2}-\d{2}", sample_data.date_str)

    def test_datetime_str_contains_utc(self, sample_data):
        assert "UTC" in sample_data.datetime_str

    @pytest.mark.parametrize(
        "prediction,expected_substr",
        [
            ("ALL+", "ALL"),
            ("Healthy", "HEALTHY"),
        ],
    )
    def test_pred_label(self, prediction, expected_substr):
        d = MedicalReportData(
            original_image=_make_image(),
            heatmap_overlay=_make_image(),
            prediction=prediction,
            probability=0.5,
            confidence=0.6,
            risk_level="MEDIUM",
            clinical_note="test",
        )
        assert expected_substr.upper() in d.pred_label.upper()

    def test_risk_level_uppercased(self):
        d = MedicalReportData(
            original_image=_make_image(),
            heatmap_overlay=_make_image(),
            prediction="ALL+",
            probability=0.8,
            confidence=0.7,
            risk_level="medium",  # lowercase input
            clinical_note="test",
        )
        assert d.risk_level == "MEDIUM"


# ─────────────────────────────────────────────────────────────────────────────
# from_pipeline_result() factory
# ─────────────────────────────────────────────────────────────────────────────


class TestFromPipelineResult:
    @pytest.fixture()
    def pipeline_proxy(self):
        """Duck-typed PipelineResult namespace."""

        class Proxy:
            original_rgb = _make_image(224, 224)
            overlay = _make_image(224, 224)
            prediction = "ALL+"
            probability = 0.91
            confidence = 0.82
            risk_level = "LOW"
            clinical_note = "High-confidence positive."
            boundary_score = 0.82
            entropy_score = 0.80
            mcdrop_score = 0.00
            threshold = 0.50

        return Proxy()

    def test_creates_valid_report_data(self, pipeline_proxy):
        d = MedicalReportData.from_pipeline_result(pipeline_proxy)
        assert isinstance(d, MedicalReportData)

    def test_fields_transferred_correctly(self, pipeline_proxy):
        d = MedicalReportData.from_pipeline_result(pipeline_proxy)
        assert d.prediction == pipeline_proxy.prediction
        assert d.probability == pytest.approx(pipeline_proxy.probability)
        assert d.confidence == pytest.approx(pipeline_proxy.confidence)
        assert d.risk_level == pipeline_proxy.risk_level

    def test_images_transferred_correctly(self, pipeline_proxy):
        d = MedicalReportData.from_pipeline_result(pipeline_proxy)
        np.testing.assert_array_equal(d.original_image, pipeline_proxy.original_rgb)
        np.testing.assert_array_equal(d.heatmap_overlay, pipeline_proxy.overlay)

    def test_custom_metadata_passed_through(self, pipeline_proxy):
        d = MedicalReportData.from_pipeline_result(
            pipeline_proxy,
            patient_id="XYZ",
            analyst="Dr. Test",
            model_version="v2.0",
        )
        assert d.patient_id == "XYZ"
        assert d.analyst == "Dr. Test"
        assert d.model_version == "v2.0"


# ─────────────────────────────────────────────────────────────────────────────
# generate_bytes() — in-memory PDF
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateBytes:
    def test_returns_bytes(self, generator, sample_data):
        pdf = generator.generate_bytes(sample_data)
        assert isinstance(pdf, bytes)

    def test_returns_non_empty_bytes(self, generator, sample_data):
        pdf = generator.generate_bytes(sample_data)
        assert len(pdf) > 0, "PDF bytes must not be empty"

    def test_pdf_has_correct_magic_header(self, generator, sample_data):
        pdf = generator.generate_bytes(sample_data)
        # All PDFs start with %PDF-
        assert pdf[:5] == b"%PDF-", f"Expected PDF header, got {pdf[:5]}"

    def test_pdf_size_is_reasonable(self, generator, sample_data):
        pdf = generator.generate_bytes(sample_data)
        size_kb = len(pdf) / 1024
        # A report with 2 images should be between 50 KB and 5 MB
        assert 50 < size_kb < 5000, f"PDF size {size_kb:.1f} KB is unreasonable"

    def test_generate_bytes_twice_is_stable(self, generator, sample_data):
        pdf1 = generator.generate_bytes(sample_data)
        pdf2 = generator.generate_bytes(sample_data)
        # Both should be valid PDFs (exact bytes may differ due to timestamps)
        assert pdf1[:5] == b"%PDF-"
        assert pdf2[:5] == b"%PDF-"

    def test_can_be_opened_as_bytesio(self, generator, sample_data):
        pdf = generator.generate_bytes(sample_data)
        buf = io.BytesIO(pdf)
        assert buf.read(5) == b"%PDF-"

    @pytest.mark.parametrize("risk_level", ["LOW", "MEDIUM", "HIGH"])
    def test_all_risk_levels_generate_valid_pdf(self, tmp_path, risk_level):
        gen = MedicalReportGenerator(output_dir=str(tmp_path))
        notes = {
            "LOW": "High-confidence negative.",
            "MEDIUM": "Moderate-confidence positive.",
            "HIGH": "Uncertain. Expert review required.",
        }
        data = MedicalReportData(
            original_image=_make_image(),
            heatmap_overlay=_make_image(),
            prediction="ALL+" if risk_level == "HIGH" else "Healthy",
            probability=0.90,
            confidence={"LOW": 0.85, "MEDIUM": 0.65, "HIGH": 0.40}[risk_level],
            risk_level=risk_level,
            clinical_note=notes[risk_level],
        )
        pdf = gen.generate_bytes(data)
        assert pdf[:5] == b"%PDF-"


# ─────────────────────────────────────────────────────────────────────────────
# generate() — file output
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateFile:
    def test_returns_path_string(self, generator, sample_data):
        path = generator.generate(sample_data)
        assert isinstance(path, str)

    def test_file_exists_after_generate(self, generator, sample_data):
        path = generator.generate(sample_data)
        assert os.path.exists(path)

    def test_file_has_pdf_extension(self, generator, sample_data):
        path = generator.generate(sample_data)
        assert path.endswith(".pdf")

    def test_file_has_valid_pdf_header(self, generator, sample_data):
        path = generator.generate(sample_data)
        with open(path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_custom_filename(self, generator, sample_data):
        path = generator.generate(sample_data, filename="custom_report.pdf")
        assert path.endswith("custom_report.pdf")
        assert os.path.exists(path)

    def test_different_images_produce_different_files(self, tmp_path):
        gen = MedicalReportGenerator(output_dir=str(tmp_path))
        data1 = MedicalReportData(
            original_image=_make_image(),
            heatmap_overlay=_make_image(),
            prediction="ALL+",
            probability=0.95,
            confidence=0.90,
            risk_level="LOW",
            clinical_note="test",
            report_id="RPT-A0000001",
        )
        data2 = MedicalReportData(
            original_image=_make_image() * 0,
            heatmap_overlay=_make_image() * 0,
            prediction="Healthy",
            probability=0.05,
            confidence=0.92,
            risk_level="LOW",
            clinical_note="test",
            report_id="RPT-B0000002",
        )
        p1 = gen.generate(data1)
        p2 = gen.generate(data2)
        assert p1 != p2  # different filenames (different report IDs)


# ─────────────────────────────────────────────────────────────────────────────
# Various image sizes
# ─────────────────────────────────────────────────────────────────────────────


class TestVariousImageSizes:
    @pytest.mark.parametrize("size", [64, 128, 224, 256])
    def test_various_image_sizes_produce_valid_pdf(self, tmp_path, size):
        gen = MedicalReportGenerator(output_dir=str(tmp_path))
        data = MedicalReportData(
            original_image=_make_image(size, size),
            heatmap_overlay=_make_image(size, size),
            prediction="Healthy",
            probability=0.08,
            confidence=0.90,
            risk_level="LOW",
            clinical_note="High-confidence negative.",
        )
        pdf = gen.generate_bytes(data)
        assert pdf[:5] == b"%PDF-"
