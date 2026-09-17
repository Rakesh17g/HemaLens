"""
Medical Report Generator
=========================
Generates a professional, single-page PDF diagnostic report for a single
blood-smear cell image analysed by the ALL Detection system.

Report layout (A4 portrait)
─────────────────────────────
┌──────────────────────────────────────────────────────────┐
│  HEADER: Logo area | Report title | Date + Report ID     │
├─────────────────────────────┬────────────────────────────┤
│  CELL IMAGE (original)      │  HEATMAP (Grad-CAM overlay)│
│  224×224 px microscopy      │  Attention regions         │
├─────────────────────────────┴────────────────────────────┤
│  DIAGNOSIS PANEL                                         │
│  ┌─────────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │  PREDICTION     │  │ PROBABILITY │  │  RISK LEVEL │  │
│  │  ALL+ / Healthy │  │    97.3 %   │  │  LOW / MED  │  │
│  └─────────────────┘  └─────────────┘  └─────────────┘  │
├──────────────────────────────────────────────────────────┤
│  CONFIDENCE BREAKDOWN (horizontal bar chart)             │
├──────────────────────────────────────────────────────────┤
│  TECHNICAL DETAILS TABLE                                 │
│  Model | Version | Threshold | Date | Report ID         │
├──────────────────────────────────────────────────────────┤
│  CLINICAL GUIDANCE                                       │
├──────────────────────────────────────────────────────────┤
│  DISCLAIMER (legal / regulatory)                        │
└──────────────────────────────────────────────────────────┘

Dependencies
------------
  pip install reportlab pillow opencv-python-headless

Why ReportLab (not matplotlib savefig)?
  ReportLab produces true vector PDF with proper fonts, table layout,
  and embeds images at native DPI — exactly what healthcare institutions
  require for archiving and printing.
"""

from __future__ import annotations

import io
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("ALLReport")

# ── ReportLab imports ────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate,
        Paragraph, Spacer, Table, TableStyle, Image as RLImage,
        HRFlowable, KeepTogether,
    )
    from reportlab.platypus.flowables import Flowable
    from reportlab.graphics.shapes import Drawing, Rect, String, Circle
    from reportlab.graphics import renderPDF
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False
    logger.warning(
        "ReportLab not installed.  Install with:  pip install reportlab\n"
        "PDF export will not be available."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────

class _C:
    """Colour palette — medical dark-blue brand."""
    NAVY        = colors.HexColor("#0B1A2F")   # header background
    DARK_BLUE   = colors.HexColor("#1A3557")   # section headers
    MID_BLUE    = colors.HexColor("#2563EB")   # accent
    LIGHT_BLUE  = colors.HexColor("#EFF6FF")   # row alternates
    WHITE       = colors.white
    NEAR_WHITE  = colors.HexColor("#F8FAFC")
    SLATE       = colors.HexColor("#64748B")   # subtext
    DARK        = colors.HexColor("#1E293B")   # body text

    # Prediction
    ALL_POS     = colors.HexColor("#DC2626")   # red   — ALL+
    HEALTHY     = colors.HexColor("#16A34A")   # green — Healthy

    # Risk
    RISK_LOW    = colors.HexColor("#15803D")
    RISK_MED    = colors.HexColor("#D97706")
    RISK_HIGH   = colors.HexColor("#DC2626")
    RISK_BG_LOW = colors.HexColor("#DCFCE7")
    RISK_BG_MED = colors.HexColor("#FEF3C7")
    RISK_BG_HI  = colors.HexColor("#FEE2E2")

    GRID        = colors.HexColor("#CBD5E1")
    DIVIDER     = colors.HexColor("#94A3B8")


def _risk_color(risk: str) -> Tuple:
    """Return (fg, bg) colour pair for a risk level string."""
    return {
        "LOW":    (_C.RISK_LOW,  _C.RISK_BG_LOW),
        "MEDIUM": (_C.RISK_MED,  _C.RISK_BG_MED),
        "HIGH":   (_C.RISK_HIGH, _C.RISK_BG_HI),
    }.get(risk.upper(), (_C.DARK, _C.NEAR_WHITE))


def _pred_color(prediction: str):
    return _C.ALL_POS if "ALL" in prediction.upper() else _C.HEALTHY


# ─────────────────────────────────────────────────────────────────────────────
# Custom flowable: colour badge
# ─────────────────────────────────────────────────────────────────────────────

class _Badge(Flowable):
    """A rounded-rect badge with centred text."""
    def __init__(
        self,
        text:    str,
        width:   float,
        height:  float,
        fg:      Any,
        bg:      Any,
        font:    str  = "Helvetica-Bold",
        font_sz: int  = 14,
        radius:  float = 6,
    ):
        super().__init__()
        self.text    = text
        self.width   = width
        self.height  = height
        self.fg      = fg
        self.bg      = bg
        self.font    = font
        self.font_sz = font_sz
        self.radius  = radius

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.setStrokeColor(self.fg)
        c.setLineWidth(1.5)
        c.roundRect(0, 0, self.width, self.height, self.radius, fill=1, stroke=1)
        c.setFillColor(self.fg)
        c.setFont(self.font, self.font_sz)
        c.drawCentredString(self.width / 2, self.height / 2 - self.font_sz * 0.36,
                            self.text)


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ndarray_to_rl_image(
    arr: np.ndarray,
    width: float,
    height: float,
) -> "RLImage":
    """
    Convert a (H,W,3) uint8 numpy array to a ReportLab Image flowable
    via in-memory PNG — no temp file needed.
    """
    from PIL import Image as PilImage
    pil = PilImage.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width, height=height)


def _path_to_rl_image(
    path: str,
    width: float,
    height: float,
) -> "RLImage":
    """Load an image from disk into a ReportLab Image."""
    return RLImage(path, width=width, height=height)


# ─────────────────────────────────────────────────────────────────────────────
# MedicalReportData — everything the report needs
# ─────────────────────────────────────────────────────────────────────────────

class MedicalReportData:
    """
    All inputs required to generate a medical report.

    Args
    ----
    original_image  : (H, W, 3) uint8 RGB  — original cell image.
    heatmap_overlay : (H, W, 3) uint8 RGB  — Grad-CAM overlay.
    prediction      : str  — "ALL+" or "Healthy".
    probability     : float — sigmoid output ∈ [0, 1].
    confidence      : float — confidence score ∈ [0, 1].
    risk_level      : str  — "LOW" | "MEDIUM" | "HIGH".
    clinical_note   : str  — one-sentence guidance.
    boundary_score  : float — confidence component.
    entropy_score   : float — confidence component.
    mcdrop_score    : float — confidence component (0 if disabled).
    threshold       : float — decision boundary.
    model_version   : str  — e.g. "EfficientNet-B0 v1.2.0".
    report_id       : str  — auto-generated UUID4 if None.
    patient_id      : str  — anonymised ID (optional).
    sample_id       : str  — slide/sample reference (optional).
    institution     : str  — reporting institution name.
    analyst         : str  — reviewing analyst/physician name.
    timestamp       : datetime — report generation time (UTC now if None).
    """

    def __init__(
        self,
        original_image:   np.ndarray,
        heatmap_overlay:  np.ndarray,
        prediction:       str,
        probability:      float,
        confidence:       float,
        risk_level:       str,
        clinical_note:    str,
        boundary_score:   float = 0.0,
        entropy_score:    float = 0.0,
        mcdrop_score:     float = 0.0,
        threshold:        float = 0.50,
        model_version:    str   = "EfficientNet-B0 v1.0",
        report_id:        Optional[str]      = None,
        patient_id:       str   = "ANON",
        sample_id:        str   = "N/A",
        institution:      str   = "ALL Detection AI System",
        analyst:          str   = "AI Diagnostic Assistant",
        timestamp:        Optional[datetime] = None,
    ) -> None:
        self.original_image  = original_image
        self.heatmap_overlay = heatmap_overlay
        self.prediction      = prediction
        self.probability     = probability
        self.confidence      = confidence
        self.risk_level      = risk_level.upper()
        self.clinical_note   = clinical_note
        self.boundary_score  = boundary_score
        self.entropy_score   = entropy_score
        self.mcdrop_score    = mcdrop_score
        self.threshold       = threshold
        self.model_version   = model_version
        self.report_id       = report_id or f"RPT-{uuid.uuid4().hex[:8].upper()}"
        self.patient_id      = patient_id
        self.sample_id       = sample_id
        self.institution     = institution
        self.analyst         = analyst
        self.timestamp       = timestamp or datetime.now(timezone.utc)

    @property
    def date_str(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d")

    @property
    def datetime_str(self) -> str:
        return self.timestamp.strftime("%Y-%m-%d  %H:%M UTC")

    @property
    def pred_label(self) -> str:
        """Formatted prediction with emoji."""
        return ("🔴 ALL POSITIVE (ALL+)"
                if "ALL" in self.prediction.upper()
                else "🟢 HEALTHY (No ALL Detected)")

    @classmethod
    def from_pipeline_result(
        cls,
        result,                        # PipelineResult (avoid circular import)
        patient_id:   str = "ANON",
        sample_id:    str = "N/A",
        institution:  str = "ALL Detection AI System",
        analyst:      str = "AI Diagnostic Assistant",
        model_version: str = "EfficientNet-B0 v1.0",
        report_id:    Optional[str] = None,
    ) -> "MedicalReportData":
        """
        Build MedicalReportData directly from a PipelineResult.

        This is the canonical bridge between InferencePipeline output
        and the PDF report — zero field duplication.

        Args
        ----
        result : src.inference.pipeline.PipelineResult
        """
        return cls(
            original_image  = result.original_rgb,
            heatmap_overlay = result.overlay,
            prediction      = result.prediction,
            probability     = result.probability,
            confidence      = result.confidence,
            risk_level      = result.risk_level,
            clinical_note   = result.clinical_note,
            boundary_score  = result.boundary_score,
            entropy_score   = result.entropy_score,
            mcdrop_score    = result.mcdrop_score,
            threshold       = result.threshold,
            model_version   = model_version,
            report_id       = report_id,
            patient_id      = patient_id,
            sample_id       = sample_id,
            institution     = institution,
            analyst         = analyst,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MedicalReportGenerator
# ─────────────────────────────────────────────────────────────────────────────

class MedicalReportGenerator:
    """
    Generates a professional A4 PDF diagnostic report.

    Args
    ----
    output_dir : Directory where PDFs will be saved.
    institution : Overrides the default institution name in the header.

    Usage
    -----
    >>> gen    = MedicalReportGenerator(output_dir="logs/reports")
    >>> data   = MedicalReportData(original_image, heatmap_overlay, ...)
    >>> path   = gen.generate(data)
    >>> print(f"Report saved: {path}")
    """

    VERSION   = "1.0.0"
    SYSTEM    = "ALL Detection AI System"
    SUBTITLE  = "Acute Lymphoblastic Leukaemia Screening Report"

    def __init__(
        self,
        output_dir:  str = "logs/reports",
        institution: Optional[str] = None,
    ) -> None:
        if not _HAS_REPORTLAB:
            raise ImportError(
                "ReportLab is required for PDF export.\n"
                "Install with:  pip install reportlab"
            )
        self.output_dir  = output_dir
        self.institution = institution or self.SYSTEM
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"MedicalReportGenerator ready | output_dir={output_dir}")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        data:     MedicalReportData,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate and save the PDF report to disk.

        Returns
        -------
        Absolute path to the saved PDF.
        """
        fname    = filename or f"{data.report_id}_{data.date_str}.pdf"
        out_path = os.path.join(self.output_dir, fname)

        doc   = self._build_doc(out_path, data)
        story = self._build_story(data)
        doc.build(story, onFirstPage=self._page_decorator(data),
                  onLaterPages=self._page_decorator(data))

        abs_path = os.path.abspath(out_path)
        logger.info(f"  Report saved: {abs_path}")
        return abs_path

    def generate_bytes(
        self,
        data: MedicalReportData,
    ) -> bytes:
        """
        Generate the PDF entirely in memory and return raw bytes.

        No file is written to disk.  Use this for Streamlit's
        ``st.download_button(data=pdf_bytes, mime='application/pdf')``.

        Returns
        -------
        bytes : Complete PDF content.
        """
        buf   = io.BytesIO()
        doc   = self._build_doc(buf, data)
        story = self._build_story(data)
        doc.build(story, onFirstPage=self._page_decorator(data),
                  onLaterPages=self._page_decorator(data))
        return buf.getvalue()

    # ── Document template ─────────────────────────────────────────────────────

    def _build_doc(self, path: str, data: MedicalReportData):
        from reportlab.platypus import SimpleDocTemplate
        page_w, page_h = A4
        margin         = 1.8 * cm

        doc = SimpleDocTemplate(
            path,
            pagesize    = A4,
            leftMargin  = margin,
            rightMargin = margin,
            topMargin   = 3.8 * cm,   # room for header banner
            bottomMargin= 2.8 * cm,   # room for footer + disclaimer
            title       = f"ALL Detection Report — {data.report_id}",
            author      = data.analyst,
            subject     = "Acute Lymphoblastic Leukaemia AI Screening",
            creator     = f"{self.SYSTEM} v{self.VERSION}",
        )

        return doc

    # ── Page decorators (header / footer) ─────────────────────────────────────

    def _page_decorator(self, data: MedicalReportData):
        """Returns an onPage callback that draws header + footer."""

        def on_page(canv: "canvas.Canvas", doc) -> None:
            page_w, page_h = A4
            canv.saveState()

            # ── Header ────────────────────────────────────────────────────────
            hdr_h = 3.2 * cm
            canv.setFillColor(_C.NAVY)
            canv.rect(0, page_h - hdr_h, page_w, hdr_h, fill=1, stroke=0)

            # ── Left: system name ───────────────────────────────────────────
            canv.setFillColor(_C.WHITE)
            canv.setFont("Helvetica-Bold", 14)
            canv.drawString(1.8 * cm, page_h - 1.6 * cm, self.SYSTEM)
            canv.setFont("Helvetica", 9)
            canv.setFillColor(colors.HexColor("#94A3B8"))
            canv.drawString(1.8 * cm, page_h - 2.2 * cm, self.SUBTITLE)

            # ── Right: report meta ──────────────────────────────────────────
            canv.setFillColor(_C.WHITE)
            canv.setFont("Helvetica-Bold", 8)
            right_x = page_w - 1.8 * cm
            canv.drawRightString(right_x, page_h - 1.35 * cm,
                                 f"Report ID: {data.report_id}")
            canv.setFont("Helvetica", 8)
            canv.setFillColor(colors.HexColor("#CBD5E1"))
            canv.drawRightString(right_x, page_h - 1.85 * cm,
                                 f"Date: {data.datetime_str}")
            canv.drawRightString(right_x, page_h - 2.25 * cm,
                                 f"Patient: {data.patient_id}  |  Sample: {data.sample_id}")

            # ── Blue accent stripe under header ─────────────────────────────
            canv.setFillColor(_C.MID_BLUE)
            canv.rect(0, page_h - hdr_h - 3, page_w, 3, fill=1, stroke=0)

            # ── Footer ────────────────────────────────────────────────────────
            canv.setFillColor(colors.HexColor("#F1F5F9"))
            canv.rect(0, 0, page_w, 2.4 * cm, fill=1, stroke=0)
            canv.setStrokeColor(_C.GRID)
            canv.setLineWidth(0.5)
            canv.line(0, 2.4 * cm, page_w, 2.4 * cm)

            canv.setFillColor(_C.SLATE)
            canv.setFont("Helvetica", 7)
            disc = (
                "FOR RESEARCH USE ONLY. NOT VALIDATED FOR CLINICAL DIAGNOSIS. "
                "AI predictions must be confirmed by a licensed pathologist."
            )
            canv.drawCentredString(page_w / 2, 1.6 * cm, disc)
            canv.setFont("Helvetica", 7)
            canv.drawCentredString(
                page_w / 2, 1.1 * cm,
                f"Model: {data.model_version}  |  Confidence threshold: {data.threshold:.2f}  |  "
                f"Generated by {self.SYSTEM} v{self.VERSION}",
            )

            # Page number
            canv.drawRightString(
                page_w - 1.5 * cm, 0.6 * cm,
                f"Page {doc.page}"
            )

            canv.restoreState()

        return on_page

    # ── Story (content) ───────────────────────────────────────────────────────

    def _build_story(self, data: MedicalReportData) -> list:
        styles = self._styles()
        story  = []

        # ── Section 1: Images side-by-side ────────────────────────────────────
        story.append(Paragraph("Cell Image Analysis", styles["section_h"]))
        story.append(Spacer(1, 0.15 * cm))

        img_w = 7.2 * cm
        img_h = 7.2 * cm
        orig_rl = _ndarray_to_rl_image(data.original_image,  img_w, img_h)
        heat_rl = _ndarray_to_rl_image(data.heatmap_overlay, img_w, img_h)

        cap_style = styles["caption"]
        img_table = Table(
            [[orig_rl,                  heat_rl],
             [Paragraph("Original Cell Image<br/>"
                        "<font color='#64748B' size='7'>"
                        "Peripheral blood smear — Wright-Giemsa stain</font>",
                        cap_style),
              Paragraph("Grad-CAM Attention Heatmap<br/>"
                        "<font color='#64748B' size='7'>"
                        "Red = high activation (model focus region)</font>",
                        cap_style)]],
            colWidths=[img_w + 1.0 * cm, img_w + 1.0 * cm],
        )
        img_table.setStyle(TableStyle([
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_C.NEAR_WHITE, _C.LIGHT_BLUE]),
            ("BOX",        (0, 0), (-1, -1), 0.5, _C.GRID),
            ("INNERGRID",  (0, 0), (-1, -1), 0.3, _C.GRID),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 0.3 * cm))

        # ── Section 2: Diagnosis panel ────────────────────────────────────────
        story.append(Paragraph("Diagnostic Result", styles["section_h"]))
        story.append(Spacer(1, 0.15 * cm))

        pred_col  = _pred_color(data.prediction)
        risk_fg, risk_bg = _risk_color(data.risk_level)
        risk_label = {"LOW": "LOW RISK", "MEDIUM": "MEDIUM RISK",
                      "HIGH": "HIGH RISK"}.get(data.risk_level, data.risk_level)

        diag_data = [
            [
                Paragraph(
                    f"<font color='#1E293B' size='8'><b>DIAGNOSIS</b></font><br/>"
                    f"<font size='18'><b><font color='{pred_col.hexval()}'>"
                    f"{data.prediction}</font></b></font>",
                    styles["diag_cell"]),
                Paragraph(
                    f"<font color='#1E293B' size='8'><b>PROBABILITY</b></font><br/>"
                    f"<font size='22'><b><font color='{pred_col.hexval()}'>"
                    f"{data.probability*100:.1f}%</font></b></font>",
                    styles["diag_cell"]),
                Paragraph(
                    f"<font color='#1E293B' size='8'><b>CONFIDENCE</b></font><br/>"
                    f"<font size='22'><b><font color='#2563EB'>"
                    f"{data.confidence*100:.1f}%</font></b></font>",
                    styles["diag_cell"]),
                Paragraph(
                    f"<font color='#1E293B' size='8'><b>RISK LEVEL</b></font><br/>"
                    f"<font size='16'><b><font color='{risk_fg.hexval()}'>"
                    f"{risk_label}</font></b></font>",
                    styles["diag_cell"]),
            ]
        ]
        avail_w = A4[0] - 3.6 * cm
        col_w   = avail_w / 4
        diag_tbl = Table(diag_data, colWidths=[col_w] * 4)
        diag_tbl.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND",    (0, 0), (0, 0),   _C.NEAR_WHITE),
            ("BACKGROUND",    (1, 0), (1, 0),   _C.NEAR_WHITE),
            ("BACKGROUND",    (2, 0), (2, 0),   _C.LIGHT_BLUE),
            ("BACKGROUND",    (3, 0), (3, 0),   risk_bg),
            ("BOX",           (0, 0), (-1, -1), 0.5, _C.GRID),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, _C.GRID),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(diag_tbl)
        story.append(Spacer(1, 0.3 * cm))

        # ── Section 3: Confidence breakdown ───────────────────────────────────
        story.append(Paragraph("Confidence Breakdown", styles["section_h"]))
        story.append(Spacer(1, 0.15 * cm))

        def _bar_row(label: str, value: float, color_hex: str, note: str):
            bar_full = 8.0 * cm
            bar_fill = bar_full * max(0.0, min(1.0, value))
            bar_cell = (
                f'<font color="{color_hex}">{"█" * int(value * 40)}'
                f'<font color="#CBD5E1">{"░" * (40 - int(value * 40))}</font></font>'
            )
            return [
                Paragraph(f"<b>{label}</b>", styles["tbl_label"]),
                Paragraph(bar_cell,          styles["tbl_mono"]),
                Paragraph(f"<b><font color='{color_hex}'>{value:.3f}</font></b>",
                          styles["tbl_val"]),
                Paragraph(f"<font color='#64748B' size='7.5'>{note}</font>",
                          styles["tbl_note"]),
            ]

        conf_rows = [
            [Paragraph("<b>Component</b>",   styles["tbl_hdr"]),
             Paragraph("<b>Score Bar</b>",   styles["tbl_hdr"]),
             Paragraph("<b>Score</b>",       styles["tbl_hdr"]),
             Paragraph("<b>Description</b>", styles["tbl_hdr"])],
            _bar_row("Boundary Distance", data.boundary_score,   "#6366F1",
                     f"|p − t| norm. | p={data.probability:.3f}, t={data.threshold:.2f}"),
            _bar_row("Predictive Entropy", data.entropy_score,   "#22D3EE",
                     "1 − H(p)/ln(2) | Low entropy = more certain"),
            _bar_row("MC Dropout",         data.mcdrop_score,    "#F59E0B",
                     "Variance over T stochastic passes (0=disabled)"),
            _bar_row("COMBINED CONFIDENCE", data.confidence,      "#2563EB",
                     "Weighted sum of all three signals"),
        ]
        cw = [3.2 * cm, 5.5 * cm, 1.8 * cm, 6.5 * cm]
        conf_tbl = Table(conf_rows, colWidths=cw)
        conf_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),   _C.DARK_BLUE),
            ("TEXTCOLOR",     (0, 0), (-1, 0),   _C.WHITE),
            ("BACKGROUND",    (0, 4), (-1, 4),   _C.LIGHT_BLUE),
            ("ROWBACKGROUNDS",(0, 1), (-1, 3),   [_C.NEAR_WHITE, _C.WHITE]),
            ("ALIGN",         (0, 0), (-1, -1),  "LEFT"),
            ("ALIGN",         (2, 0), (2, -1),   "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1),  "MIDDLE"),
            ("BOX",           (0, 0), (-1, -1),  0.5, _C.GRID),
            ("INNERGRID",     (0, 0), (-1, -1),  0.3, _C.GRID),
            ("TOPPADDING",    (0, 0), (-1, -1),  5),
            ("BOTTOMPADDING", (0, 0), (-1, -1),  5),
            ("LEFTPADDING",   (0, 0), (-1, -1),  6),
            ("FONTNAME",      (0, 4), (-1, 4),   "Helvetica-Bold"),
        ]))
        story.append(conf_tbl)
        story.append(Spacer(1, 0.3 * cm))

        # ── Section 4: Technical details ──────────────────────────────────────
        story.append(Paragraph("Technical Details", styles["section_h"]))
        story.append(Spacer(1, 0.15 * cm))

        technical = [
            ["Report ID",       data.report_id,       "Patient ID",     data.patient_id],
            ["Sample ID",       data.sample_id,        "Date / Time",    data.datetime_str],
            ["Model Version",   data.model_version,    "Decision Thr.",  f"{data.threshold:.3f}"],
            ["Institution",     data.institution,      "Analyst",        data.analyst],
            ["Prediction",      data.prediction,        "Risk Level",     data.risk_level],
        ]
        tech_tbl = Table(technical, colWidths=[3.5*cm, 6.3*cm, 3.5*cm, 4.0*cm])
        tbl_style = [
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_C.NEAR_WHITE, _C.WHITE]),
            ("FONTNAME",       (0, 0), (0, -1),  "Helvetica-Bold"),
            ("FONTNAME",       (2, 0), (2, -1),  "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1),  8.5),
            ("TEXTCOLOR",      (0, 0), (0, -1),  _C.DARK_BLUE),
            ("TEXTCOLOR",      (2, 0), (2, -1),  _C.DARK_BLUE),
            ("BOX",            (0, 0), (-1, -1),  0.5, _C.GRID),
            ("INNERGRID",      (0, 0), (-1, -1),  0.2, _C.GRID),
            ("TOPPADDING",     (0, 0), (-1, -1),  4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1),  4),
            ("LEFTPADDING",    (0, 0), (-1, -1),  7),
        ]
        tech_tbl.setStyle(TableStyle(tbl_style))
        story.append(tech_tbl)
        story.append(Spacer(1, 0.3 * cm))

        # ── Section 5: Clinical guidance ──────────────────────────────────────
        story.append(Paragraph("Clinical Guidance", styles["section_h"]))
        story.append(Spacer(1, 0.1 * cm))

        risk_fg2, risk_bg2 = _risk_color(data.risk_level)
        guidance_tbl = Table(
            [[Paragraph(
                f"<b><font color='{risk_fg2.hexval()}'>"
                f"[{data.risk_level} RISK]</font></b>  {data.clinical_note}",
                styles["body"]
            )]],
            colWidths=[A4[0] - 3.6 * cm],
        )
        guidance_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), risk_bg2),
            ("BOX",           (0, 0), (-1, -1), 1.0, risk_fg2),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        story.append(guidance_tbl)
        story.append(Spacer(1, 0.3 * cm))

        # ── Section 6: Disclaimer ─────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=_C.DIVIDER, spaceAfter=0.15 * cm))
        story.append(Paragraph("Regulatory Disclaimer", styles["section_h"]))
        story.append(Spacer(1, 0.1 * cm))

        disclaimer = (
            "This report is generated by an AI-assisted screening system and is "
            "<b>intended for research and educational purposes only</b>. "
            "It has <b>not been approved or cleared by the FDA, CE, or any regulatory "
            "authority</b> as a medical diagnostic device. "
            "<br/><br/>"
            "The predictions produced by this system <b>must not be used as the sole "
            "basis for clinical decision-making</b>. All findings must be independently "
            "reviewed and confirmed by a qualified medical professional (e.g., "
            "haematologist or clinical pathologist) before any treatment or management "
            "decisions are made. "
            "<br/><br/>"
            "This system is trained on the IDB2 dataset. Performance may vary on "
            "images acquired under different staining protocols, microscope settings, "
            "or patient populations. The confidence score provides an estimate of "
            "model uncertainty and does not reflect clinical certainty. "
            "<br/><br/>"
            "Patient privacy is the responsibility of the institution deploying this "
            "system. This report should be handled in accordance with applicable data "
            "protection regulations (e.g., HIPAA, GDPR)."
        )
        story.append(Paragraph(disclaimer, styles["disclaimer"]))

        return story

    # ── Styles ────────────────────────────────────────────────────────────────

    def _styles(self) -> Dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        s    = {}

        s["section_h"] = ParagraphStyle(
            "SectionHead",
            fontName  = "Helvetica-Bold",
            fontSize  = 10,
            textColor = _C.DARK_BLUE,
            spaceBefore = 4,
            spaceAfter  = 2,
            borderPad   = 3,
        )
        s["body"] = ParagraphStyle(
            "Body",
            fontName  = "Helvetica",
            fontSize  = 8.5,
            textColor = _C.DARK,
            leading   = 13,
        )
        s["caption"] = ParagraphStyle(
            "Caption",
            fontName  = "Helvetica",
            fontSize  = 8,
            textColor = _C.DARK,
            alignment = TA_CENTER,
            leading   = 12,
        )
        s["diag_cell"] = ParagraphStyle(
            "DiagCell",
            fontName  = "Helvetica",
            fontSize  = 9,
            textColor = _C.DARK,
            alignment = TA_CENTER,
            leading   = 16,
        )
        s["tbl_hdr"] = ParagraphStyle(
            "TblHdr",
            fontName  = "Helvetica-Bold",
            fontSize  = 8,
            textColor = _C.WHITE,
            alignment = TA_LEFT,
        )
        s["tbl_label"] = ParagraphStyle(
            "TblLabel",
            fontName  = "Helvetica-Bold",
            fontSize  = 8,
            textColor = _C.DARK,
        )
        s["tbl_val"] = ParagraphStyle(
            "TblVal",
            fontName  = "Helvetica-Bold",
            fontSize  = 9,
            alignment = TA_CENTER,
        )
        s["tbl_note"] = ParagraphStyle(
            "TblNote",
            fontName  = "Helvetica",
            fontSize  = 7.5,
            textColor = _C.SLATE,
            leading   = 11,
        )
        s["tbl_mono"] = ParagraphStyle(
            "TblMono",
            fontName  = "Courier",
            fontSize  = 7,
            leading   = 10,
        )
        s["disclaimer"] = ParagraphStyle(
            "Disclaimer",
            fontName  = "Helvetica",
            fontSize  = 7.5,
            textColor = _C.SLATE,
            leading   = 12,
            alignment = TA_JUSTIFY,
        )
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build_report()
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    model,
    image_tensor,
    confidence_result,       # ConfidenceResult
    output_dir:  str  = "logs/reports",
    patient_id:  str  = "ANON",
    sample_id:   str  = "N/A",
    institution: str  = "ALL Detection AI System",
    analyst:     str  = "AI Diagnostic Assistant",
    model_version: str = "EfficientNet-B0 v1.0",
    device       = None,
    gradcam_method: str = "gradcam",
) -> str:
    """
    One-call convenience function.

    Runs Grad-CAM, assembles MedicalReportData, and generates the PDF.

    Args
    ----
    model            : Trained EfficientNetB0.
    image_tensor     : (1,C,H,W) or (C,H,W) ImageNet-normalised tensor.
    confidence_result: ConfidenceResult from ConfidenceEstimator.estimate().
    output_dir       : Where to save the PDF.
    ... (see MedicalReportData for other args)

    Returns
    -------
    Absolute path to saved PDF.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.explainability.gradcam import GradCAM, _denorm_to_uint8

    device = device or next(model.parameters()).device

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    cam_engine = GradCAM(model, device=device, method=gradcam_method)
    cam_result = cam_engine(image_tensor)

    original_image  = cam_result.original    # (H,W,3) uint8 RGB
    heatmap_overlay = cam_result.overlay     # (H,W,3) uint8 RGB

    # ── Report data ───────────────────────────────────────────────────────────
    data = MedicalReportData(
        original_image  = original_image,
        heatmap_overlay = heatmap_overlay,
        prediction      = confidence_result.prediction,
        probability     = confidence_result.probability,
        confidence      = confidence_result.confidence,
        risk_level      = confidence_result.risk_level,
        clinical_note   = confidence_result.clinical_note,
        boundary_score  = confidence_result.boundary_score,
        entropy_score   = confidence_result.entropy_score,
        mcdrop_score    = confidence_result.mcdrop_score,
        threshold       = confidence_result.threshold,
        model_version   = model_version,
        patient_id      = patient_id,
        sample_id       = sample_id,
        institution     = institution,
        analyst         = analyst,
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    gen  = MedicalReportGenerator(output_dir=output_dir, institution=institution)
    path = gen.generate(data)
    return path
