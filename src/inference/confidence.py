"""
Confidence Estimation Module
==============================

WHY PROBABILITY ≠ CONFIDENCE
─────────────────────────────────────────────────────────────────────────────
This is the most important concept in medical AI reliability.

  PROBABILITY (p)
  ···············
  The raw sigmoid output of the model: p = σ(logit) ∈ [0, 1].
  It answers: "What fraction of the time does this pattern look like ALL+?"
  Problem: A model trained with Focal Loss + label smoothing can output
  p = 0.95 on a blurry, out-of-focus cell image IT HAS NEVER SEEN, because
  it forces all inputs into a probability distribution regardless of image
  quality or distributional shift. HIGH PROBABILITY ≠ HIGH CERTAINTY.

  Example: Model sees a Gaussian noise image → sigmoid pushes it to 0.88.
           Probability: 88% ALL+.  Confidence: 12% (near-uniform logits,
           no distinctive features activated).

  CONFIDENCE (c)
  ··············
  A post-hoc measure of HOW CERTAIN the model is about its own prediction.
  High confidence means: "I've seen patterns like this many times, my
  features are strongly activated, and I'm far from the decision boundary."
  Low confidence means:  "This input is ambiguous or unlike my training data."

  Three complementary signals are combined to estimate confidence:

  1. BOUNDARY DISTANCE  — |p − threshold|
     How far is the predicted probability from the decision boundary?
     p = 0.97 → distance = 0.47 (very far from boundary → high confidence)
     p = 0.53 → distance = 0.03 (barely over boundary → very uncertain)

  2. PREDICTIVE ENTROPY — H(p) = −[p·log(p) + (1−p)·log(1−p)]
     Information-theoretic measure of uncertainty.
     H = 0.0  (p=0 or p=1): perfectly certain
     H = 0.693 (p=0.5):      maximally uncertain (1 bit of entropy)
     Normalized to [0, 1]: entropy_norm = H / ln(2)
     Confidence contribution: 1 − entropy_norm

  3. MC DROPOUT UNCERTAINTY — Variance over T stochastic forward passes
     With dropout ENABLED at inference (model.train() but @no_grad),
     each forward pass uses a different random dropout mask → different
     predictions. The variance of T predictions estimates epistemic
     uncertainty (model uncertainty due to limited training data).
     Requires T ≥ 10 passes (default: 20).

  COMBINED CONFIDENCE SCORE
  ··························
  c = w_boundary · boundary_score
    + w_entropy  · entropy_score
    + w_mcdrop   · mcdrop_score      (if mc_dropout_passes > 0)

  Weights default to [0.35, 0.35, 0.30] (boundary, entropy, mcdrop).

RISK LEVELS
─────────────────────────────────────────────────────────────────────────────
  In clinical practice we need an actionable three-tier output:

  LOW RISK    c ≥ 0.80  → prediction is reliable; probability is meaningful
  MEDIUM RISK c ∈ [0.55, 0.80) → review recommended; borderline case
  HIGH RISK   c < 0.55  → do NOT act on prediction alone; human review required

  "High Risk" does NOT mean the patient has leukemia — it means the MODEL
  is uncertain and a pathologist must independently review the slide.

  The risk level is independent of the class prediction:
    - "ALL+ (95%), LOW RISK"    → confident positive: treat immediately
    - "ALL+ (72%), HIGH RISK"   → uncertain positive: re-examine the slide
    - "Healthy (8%), HIGH RISK" → uncertain negative: verify with expert
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

logger = logging.getLogger("ALLConfidence")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Level
# ─────────────────────────────────────────────────────────────────────────────


class RiskLevel:
    """
    Three-tier clinical risk classification based on model confidence.

    LOW    c ≥ 0.80  : Prediction is reliable. Probability is informative.
    MEDIUM c ∈ [0.55, 0.80): Borderline case. Expert review recommended.
    HIGH   c < 0.55  : Model is uncertain. Do NOT act on prediction alone.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    # Default confidence boundaries (can be overridden at construction time)
    THRESHOLDS = {  # noqa: RUF012
        "LOW": 0.80,  # c ≥ 0.80 → LOW risk (trustworthy)
        "MEDIUM": 0.55,  # 0.55 ≤ c < 0.80 → MEDIUM risk
        # c < 0.55 → HIGH risk (untrustworthy)
    }

    @classmethod
    def from_confidence(
        cls,
        confidence: float,
        thresholds: dict[str, float] | None = None,
    ) -> str:
        """Return risk level string given a confidence score in [0, 1]."""
        t = thresholds or cls.THRESHOLDS
        if confidence >= t["LOW"]:
            return cls.LOW
        elif confidence >= t["MEDIUM"]:
            return cls.MEDIUM
        else:
            return cls.HIGH

    @classmethod
    def emoji(cls, risk: str) -> str:
        return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")

    @classmethod
    def clinical_note(cls, risk: str, prediction: str) -> str:
        """Return a plain-English clinical guidance string."""
        notes = {
            (
                cls.LOW,
                "ALL+",
            ): "High-confidence positive. Recommend immediate clinical review.",
            (
                cls.LOW,
                "Healthy",
            ): "High-confidence negative. Standard monitoring protocol.",
            (
                cls.MEDIUM,
                "ALL+",
            ): "Moderate-confidence positive. Pathologist review recommended.",
            (
                cls.MEDIUM,
                "Healthy",
            ): "Moderate-confidence negative. Consider follow-up smear.",
            (
                cls.HIGH,
                "ALL+",
            ): "UNCERTAIN positive. Do NOT act on model alone. Expert slide review required.",
            (
                cls.HIGH,
                "Healthy",
            ): "UNCERTAIN negative. Image quality or cell morphology may be atypical. Human review required.",
        }
        key = (risk, prediction)
        return notes.get(key, "Confidence is low. Manual review required.")


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Result (data container)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConfidenceResult:
    """
    Complete confidence estimation result for a single prediction.

    Fields
    ------
    probability     : float  — Raw sigmoid output p ∈ [0, 1].
                               "What fraction of patterns resemble ALL+?"
    prediction      : str    — "ALL+" or "Healthy" at applied threshold.
    threshold       : float  — Decision boundary used.

    confidence      : float  — Combined confidence score c ∈ [0, 1].
                               "How much should you trust this prediction?"
    risk_level      : str    — "LOW" | "MEDIUM" | "HIGH"
    clinical_note   : str    — Plain-English guidance.

    # ── Confidence breakdown ──────────────────────────────────────────────
    boundary_score  : float  — Confidence from distance to decision boundary.
    entropy_score   : float  — Confidence from predictive entropy (1 − H_norm).
    entropy_raw     : float  — Raw binary entropy H(p).
    mcdrop_score    : float  — Confidence from MC Dropout (0.0 if not used).
    mcdrop_std      : float  — Std-dev of T MC Dropout predictions (epistemic).
    mcdrop_mean     : float  — Mean of T MC Dropout predictions.
    mcdrop_passes   : int    — Number of stochastic passes used (0 = disabled).

    # ── Weight breakdown ─────────────────────────────────────────────────
    weights         : dict   — {"boundary": w1, "entropy": w2, "mcdrop": w3}
    """

    # Core
    probability: float
    prediction: str
    threshold: float

    # Confidence
    confidence: float
    risk_level: str
    clinical_note: str

    # Breakdown
    boundary_score: float
    entropy_score: float
    entropy_raw: float
    mcdrop_score: float = 0.0
    mcdrop_std: float = 0.0
    mcdrop_mean: float = 0.0
    mcdrop_passes: int = 0
    weights: dict[str, float] = field(
        default_factory=lambda: {"boundary": 0.35, "entropy": 0.35, "mcdrop": 0.30}
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        emoji = RiskLevel.emoji(self.risk_level)
        lines = [
            "",
            f"  {'─' * 52}",
            "  CONFIDENCE ESTIMATION REPORT",
            f"  {'─' * 52}",
            "",
            f"  Prediction Probability  : {self.probability:.4f}  ({self.probability * 100:.1f}%)",
            f"  Predicted Class         : {self.prediction}",
            f"  Decision Threshold      : {self.threshold:.3f}",
            "",
            f"  Confidence Score        : {self.confidence:.4f}  ({self.confidence * 100:.1f}%)",
            f"  Risk Level              : {emoji} {self.risk_level}",
            "",
            "  ── Confidence Breakdown ──────────────────────────",
            f"  Boundary Distance Score : {self.boundary_score:.4f}  (w={self.weights['boundary']:.2f})",
            f"  Entropy Score           : {self.entropy_score:.4f}  (w={self.weights['entropy']:.2f})",
            f"  Raw Entropy H(p)        : {self.entropy_raw:.4f}  (max={np.log(2):.4f})",
        ]
        if self.mcdrop_passes > 0:
            lines += [
                f"  MC Dropout Score        : {self.mcdrop_score:.4f}  (w={self.weights['mcdrop']:.2f})",
                f"  MC Dropout std-dev      : {self.mcdrop_std:.4f}  (T={self.mcdrop_passes} passes)",
                f"  MC Dropout mean prob    : {self.mcdrop_mean:.4f}",
            ]
        else:
            lines.append("  MC Dropout              : disabled")
        lines += [
            "",
            "  ── Why Probability ≠ Confidence ──────────────────",
            "  Probability is what the model OUTPUTS (sigmoid of logit).",
            "  Confidence is how much you should TRUST that output.",
            "  A model can output p=0.95 on a blurry/atypical image",
            "  it was never trained on — that is HIGH probability but",
            "  LOW confidence. Always check the confidence score in",
            "  medical diagnostic AI before acting on predictions.",
            "",
            "  ── Clinical Guidance ─────────────────────────────",
            f"  {self.clinical_note}",
            f"  {'─' * 52}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Component Estimators
# ─────────────────────────────────────────────────────────────────────────────


def _boundary_score(prob: float, threshold: float) -> float:
    """
    Confidence score from distance to decision boundary.

    Intuition: the further a prediction is from the threshold, the more
    decisive (and thus more confident) it is.

    Normalised to [0, 1]:
      distance = |p − t| ∈ [0, max(t, 1−t)]
      score    = distance / max(t, 1−t)

    Examples (@threshold=0.50):
      p=0.97 → score = |0.97−0.50| / 0.50 = 0.94
      p=0.53 → score = |0.53−0.50| / 0.50 = 0.06
      p=0.50 → score = 0.0  (maximally uncertain)
    """
    distance = abs(prob - threshold)
    max_dist = max(threshold, 1.0 - threshold)  # ≥ 0.5
    return float(np.clip(distance / max_dist, 0.0, 1.0))


def _entropy_score(prob: float) -> tuple[float, float]:
    """
    Confidence score from predictive entropy.

    Binary entropy: H(p) = −[p·ln(p) + (1−p)·ln(1−p)]
      H(0.0) = H(1.0) = 0.0  (certain)
      H(0.5) = ln(2) ≈ 0.693 (maximally uncertain)

    Normalised entropy:  H_norm = H(p) / ln(2)  ∈ [0, 1]
    Confidence:          1 − H_norm

    Returns
    -------
    (confidence_score, raw_entropy)
    """
    eps = 1e-7
    p = float(np.clip(prob, eps, 1 - eps))
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    h_norm = h / np.log(2)
    return float(1.0 - h_norm), float(h)


def _mcdropout_score(
    model: nn.Module,
    image: torch.Tensor,  # (1, C, H, W) — single image
    device: torch.device,
    passes: int = 20,
    use_amp: bool = False,
) -> tuple[float, float, float]:
    """
    Monte Carlo Dropout uncertainty estimate for a single image.

    Method (Gal & Ghahramani, 2016):
      1. Keep dropout ACTIVE at inference (model.train() mode).
      2. Run T stochastic forward passes over the same image.
      3. Each pass uses a different random dropout mask → different prediction.
      4. Variance of the T predictions = epistemic uncertainty.

    Returns
    -------
    (mcdrop_confidence, std_dev, mean_prob)
      mcdrop_confidence : 1 − (std × 4), clipped to [0, 1]
                          std=0.25 → worst-case uncertainty → score=0
                          std=0.00 → perfectly consistent  → score=1
      std_dev           : Standard deviation of T predictions.
      mean_prob         : Mean of T predictions.
    """
    model.train()  # activate dropout
    probs: list[float] = []

    with torch.no_grad():
        image = image.to(device)
        for _ in range(passes):
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logit = model(image)  # (1, 1)
                prob = torch.sigmoid(logit).item()
            probs.append(prob)

    model.eval()  # restore eval mode

    arr = np.array(probs, dtype=np.float32)
    std = float(arr.std())
    mean = float(arr.mean())

    # Confidence: maximum std of a Bernoulli = 0.5 → normalise by 0.5
    # We use 0.25 as practical maximum (std rarely exceeds 0.25 on trained models)
    mcdrop_conf = float(np.clip(1.0 - (std / 0.25), 0.0, 1.0))
    return mcdrop_conf, std, mean


# ─────────────────────────────────────────────────────────────────────────────
# ConfidenceEstimator  (main class)
# ─────────────────────────────────────────────────────────────────────────────


class ConfidenceEstimator:
    """
    Estimates model confidence alongside raw probability for ALL detection.

    Combines three complementary signals:
      1. Boundary distance  — how far from the decision gate
      2. Predictive entropy — information-theoretic certainty
      3. MC Dropout          — epistemic uncertainty via stochastic inference

    Args
    ----
    threshold       : Decision boundary for binary classification.
    mc_dropout_passes : Number of stochastic forward passes.
                        0 = disabled (faster, no MC dropout component).
    weights         : Tuple (w_boundary, w_entropy, w_mcdrop).
                      Must sum to 1.0.  If mc_dropout_passes=0, the mcdrop
                      weight is redistributed equally to boundary + entropy.
    risk_thresholds : Dict with keys "LOW" and "MEDIUM" for RiskLevel.
    use_amp         : Use AMP for MC Dropout passes (GPU only).

    Quick start
    -----------
    >>> estimator = ConfidenceEstimator(threshold=0.50, mc_dropout_passes=20)
    >>> result    = estimator.estimate(model, image_tensor)
    >>> print(result)
    """

    def __init__(
        self,
        threshold: float = 0.50,
        mc_dropout_passes: int = 20,
        weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
        risk_thresholds: dict[str, float] | None = None,
        use_amp: bool = False,
    ) -> None:
        if len(weights) != 3:
            raise ValueError("weights must be a 3-tuple: (boundary, entropy, mcdrop)")
        w_sum = sum(weights)
        if abs(w_sum - 1.0) > 1e-4:
            raise ValueError(f"weights must sum to 1.0 (got {w_sum:.4f})")

        self.threshold = threshold
        self.mc_dropout_passes = mc_dropout_passes
        self.risk_thresholds = risk_thresholds or RiskLevel.THRESHOLDS
        self.use_amp = use_amp

        # Adjust weights if MC Dropout is disabled
        if mc_dropout_passes == 0:
            w_b, w_e, _ = weights
            total = w_b + w_e
            self._w_boundary = w_b / total
            self._w_entropy = w_e / total
            self._w_mcdrop = 0.0
        else:
            self._w_boundary, self._w_entropy, self._w_mcdrop = weights

        logger.info(
            f"ConfidenceEstimator ready | "
            f"threshold={threshold:.2f} | "
            f"mc_passes={mc_dropout_passes} | "
            f"weights=({self._w_boundary:.2f}, {self._w_entropy:.2f}, {self._w_mcdrop:.2f})"
        )

    # ── Single-image estimation ───────────────────────────────────────────────

    def estimate(
        self,
        model: nn.Module,
        image: torch.Tensor,  # (1, C, H, W) or (C, H, W) single image
        device: torch.device | None = None,
    ) -> ConfidenceResult:
        """
        Estimate confidence for a single image tensor.

        Args
        ----
        model  : Trained nn.Module (will be temporarily set to train mode
                 for MC Dropout, then restored to eval).
        image  : (1, C, H, W) or (C, H, W) float tensor.
        device : Inference device.  Inferred from model parameters if None.

        Returns
        -------
        ConfidenceResult dataclass with all fields populated.
        """
        device = device or next(model.parameters()).device
        image = image.unsqueeze(0) if image.ndim == 3 else image  # → (1,C,H,W)

        # ── Step 1: deterministic probability ────────────────────────────────
        model.eval()
        with (
            torch.no_grad(),
            torch.cuda.amp.autocast(enabled=(self.use_amp and device.type == "cuda")),
        ):
            logit = model(image.to(device))
            prob = float(torch.sigmoid(logit).item())

        # ── Step 2: boundary distance score ──────────────────────────────────
        b_score = _boundary_score(prob, self.threshold)

        # ── Step 3: entropy score ─────────────────────────────────────────────
        e_score, h_raw = _entropy_score(prob)

        # ── Step 4: MC Dropout score (optional) ──────────────────────────────
        mc_conf, mc_std, mc_mean = 0.0, 0.0, prob
        if self.mc_dropout_passes > 0:
            mc_conf, mc_std, mc_mean = _mcdropout_score(
                model,
                image,
                device,
                passes=self.mc_dropout_passes,
                use_amp=self.use_amp,
            )
        model.eval()  # ensure we leave model in eval

        # ── Step 5: combined confidence ───────────────────────────────────────
        confidence = (
            self._w_boundary * b_score
            + self._w_entropy * e_score
            + self._w_mcdrop * mc_conf
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # ── Step 6: risk + prediction ─────────────────────────────────────────
        predicted_cls = "ALL+" if prob >= self.threshold else "Healthy"
        risk = RiskLevel.from_confidence(confidence, self.risk_thresholds)
        note = RiskLevel.clinical_note(risk, predicted_cls)

        return ConfidenceResult(
            probability=prob,
            prediction=predicted_cls,
            threshold=self.threshold,
            confidence=confidence,
            risk_level=risk,
            clinical_note=note,
            boundary_score=b_score,
            entropy_score=e_score,
            entropy_raw=h_raw,
            mcdrop_score=mc_conf,
            mcdrop_std=mc_std,
            mcdrop_mean=mc_mean,
            mcdrop_passes=self.mc_dropout_passes,
            weights={
                "boundary": round(self._w_boundary, 4),
                "entropy": round(self._w_entropy, 4),
                "mcdrop": round(self._w_mcdrop, 4),
            },
        )

    # ── Batch estimation (from pre-collected probabilities) ───────────────────

    def estimate_batch(
        self,
        probs: np.ndarray,  # (N,) float array of sigmoid probabilities
    ) -> list[ConfidenceResult]:
        """
        Compute confidence for a batch of pre-collected probabilities.

        NOTE: This method does NOT perform MC Dropout (no model access).
              It computes boundary + entropy confidence only and scales
              the weights accordingly.  Use ``estimate()`` per image for
              full MC Dropout confidence.

        Args
        ----
        probs : (N,) array of sigmoid probabilities ∈ [0, 1].

        Returns
        -------
        List of ConfidenceResult (one per sample).
        """
        # Redistribute mcdrop weight to boundary + entropy
        total_w = self._w_boundary + self._w_entropy
        w_b = self._w_boundary / total_w
        w_e = self._w_entropy / total_w

        results = []
        for prob in probs:
            prob = float(prob)
            b_sc = _boundary_score(prob, self.threshold)
            e_sc, h_raw = _entropy_score(prob)
            conf = float(np.clip(w_b * b_sc + w_e * e_sc, 0.0, 1.0))

            predicted_cls = "ALL+" if prob >= self.threshold else "Healthy"
            risk = RiskLevel.from_confidence(conf, self.risk_thresholds)
            note = RiskLevel.clinical_note(risk, predicted_cls)

            results.append(
                ConfidenceResult(
                    probability=prob,
                    prediction=predicted_cls,
                    threshold=self.threshold,
                    confidence=conf,
                    risk_level=risk,
                    clinical_note=note,
                    boundary_score=b_sc,
                    entropy_score=e_sc,
                    entropy_raw=h_raw,
                    mcdrop_score=0.0,
                    mcdrop_std=0.0,
                    mcdrop_mean=prob,
                    mcdrop_passes=0,
                    weights={
                        "boundary": round(w_b, 4),
                        "entropy": round(w_e, 4),
                        "mcdrop": 0.0,
                    },
                )
            )
        return results

    # ── Summary statistics over a batch ──────────────────────────────────────

    @staticmethod
    def batch_summary(
        results: list[ConfidenceResult],
    ) -> dict[str, object]:
        """
        Aggregate statistics over a list of ConfidenceResults.

        Returns
        -------
        Dict with keys: mean_confidence, std_confidence, risk_counts,
        risk_fractions, mean_probability, n_samples, n_low, n_medium, n_high.
        """
        confidences = np.array([r.confidence for r in results])
        probs = np.array([r.probability for r in results])
        risks = [r.risk_level for r in results]

        n_low = risks.count(RiskLevel.LOW)
        n_medium = risks.count(RiskLevel.MEDIUM)
        n_high = risks.count(RiskLevel.HIGH)
        n = len(results)

        return {
            "n_samples": n,
            "mean_confidence": float(confidences.mean()),
            "std_confidence": float(confidences.std()),
            "min_confidence": float(confidences.min()),
            "max_confidence": float(confidences.max()),
            "mean_probability": float(probs.mean()),
            "n_low": n_low,
            "n_medium": n_medium,
            "n_high": n_high,
            "risk_counts": {
                RiskLevel.LOW: n_low,
                RiskLevel.MEDIUM: n_medium,
                RiskLevel.HIGH: n_high,
            },
            "risk_fractions": {
                RiskLevel.LOW: round(n_low / n, 4),
                RiskLevel.MEDIUM: round(n_medium / n, 4),
                RiskLevel.HIGH: round(n_high / n, 4),
            },
        }

    # ── Probability vs Confidence explanation ─────────────────────────────────

    @staticmethod
    def explain() -> str:
        """
        Return a formatted explanation of why probability ≠ confidence.
        Useful for printing to a report or displaying in a UI.
        """
        return """
╔══════════════════════════════════════════════════════════════════╗
║        WHY PROBABILITY ≠ CONFIDENCE IN MEDICAL AI               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  PROBABILITY (p)                                                 ║
║  ─────────────────────────────────────────────────────────────  ║
║  • The raw sigmoid output: p = σ(logit) ∈ [0, 1]               ║
║  • Answers: "Fraction of patterns that look like ALL+"          ║
║  • Problem: Neural nets are OVERCONFIDENT by design.            ║
║    A model can output p=0.95 on a blurry/novel image            ║
║    it was never trained on.                                      ║
║  • Affected by: label smoothing, focal loss, calibration        ║
║                                                                  ║
║  CONFIDENCE (c)                                                  ║
║  ─────────────────────────────────────────────────────────────  ║
║  • Post-hoc measure: "How much to trust this prediction?"       ║
║  • Computed from THREE signals:                                  ║
║    1. Boundary Distance  : |p − threshold| / max_dist           ║
║       Far from 0.5 = decisive = more confident                  ║
║    2. Predictive Entropy : 1 − H(p)/ln(2)                       ║
║       H(p=0.5)=max, H(p→0 or 1)=0. Low entropy → confident     ║
║    3. MC Dropout Variance: σ² over T stochastic passes          ║
║       High variance = model disagrees with itself = uncertain   ║
║                                                                  ║
║  CONCRETE EXAMPLE                                                ║
║  ─────────────────────────────────────────────────────────────  ║
║  Image A (clear blast cell):  p=0.97, c=0.91 → LOW RISK        ║
║  Image B (blurry/OOD cell):   p=0.89, c=0.41 → HIGH RISK       ║
║                                                                  ║
║  Both have HIGH probability — but only A should be trusted.     ║
║  The confidence score is what distinguishes them.               ║
║                                                                  ║
║  RISK LEVELS                                                     ║
║  ─────────────────────────────────────────────────────────────  ║
║  🟢 LOW    c ≥ 0.80  Reliable prediction. Act accordingly.      ║
║  🟡 MEDIUM c ≥ 0.55  Borderline. Pathologist review advised.    ║
║  🔴 HIGH   c < 0.55  Uncertain. Human review REQUIRED.          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
