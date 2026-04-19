"""Confidence scoring for the Wine Photo Pipeline.

Implements field comparison, weighted scoring, OCR adjustment, and verdict assignment.
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from wine_pipeline.models import (
    FieldScores,
    OCRResult,
    QualityResult,
    ScoredResult,
    Verdict,
)


def compare_field(extracted: str | None, expected: str | None) -> float:
    """Compare two field strings using normalised Levenshtein similarity,
    with a token-overlap boost for partial matches common in wine appellations.

    Returns:
        0.0  – if either input is None or empty
        1.0  – if the strings are identical (case-insensitive)
        (0, 1) – fuzzy similarity otherwise
    """
    if not extracted or not expected:
        return 0.0
    ext_lower = extracted.lower()
    exp_lower = expected.lower()

    # Base fuzzy similarity
    base = Levenshtein.normalized_similarity(ext_lower, exp_lower)

    # Token overlap boost: what fraction of expected tokens appear in extracted?
    exp_tokens = set(exp_lower.split())
    ext_tokens = set(ext_lower.split())
    if exp_tokens:
        overlap = len(exp_tokens & ext_tokens) / len(exp_tokens)
    else:
        overlap = 0.0

    # Return the better of the two signals
    return max(base, overlap)


# Default scoring weights
DEFAULT_WEIGHTS = {
    "producer_match": 0.35,
    "appellation_match": 0.25,
    "cru_match": 0.15,
    "vintage_match": 0.15,
    "image_quality": 0.10,
}

# OCR adjustment factors
OCR_BOOST = 0.05
OCR_PENALTY = 0.05


class ConfidenceScorer:
    """Computes per-dimension and overall confidence scores, assigns verdict."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        field_scores: FieldScores,
        ocr_result: OCRResult,
        quality_result: QualityResult,
        label_detected: bool,
        is_nv: bool = False,
    ) -> ScoredResult:
        """Compute per-dimension scores, weighted overall confidence, and verdict."""

        # Start from base field scores
        producer = field_scores.producer_match
        appellation = field_scores.appellation_match
        cru = field_scores.cru_match
        vintage = field_scores.vintage_match

        # --- OCR adjustment ---
        producer = self._adjust_ocr(producer, "producer", ocr_result)
        appellation = self._adjust_ocr(appellation, "appellation", ocr_result)
        cru = self._adjust_ocr(cru, "cru", ocr_result)
        vintage = self._adjust_ocr(vintage, "vintage", ocr_result)

        # Clamp to [0, 1]
        producer = _clamp(producer)
        appellation = _clamp(appellation)
        cru = _clamp(cru)
        vintage = _clamp(vintage)
        quality = _clamp(quality_result.image_quality_score)

        # --- Resolve weights (handle NV redistribution) ---
        weights = self._resolve_weights(is_nv)

        # --- Weighted overall confidence ---
        overall = (
            weights["producer_match"] * producer
            + weights["appellation_match"] * appellation
            + weights["cru_match"] * cru
            + weights["vintage_match"] * vintage
            + weights["image_quality"] * quality
        )

        verdict = _verdict_from_confidence(overall)

        return ScoredResult(
            sku_id="",  # caller fills in
            image_url=None,
            producer_match=producer,
            appellation_match=appellation,
            cru_match=cru,
            vintage_match=vintage,
            image_quality=quality,
            overall_confidence=overall,
            verdict=verdict,
            fingerprint=None,
            rejection_reasons=list(quality_result.rejection_reasons),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_weights(self, is_nv: bool) -> dict[str, float]:
        """Return weights dict; for NV wines redistribute vintage weight."""
        w = dict(self.weights)
        if is_nv:
            vintage_w = w.pop("vintage_match")
            remaining_total = sum(w.values())
            if remaining_total > 0:
                for k in w:
                    w[k] += vintage_w * (w[k] / remaining_total)
            w["vintage_match"] = 0.0
        return w

    @staticmethod
    def _adjust_ocr(base_score: float, field_name: str, ocr: OCRResult) -> float:
        confirmed = {
            "producer": ocr.producer_confirmed,
            "appellation": ocr.appellation_confirmed,
            "cru": ocr.cru_confirmed,
            "vintage": ocr.vintage_confirmed,
        }
        disagreed = field_name in ocr.fields_disagreed

        if disagreed:
            return base_score - OCR_PENALTY
        if confirmed.get(field_name, False):
            return base_score + OCR_BOOST
        return base_score


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _verdict_from_confidence(overall: float) -> Verdict:
    if overall >= 0.70:
        return Verdict.PASS
    if overall >= 0.50:
        return Verdict.QUARANTINE
    return Verdict.REJECT
