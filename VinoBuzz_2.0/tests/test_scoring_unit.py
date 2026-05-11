"""Unit tests for Confidence Scorer edge cases.

Validates: Requirements 6.3, 6.4, 6.5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from wine_pipeline.models import FieldScores, OCRResult, QualityResult, Verdict
from wine_pipeline.scoring import ConfidenceScorer, _verdict_from_confidence


# ---------------------------------------------------------------------------
# Verdict boundary values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("conf, expected", [
    (0.0, Verdict.REJECT),
    (0.49, Verdict.REJECT),
    (0.50, Verdict.QUARANTINE),
    (0.699, Verdict.QUARANTINE),
    (0.70, Verdict.PASS),
    (1.0, Verdict.PASS),
])
def test_verdict_boundaries(conf, expected):
    assert _verdict_from_confidence(conf) == expected


# ---------------------------------------------------------------------------
# NV wine with all fields matching perfectly
# ---------------------------------------------------------------------------

def _neutral_ocr():
    return OCRResult(
        raw_text="", producer_confirmed=False, appellation_confirmed=False,
        cru_confirmed=False, vintage_confirmed=False, fields_disagreed=[],
    )


def test_nv_perfect_match():
    scorer = ConfidenceScorer()
    fs = FieldScores(producer_match=1.0, appellation_match=1.0, cru_match=1.0, vintage_match=0.0)
    qr = QualityResult(passed=True, image_quality_score=1.0, rejection_reasons=[])
    result = scorer.score(fs, _neutral_ocr(), qr, label_detected=True, is_nv=True)
    # All non-vintage fields are 1.0, quality is 1.0 → overall should be 1.0
    assert abs(result.overall_confidence - 1.0) < 1e-9
    assert result.verdict == Verdict.PASS


# ---------------------------------------------------------------------------
# All fields disagreed by OCR
# ---------------------------------------------------------------------------

def test_all_fields_disagreed():
    scorer = ConfidenceScorer()
    fs = FieldScores(producer_match=0.8, appellation_match=0.8, cru_match=0.8, vintage_match=0.8)
    ocr = OCRResult(
        raw_text="", producer_confirmed=False, appellation_confirmed=False,
        cru_confirmed=False, vintage_confirmed=False,
        fields_disagreed=["producer", "appellation", "cru", "vintage"],
    )
    qr = QualityResult(passed=True, image_quality_score=0.9, rejection_reasons=[])
    result = scorer.score(fs, ocr, qr, label_detected=True, is_nv=False)
    # Each field should be reduced from 0.8 by the penalty
    assert result.producer_match < 0.8
    assert result.appellation_match < 0.8
    assert result.cru_match < 0.8
    assert result.vintage_match < 0.8
