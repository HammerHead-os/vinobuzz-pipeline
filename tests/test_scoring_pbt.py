"""Property-based tests for the Confidence Scorer.

Feature: wine-photo-pipeline
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypothesis import given, settings
from hypothesis import strategies as st

from wine_pipeline.scoring import compare_field


# ---------------------------------------------------------------------------
# Property 5: Field comparison scores are bounded and identity-correct
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------

@given(a=st.text(min_size=1, max_size=80), b=st.text(min_size=1, max_size=80))
@settings(max_examples=100)
def test_field_comparison_bounded(a: str, b: str):
    """Property 5a: compare_field always returns a score in [0.0, 1.0]."""
    score = compare_field(a, b)
    assert 0.0 <= score <= 1.0


@given(s=st.text(min_size=1, max_size=80))
@settings(max_examples=100)
def test_field_comparison_identity(s: str):
    """Property 5b: self-comparison always returns 1.0."""
    assert compare_field(s, s) == 1.0


from wine_pipeline.scoring import ConfidenceScorer, DEFAULT_WEIGHTS, _verdict_from_confidence, _clamp
from wine_pipeline.models import FieldScores, OCRResult, QualityResult, Verdict

# Reusable strategies
unit_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

field_scores_strategy = st.builds(
    FieldScores,
    producer_match=unit_float,
    appellation_match=unit_float,
    cru_match=unit_float,
    vintage_match=unit_float,
)

quality_strategy = st.builds(
    QualityResult,
    passed=st.booleans(),
    image_quality_score=unit_float,
    rejection_reasons=st.just([]),
)

# OCR where all fields confirmed, none disagreed
ocr_all_confirmed = st.builds(
    OCRResult,
    raw_text=st.just(""),
    producer_confirmed=st.just(True),
    appellation_confirmed=st.just(True),
    cru_confirmed=st.just(True),
    vintage_confirmed=st.just(True),
    fields_disagreed=st.just([]),
)

# OCR neutral (nothing confirmed, nothing disagreed)
ocr_neutral = st.builds(
    OCRResult,
    raw_text=st.just(""),
    producer_confirmed=st.just(False),
    appellation_confirmed=st.just(False),
    cru_confirmed=st.just(False),
    vintage_confirmed=st.just(False),
    fields_disagreed=st.just([]),
)


# ---------------------------------------------------------------------------
# Property 6: NV wine scoring redistributes vintage weight
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

@given(
    fs=field_scores_strategy,
    qr=quality_strategy,
    ocr=ocr_neutral,
)
@settings(max_examples=100)
def test_nv_weight_redistribution_sums_to_one(fs, qr, ocr):
    """Property 6: For NV wines, remaining weights (excl. vintage) sum to 1.0."""
    scorer = ConfidenceScorer()
    weights = scorer._resolve_weights(is_nv=True)
    assert weights["vintage_match"] == 0.0
    remaining = sum(v for k, v in weights.items() if k != "vintage_match")
    assert abs(remaining - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Property 7: OCR agreement boosts and disagreement reduces field scores
# Validates: Requirements 3.4, 4.3, 4.4
# ---------------------------------------------------------------------------

@given(base=st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
@settings(max_examples=100)
def test_ocr_agreement_boosts_score(base):
    """Property 7a: OCR confirmation boosts the field score."""
    ocr = OCRResult(
        raw_text="", producer_confirmed=True,
        appellation_confirmed=False, cru_confirmed=False,
        vintage_confirmed=False, fields_disagreed=[],
    )
    adjusted = ConfidenceScorer._adjust_ocr(base, "producer", ocr)
    assert adjusted >= base


@given(base=st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
@settings(max_examples=100)
def test_ocr_disagreement_reduces_score(base):
    """Property 7b: OCR disagreement reduces the field score."""
    ocr = OCRResult(
        raw_text="", producer_confirmed=False,
        appellation_confirmed=False, cru_confirmed=False,
        vintage_confirmed=False, fields_disagreed=["producer"],
    )
    adjusted = ConfidenceScorer._adjust_ocr(base, "producer", ocr)
    assert adjusted < base


# ---------------------------------------------------------------------------
# Property 10: All dimension scores are bounded in [0.0, 1.0]
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------

@given(
    fs=field_scores_strategy,
    qr=quality_strategy,
    ocr=ocr_neutral,
    is_nv=st.booleans(),
)
@settings(max_examples=100)
def test_all_dimension_scores_bounded(fs, qr, ocr, is_nv):
    """Property 10: Every dimension score in the result is in [0.0, 1.0]."""
    scorer = ConfidenceScorer()
    result = scorer.score(fs, ocr, qr, label_detected=True, is_nv=is_nv)
    for val in [result.producer_match, result.appellation_match,
                result.cru_match, result.vintage_match, result.image_quality]:
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Property 11: Overall confidence equals weighted sum
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------

@given(
    fs=field_scores_strategy,
    qr=quality_strategy,
    ocr=ocr_neutral,
    is_nv=st.booleans(),
)
@settings(max_examples=100)
def test_overall_confidence_equals_weighted_sum(fs, qr, ocr, is_nv):
    """Property 11: overall_confidence == weighted sum of dimension scores."""
    scorer = ConfidenceScorer()
    result = scorer.score(fs, ocr, qr, label_detected=True, is_nv=is_nv)
    weights = scorer._resolve_weights(is_nv)
    expected = (
        weights["producer_match"] * result.producer_match
        + weights["appellation_match"] * result.appellation_match
        + weights["cru_match"] * result.cru_match
        + weights["vintage_match"] * result.vintage_match
        + weights["image_quality"] * result.image_quality
    )
    assert abs(result.overall_confidence - expected) < 1e-9


# ---------------------------------------------------------------------------
# Property 12: Verdict thresholds correctly applied
# Validates: Requirements 6.3, 6.4, 6.5
# ---------------------------------------------------------------------------

@given(conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=100)
def test_verdict_thresholds(conf):
    """Property 12: Verdict matches threshold rules."""
    verdict = _verdict_from_confidence(conf)
    if conf >= 0.70:
        assert verdict == Verdict.PASS
    elif conf >= 0.50:
        assert verdict == Verdict.QUARANTINE
    else:
        assert verdict == Verdict.REJECT
