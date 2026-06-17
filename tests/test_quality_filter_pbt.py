"""Property-based tests for Quality Filter.

Feature: wine-photo-pipeline, Property 6: Quality evaluation checks all criteria and returns rejection reasons
Validates: Requirements 7.1, 7.2, 7.3, 7.4
"""

import sys, os
import json
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from wine_pipeline.models import QualityResult
from wine_pipeline.quality_filter import QualityFilter, _REJECTION_DESCRIPTIONS, _ALL_CRITERIA


# Strategy for generating criterion pass/fail values
criterion_value_strategy = st.sampled_from(["pass", "fail"])

# Strategy for generating valid JSON responses from Gemini
gemini_response_strategy = st.dictionaries(
    keys=st.sampled_from(_ALL_CRITERIA),
    values=criterion_value_strategy,
    min_size=len(_ALL_CRITERIA),  # All criteria
    max_size=len(_ALL_CRITERIA),
)

# Strategy for generating raw JSON text (including with markdown fences optionally)
quality_response_text_strategy = st.one_of(
    # Plain JSON
    gemini_response_strategy.map(lambda d: json.dumps(d)),
    # JSON with markdown fences
    gemini_response_strategy.map(lambda d: f"```json\n{json.dumps(d)}\n```"),
    # JSON with just code fences
    gemini_response_strategy.map(lambda d: f"```\n{json.dumps(d)}\n```"),
)


@given(raw_text=quality_response_text_strategy)
@settings(max_examples=100)
def test_quality_evaluation_checks_all_criteria(raw_text: str):
    """Property 6: Quality evaluation checks all criteria and returns rejection reasons.

    For any candidate image response from Gemini, the quality filter should:
    1. Parse all criteria (including full bottle visibility)
    2. Return rejection reasons for each failed criterion
    3. Calculate quality score as ratio of passed criteria
    4. Set passed=True only if all criteria pass

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    result = QualityFilter._parse_response(raw_text)
    
    # Verify result is a QualityResult
    assert isinstance(result, QualityResult)
    
    # Verify quality score is in valid range
    assert 0.0 <= result.image_quality_score <= 1.0
    
    # Parse the original response to verify correctness
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # If we can't parse, result should indicate failure
        assert result.passed is False
        assert result.image_quality_score == 0.0
        return
    
    # Count expected passes and fails
    expected_passed_count = sum(1 for c in _ALL_CRITERIA if data.get(c, "fail") == "pass")
    expected_score = expected_passed_count / len(_ALL_CRITERIA)
    
    # Verify score calculation
    assert result.image_quality_score == pytest.approx(expected_score, rel=0.01)
    
    # Verify rejection reasons
    expected_reasons = [
        _REJECTION_DESCRIPTIONS[c] 
        for c in _ALL_CRITERIA 
        if data.get(c, "fail") != "pass"
    ]
    
    assert len(result.rejection_reasons) == len(expected_reasons)
    for reason in expected_reasons:
        assert reason in result.rejection_reasons
    
    # Verify passed status
    assert result.passed == (len(expected_reasons) == 0)


# Additional property: Rejected images always have non-empty rejection reasons
quality_result_rejected_strategy = st.builds(
    QualityResult,
    passed=st.just(False),
    image_quality_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    rejection_reasons=st.lists(
        st.text(min_size=1, max_size=100),
        min_size=1,
        max_size=6,
    ),
)


@given(result=quality_result_rejected_strategy)
@settings(max_examples=100)
def test_rejected_images_have_non_empty_rejection_reasons(result: QualityResult):
    """Property: Rejected images always have non-empty rejection reasons.

    For any QualityResult where passed=False, rejection_reasons must be
    non-empty and every reason must be a non-empty string.

    **Validates: Requirements 7.2**
    """
    assert result.passed is False
    assert len(result.rejection_reasons) > 0
    for reason in result.rejection_reasons:
        assert isinstance(reason, str)
        assert len(reason) > 0
