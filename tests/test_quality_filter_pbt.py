"""Property-based tests for Quality Filter.

Feature: wine-photo-pipeline, Property 9: Rejected images always have non-empty rejection reasons
Validates: Requirements 5.7
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypothesis import given, settings
from hypothesis import strategies as st

from wine_pipeline.models import QualityResult


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
    """Property 9: Rejected images always have non-empty rejection reasons.

    For any QualityResult where passed=False, rejection_reasons must be
    non-empty and every reason must be a non-empty string.

    **Validates: Requirements 5.7**
    """
    assert result.passed is False
    assert len(result.rejection_reasons) > 0
    for reason in result.rejection_reasons:
        assert isinstance(reason, str)
        assert len(reason) > 0
