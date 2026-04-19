"""Property-based tests for the Streamlit Demo UI summary metrics.

Feature: wine-photo-pipeline, Property 16: Summary metrics correctly count verdicts
Validates: Requirements 8.4
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypothesis import given, settings
from hypothesis import strategies as st

from wine_pipeline.models import Fingerprint, ScoredResult, Verdict
from wine_pipeline.app import compute_summary_metrics


scored_result_strategy = st.builds(
    ScoredResult,
    sku_id=st.text(min_size=1, max_size=30),
    image_url=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    producer_match=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    appellation_match=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    cru_match=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    vintage_match=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    image_quality=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    overall_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    verdict=st.sampled_from(Verdict),
    fingerprint=st.none(),
    rejection_reasons=st.just([]),
)


# ---------------------------------------------------------------------------
# Property 16: Summary metrics correctly count verdicts
# Validates: Requirements 8.4
# ---------------------------------------------------------------------------


@given(results=st.lists(scored_result_strategy, min_size=0, max_size=50))
@settings(max_examples=100)
def test_summary_metrics_counts_sum_to_total(results: list[ScoredResult]):
    """Property 16a: PASS + QUARANTINE + REJECT counts == total."""
    metrics = compute_summary_metrics(results)
    assert metrics["pass_count"] + metrics["quarantine_count"] + metrics["reject_count"] == metrics["total"]
    assert metrics["total"] == len(results)


@given(results=st.lists(scored_result_strategy, min_size=1, max_size=50))
@settings(max_examples=100)
def test_summary_metrics_percentages_equal_count_over_total(results: list[ScoredResult]):
    """Property 16b: Each percentage == count / total."""
    metrics = compute_summary_metrics(results)
    total = metrics["total"]
    assert abs(metrics["pass_pct"] - metrics["pass_count"] / total) < 1e-9
    assert abs(metrics["quarantine_pct"] - metrics["quarantine_count"] / total) < 1e-9
    assert abs(metrics["reject_pct"] - metrics["reject_count"] / total) < 1e-9
