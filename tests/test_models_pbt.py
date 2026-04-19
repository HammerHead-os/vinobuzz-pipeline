"""Property-based tests for core data models.

Feature: wine-photo-pipeline, Property 13: ScoredResult JSON round-trip
Validates: Requirements 6.6
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypothesis import given, settings
from hypothesis import strategies as st

from wine_pipeline.models import Fingerprint, ScoredResult, Verdict


fingerprint_strategy = st.one_of(
    st.none(),
    st.builds(
        Fingerprint,
        producer=st.one_of(st.none(), st.text(min_size=0, max_size=50)),
        appellation=st.one_of(st.none(), st.text(min_size=0, max_size=50)),
        cru_vineyard=st.one_of(st.none(), st.text(min_size=0, max_size=50)),
        vintage=st.one_of(st.none(), st.text(min_size=0, max_size=10)),
    ),
)

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
    fingerprint=fingerprint_strategy,
    rejection_reasons=st.lists(st.text(min_size=0, max_size=50), max_size=5),
)


@given(result=scored_result_strategy)
@settings(max_examples=100)
def test_scored_result_json_round_trip(result: ScoredResult):
    """Property 13: ScoredResult JSON round-trip.

    For any valid ScoredResult, from_json(to_json(x)) == x.
    """
    restored = ScoredResult.from_json(result.to_json())
    assert restored.sku_id == result.sku_id
    assert restored.image_url == result.image_url
    assert restored.producer_match == result.producer_match
    assert restored.appellation_match == result.appellation_match
    assert restored.cru_match == result.cru_match
    assert restored.vintage_match == result.vintage_match
    assert restored.image_quality == result.image_quality
    assert restored.overall_confidence == result.overall_confidence
    assert restored.verdict == result.verdict
    assert restored.fingerprint == result.fingerprint
    assert restored.rejection_reasons == result.rejection_reasons
