"""Property-based tests for the SQLite cache layer.

Feature: wine-photo-pipeline, Property 14: Cache store/retrieve round-trip
Validates: Requirements 7.1
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypothesis import given, settings
from hypothesis import strategies as st

from wine_pipeline.cache import ResultCache
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
def test_cache_store_retrieve_round_trip(result: ScoredResult):
    """Property 14: Cache store/retrieve round-trip.

    For any valid ScoredResult stored in the cache, retrieving it by
    the same SKU ID produces an equivalent ScoredResult.

    **Validates: Requirements 7.1**
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        cache = ResultCache(db_path=tmp.name)
        try:
            cache.put(result.sku_id, result)
            retrieved = cache.get(result.sku_id)

            assert retrieved is not None
            assert retrieved.sku_id == result.sku_id
            assert retrieved.image_url == result.image_url
            assert retrieved.producer_match == result.producer_match
            assert retrieved.appellation_match == result.appellation_match
            assert retrieved.cru_match == result.cru_match
            assert retrieved.vintage_match == result.vintage_match
            assert retrieved.image_quality == result.image_quality
            assert retrieved.overall_confidence == result.overall_confidence
            assert retrieved.verdict == result.verdict
            assert retrieved.fingerprint == result.fingerprint
            assert retrieved.rejection_reasons == result.rejection_reasons
        finally:
            cache.close()
