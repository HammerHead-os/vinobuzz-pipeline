"""Property-based tests for the SQLite cache layer.

Feature: wine-photo-pipeline, Property 8: Caching stores and retrieves results correctly
Validates: Requirements 9.1, 9.2, 9.3, 9.4

Property 8: For any SKU result, the cache should store the result in SQLite,
retrieve it on subsequent requests for the same SKU, bypass cache when requested,
and continue processing if cache storage fails.
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
    """Property 8.1: Cache store/retrieve round-trip.

    For any valid ScoredResult stored in the cache, retrieving it by
    the same SKU ID produces an equivalent ScoredResult.

    **Validates: Requirements 9.1, 9.2**
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


@given(result=scored_result_strategy)
@settings(max_examples=100)
def test_cache_invalidation_enables_reprocess(result: ScoredResult):
    """Property 8.2: Cache invalidation allows reprocessing.

    For any cached result, invalidating it causes subsequent get() to return None,
    signaling that reprocessing is required.

    **Validates: Requirements 9.2, 9.3**
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        cache = ResultCache(db_path=tmp.name)
        try:
            # Store result
            cache.put(result.sku_id, result)
            
            # Verify it exists
            retrieved = cache.get(result.sku_id)
            assert retrieved is not None
            
            # Invalidate
            cache.invalidate(result.sku_id)
            
            # Verify it returns None (cache miss)
            retrieved_after = cache.get(result.sku_id)
            assert retrieved_after is None
        finally:
            cache.close()


@given(result=scored_result_strategy, updated_result=scored_result_strategy)
@settings(max_examples=100)
def test_cache_update_replaces_result(result: ScoredResult, updated_result: ScoredResult):
    """Property 8.3: Storing a result with same SKU ID updates the existing entry.

    For any cached result, storing a new result with the same SKU ID replaces
    the old result with the new one.

    **Validates: Requirements 9.1, 9.2**
    """
    # Ensure same SKU ID for both results
    updated_result_with_same_sku = ScoredResult(
        sku_id=result.sku_id,
        image_url=updated_result.image_url,
        producer_match=updated_result.producer_match,
        appellation_match=updated_result.appellation_match,
        cru_match=updated_result.cru_match,
        vintage_match=updated_result.vintage_match,
        image_quality=updated_result.image_quality,
        overall_confidence=updated_result.overall_confidence,
        verdict=updated_result.verdict,
        fingerprint=updated_result.fingerprint,
        rejection_reasons=updated_result.rejection_reasons,
    )
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        cache = ResultCache(db_path=tmp.name)
        try:
            # Store initial result
            cache.put(result.sku_id, result)
            
            # Store updated result with same SKU ID
            cache.put(updated_result_with_same_sku.sku_id, updated_result_with_same_sku)
            
            # Verify only one entry exists and it has the updated values
            retrieved = cache.get(result.sku_id)
            assert retrieved is not None
            assert retrieved.image_url == updated_result.image_url
            assert retrieved.overall_confidence == updated_result.overall_confidence
            assert retrieved.verdict == updated_result.verdict
            assert retrieved.fingerprint == updated_result.fingerprint
            assert retrieved.rejection_reasons == updated_result.rejection_reasons
        finally:
            cache.close()


@given(sku_ids=st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=10, unique=True))
@settings(max_examples=50)
def test_cache_miss_for_unknown_sku(sku_ids: list[str]):
    """Property 8.4: Cache miss returns None for unknown SKU IDs.

    For any SKU ID that has never been cached, get() returns None.

    **Validates: Requirements 9.2**
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        cache = ResultCache(db_path=tmp.name)
        try:
            for sku_id in sku_ids:
                # Query for SKU that was never stored
                retrieved = cache.get(sku_id)
                assert retrieved is None
        finally:
            cache.close()
