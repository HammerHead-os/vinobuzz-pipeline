"""Property-based tests for output generation.

Feature: wine-photo-pipeline, Property 9: Output generation includes all required fields
Validates: Requirements 10.1, 10.2, 10.3, 10.4
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypothesis import given, settings
from hypothesis import strategies as st

from wine_pipeline.models import Fingerprint, ScoredResult, Verdict


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

fingerprints = st.one_of(
    st.none(),
    st.builds(
        Fingerprint,
        producer=st.one_of(st.none(), st.text(min_size=0, max_size=50)),
        appellation=st.one_of(st.none(), st.text(min_size=0, max_size=50)),
        cru_vineyard=st.one_of(st.none(), st.text(min_size=0, max_size=50)),
        vintage=st.one_of(st.none(), st.text(min_size=0, max_size=10)),
    ),
)

scored_results = st.builds(
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
    fingerprint=fingerprints,
    rejection_reasons=st.lists(st.text(min_size=0, max_size=50), max_size=5),
)


# ---------------------------------------------------------------------------
# Property 9: Output generation includes all required fields
# ---------------------------------------------------------------------------


@given(result=scored_results)
@settings(max_examples=100)
def test_output_includes_sku_id(result: ScoredResult):
    """Property 9.1: Output always includes SKU ID.
    
    **Validates: Requirements 10.1, 10.4**
    """
    json_data = result.to_json()
    assert "sku_id" in json_data
    assert json_data["sku_id"] == result.sku_id
    assert json_data["sku_id"] is not None


@given(result=scored_results)
@settings(max_examples=100)
def test_output_includes_image_url(result: ScoredResult):
    """Property 9.2: Output includes image URL (or None for no match).
    
    **Validates: Requirements 10.1, 10.2**
    """
    json_data = result.to_json()
    assert "image_url" in json_data
    # image_url can be None (for "No Image" case)
    if result.image_url is None:
        assert json_data["image_url"] is None


@given(result=scored_results)
@settings(max_examples=100)
def test_output_includes_field_scores(result: ScoredResult):
    """Property 9.3: Output includes all field match scores.
    
    **Validates: Requirements 10.4**
    """
    json_data = result.to_json()
    assert "producer_match" in json_data
    assert "appellation_match" in json_data
    assert "cru_match" in json_data
    assert "vintage_match" in json_data
    
    # All scores should be between 0.0 and 1.0
    for field in ["producer_match", "appellation_match", "cru_match", "vintage_match"]:
        assert 0.0 <= json_data[field] <= 1.0


@given(result=scored_results)
@settings(max_examples=100)
def test_output_includes_quality_and_confidence(result: ScoredResult):
    """Property 9.4: Output includes image quality and overall confidence.
    
    **Validates: Requirements 10.4**
    """
    json_data = result.to_json()
    assert "image_quality" in json_data
    assert "overall_confidence" in json_data
    
    # Both should be between 0.0 and 1.0
    assert 0.0 <= json_data["image_quality"] <= 1.0
    assert 0.0 <= json_data["overall_confidence"] <= 1.0


@given(result=scored_results)
@settings(max_examples=100)
def test_output_includes_verdict(result: ScoredResult):
    """Property 9.5: Output includes verdict (PASS, QUARANTINE, or REJECT).
    
    **Validates: Requirements 10.1, 10.4**
    """
    json_data = result.to_json()
    assert "verdict" in json_data
    assert json_data["verdict"] in ["PASS", "QUARANTINE", "REJECT"]
    assert json_data["verdict"] == result.verdict.value


@given(result=scored_results)
@settings(max_examples=100)
def test_output_includes_fingerprint(result: ScoredResult):
    """Property 9.6: Output includes fingerprint (or None).
    
    **Validates: Requirements 10.4**
    """
    json_data = result.to_json()
    assert "fingerprint" in json_data
    
    if result.fingerprint is None:
        assert json_data["fingerprint"] is None
    else:
        fp = json_data["fingerprint"]
        assert "producer" in fp
        assert "appellation" in fp
        assert "cru_vineyard" in fp
        assert "vintage" in fp


@given(result=scored_results)
@settings(max_examples=100)
def test_output_includes_rejection_reasons(result: ScoredResult):
    """Property 9.7: Output includes rejection reasons (empty if passed).
    
    **Validates: Requirements 10.3, 10.4**
    """
    json_data = result.to_json()
    assert "rejection_reasons" in json_data
    assert isinstance(json_data["rejection_reasons"], list)


@given(result=scored_results)
@settings(max_examples=100)
def test_output_json_is_serializable(result: ScoredResult):
    """Property 9.8: Output to_json() produces JSON-serializable data.
    
    **Validates: Requirements 10.4**
    """
    import json
    json_data = result.to_json()
    
    # Should not raise
    json_str = json.dumps(json_data)
    
    # Should be able to parse back
    parsed = json.loads(json_str)
    assert parsed["sku_id"] == result.sku_id


@given(result=scored_results)
@settings(max_examples=100)
def test_output_round_trip_preserves_all_fields(result: ScoredResult):
    """Property 9.9: Output round trip (to_json -> from_json) preserves all fields.
    
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4**
    """
    json_data = result.to_json()
    restored = ScoredResult.from_json(json_data)
    
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

