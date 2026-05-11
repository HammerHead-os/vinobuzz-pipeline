"""Property-based tests for the Fingerprint Verifier.

Feature: wine-photo-pipeline
Property 4: Fingerprint extraction always produces all four fields
Property 5: Field comparison uses fuzzy matching with cross-field fallback

Validates: Requirements 3.1, 4.1, 4.2, 6.1, 6.2, 6.3, 6.4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from wine_pipeline.models import Fingerprint, SKU, FieldScores
from wine_pipeline.verifier import FingerprintVerifier


# Strategy: random JSON-like dicts that may or may not contain the expected keys
_json_value = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), st.text(max_size=30)),
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(st.text(max_size=15), children, max_size=5),
    ),
    max_leaves=10,
)

random_json_strategy = st.dictionaries(st.text(max_size=20), _json_value, max_size=6)


# ---------------------------------------------------------------------------
# Property 4: Fingerprint extraction always produces all four fields
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------

@given(data=random_json_strategy)
@settings(max_examples=100)
def test_parse_fingerprint_always_has_four_fields(data: dict):
    """Property 4: Parsed Fingerprint always has all four fields (str or None).

    **Validates: Requirements 3.1**
    """
    raw_text = json.dumps(data)
    fp = FingerprintVerifier._parse_fingerprint(raw_text)

    assert isinstance(fp, Fingerprint)
    assert isinstance(fp.producer, (str, type(None)))
    assert isinstance(fp.appellation, (str, type(None)))
    assert isinstance(fp.cru_vineyard, (str, type(None)))
    assert isinstance(fp.vintage, (str, type(None)))


@given(raw=st.text(max_size=200))
@settings(max_examples=100)
def test_parse_fingerprint_handles_arbitrary_text(raw: str):
    """Property 4 (extended): Even completely invalid text produces a valid Fingerprint."""
    fp = FingerprintVerifier._parse_fingerprint(raw)

    assert isinstance(fp, Fingerprint)
    assert isinstance(fp.producer, (str, type(None)))
    assert isinstance(fp.appellation, (str, type(None)))
    assert isinstance(fp.cru_vineyard, (str, type(None)))
    assert isinstance(fp.vintage, (str, type(None)))


# ---------------------------------------------------------------------------
# Property 5: Field comparison uses fuzzy matching with cross-field fallback
# Validates: Requirements 6.1, 6.2, 6.3, 6.4
# ---------------------------------------------------------------------------

# Strategies for SKU and Fingerprint
optional_text = st.one_of(st.none(), st.text(min_size=1, max_size=50))
vintage_strategy = st.one_of(st.none(), st.integers(min_value=1900, max_value=2030))

fingerprint_strategy = st.builds(
    Fingerprint,
    producer=optional_text,
    appellation=optional_text,
    cru_vineyard=optional_text,
    vintage=st.one_of(st.none(), st.integers(min_value=1900, max_value=2030).map(str)),
)

sku_strategy = st.builds(
    SKU,
    id=st.text(min_size=1, max_size=20),
    producer=st.text(min_size=1, max_size=50),
    appellation=st.text(min_size=1, max_size=50),
    cru_vineyard=optional_text,
    vintage=vintage_strategy,
    format=st.just("750ml"),
    region=st.text(min_size=1, max_size=50),
)


@given(fingerprint=fingerprint_strategy, sku=sku_strategy)
@settings(max_examples=100)
def test_field_comparison_always_returns_valid_scores(fingerprint: Fingerprint, sku: SKU):
    """Property 5a: Field comparison always returns scores in [0.0, 1.0].

    **Validates: Requirements 6.1, 6.2**
    """
    scores = FingerprintVerifier._compare_fields(fingerprint, sku)

    assert isinstance(scores, FieldScores)
    assert 0.0 <= scores.producer_match <= 1.0
    assert 0.0 <= scores.appellation_match <= 1.0
    assert 0.0 <= scores.cru_match <= 1.0
    assert 0.0 <= scores.vintage_match <= 1.0


@given(fingerprint=fingerprint_strategy, sku=sku_strategy)
@settings(max_examples=100)
def test_nv_wine_vintage_score_is_zero(fingerprint: Fingerprint, sku: SKU):
    """Property 5b: NV wines (vintage=None) always get vintage score of 0.0.

    **Validates: Requirements 6.3**
    """
    assume(sku.vintage is None)  # Only test NV wines

    scores = FingerprintVerifier._compare_fields(fingerprint, sku)

    assert scores.vintage_match == 0.0


@given(
    producer=st.text(min_size=1, max_size=50),
    appellation=st.text(min_size=1, max_size=50),
    cru=st.text(min_size=1, max_size=50),
    vintage=st.integers(min_value=1900, max_value=2030),
)
@settings(max_examples=100)
def test_exact_match_produces_perfect_scores(producer: str, appellation: str, cru: str, vintage: int):
    """Property 5c: Exact matches produce perfect scores (1.0).

    **Validates: Requirements 6.1, 6.2, 6.4**
    """
    fingerprint = Fingerprint(
        producer=producer,
        appellation=appellation,
        cru_vineyard=cru,
        vintage=str(vintage),
    )
    sku = SKU(
        id="test",
        producer=producer,
        appellation=appellation,
        cru_vineyard=cru,
        vintage=vintage,
        format="750ml",
        region="Test Region",
    )

    scores = FingerprintVerifier._compare_fields(fingerprint, sku)

    assert scores.producer_match == 1.0
    assert scores.appellation_match == 1.0
    assert scores.cru_match == 1.0
    assert scores.vintage_match == 1.0


@given(
    producer=st.text(min_size=1, max_size=50),
    other_text=st.text(min_size=1, max_size=50),
)
@settings(max_examples=100)
def test_cross_field_matching_boosts_score(producer: str, other_text: str):
    """Property 5d: Cross-field matching can improve scores when direct match fails.

    Cross-field matching is applied to appellation and cru fields only.
    **Validates: Requirements 6.4**
    """
    assume(producer != other_text)

    # Direct match fails for appellation, but cross-field should find it in cru_vineyard
    fingerprint = Fingerprint(
        producer="Producer",  # Exact match
        appellation=other_text,  # Wrong appellation
        cru_vineyard=producer,  # Actual appellation is here
        vintage="2020",
    )
    sku = SKU(
        id="test",
        producer="Producer",
        appellation=producer,
        cru_vineyard=None,
        vintage=2020,
        format="750ml",
        region="Test Region",
    )

    scores = FingerprintVerifier._compare_fields(fingerprint, sku)

    # Cross-field match should find appellation in cru_vineyard field
    assert scores.appellation_match > 0.0


@given(fingerprint=fingerprint_strategy)
@settings(max_examples=100)
def test_missing_fingerprint_producer_scores_zero(fingerprint: Fingerprint):
    """Property 5e: Missing producer field in fingerprint scores 0.0 (no cross-field for producer).

    **Validates: Requirements 6.2**
    """
    # Use SKU with non-empty expected fields
    sku = SKU(
        id="test",
        producer="Some Producer",
        appellation="Some Appellation",
        cru_vineyard="Some Cru",
        vintage=2020,
        format="750ml",
        region="Test Region",
    )

    scores = FingerprintVerifier._compare_fields(fingerprint, sku)

    # Producer field does NOT use cross-field matching, so None should give 0.0
    if fingerprint.producer is None:
        assert scores.producer_match == 0.0
