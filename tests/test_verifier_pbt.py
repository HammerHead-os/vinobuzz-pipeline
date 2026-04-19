"""Property-based tests for the Fingerprint Verifier.

Feature: wine-photo-pipeline, Property 4: Fingerprint extraction always produces all four fields
Validates: Requirements 3.1
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from wine_pipeline.models import Fingerprint
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
