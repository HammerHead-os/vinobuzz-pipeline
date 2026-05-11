"""Property-based tests for the OCR Cross-Checker.

Feature: wine-photo-pipeline, Property 4: OCR extraction returns structured fields with cross-check validation
Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from wine_pipeline.models import Fingerprint
from wine_pipeline.ocr import OCRCrossChecker, _token_found


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Wine-like tokens that are realistic enough for fuzzy matching
wine_token = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs"), min_codepoint=32, max_codepoint=122),
    min_size=4,
    max_size=40,
)

# Random padding text to surround embedded tokens
padding = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P"), min_codepoint=32, max_codepoint=122),
    min_size=0,
    max_size=60,
)


# ---------------------------------------------------------------------------
# Property 4: OCR extraction returns structured fields with cross-check validation
# Validates: Requirements 4.1, 4.2, 4.3, 4.4
# ---------------------------------------------------------------------------

@given(token=wine_token, before=padding, after=padding)
@settings(max_examples=100)
def test_embedded_token_is_found(token: str, before: str, after: str):
    """Property 4.1: When a known token is embedded in raw OCR text,
    _token_found identifies it using fuzzy matching.

    **Validates: Requirements 4.2**
    """
    assume(len(token.strip()) >= 4)
    raw_text = before + " " + token + " " + after
    assert _token_found(raw_text, token) is True


@given(
    producer=wine_token,
    appellation=wine_token,
    before=padding,
    middle=padding,
    after=padding,
)
@settings(max_examples=100)
def test_cross_reference_confirms_embedded_fields(
    producer: str, appellation: str, before: str, middle: str, after: str
):
    """Property 4.2: _cross_reference confirms fields whose values
    appear in the raw OCR text and correctly identifies disagreements.

    **Validates: Requirements 4.2, 4.4**
    """
    assume(len(producer.strip()) >= 4)
    assume(len(appellation.strip()) >= 4)

    raw_text = before + " " + producer + " " + middle + " " + appellation + " " + after

    fingerprint = Fingerprint(
        producer=producer,
        appellation=appellation,
        cru_vineyard=None,
        vintage=None,
    )

    result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

    assert result.producer_confirmed is True
    assert result.appellation_confirmed is True
    # cru and vintage are None so should not be confirmed
    assert result.cru_confirmed is False
    assert result.vintage_confirmed is False
    # No disagreements since non-provided fields are not flagged
    assert result.fields_disagreed == []


@given(
    raw_text=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P"), min_codepoint=32, max_codepoint=122),
        min_size=0,
        max_size=200,
    ),
    producer=wine_token,
    appellation=wine_token,
    cru=wine_token,
    vintage=st.text(min_size=4, max_size=4, alphabet=st.characters(whitelist_categories=("N"))),
)
@settings(max_examples=100)
def test_cross_reference_identifies_disagreements(
    raw_text: str, producer: str, appellation: str, cru: str, vintage: str
):
    """Property 4.3: _cross_reference correctly identifies disagreed fields
    when OCR text does not contain fingerprint values.

    **Validates: Requirements 4.4**
    """
    assume(len(producer.strip()) >= 4)
    assume(len(appellation.strip()) >= 4)
    assume(len(cru.strip()) >= 4)
    assume(len(vintage.strip()) == 4)
    
    # Assume raw_text doesn't contain the fingerprint values
    assume(not _token_found(raw_text, producer))
    assume(not _token_found(raw_text, appellation))
    assume(not _token_found(raw_text, cru))
    assume(not _token_found(raw_text, vintage))

    fingerprint = Fingerprint(
        producer=producer,
        appellation=appellation,
        cru_vineyard=cru,
        vintage=vintage,
    )

    result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

    # All fields should be marked as not confirmed
    assert result.producer_confirmed is False
    assert result.appellation_confirmed is False
    assert result.cru_confirmed is False
    assert result.vintage_confirmed is False
    
    # All fields should be in disagreements
    assert "producer" in result.fields_disagreed
    assert "appellation" in result.fields_disagreed
    assert "cru" in result.fields_disagreed
    assert "vintage" in result.fields_disagreed


@given(
    text_with_token=st.tuples(padding, wine_token, padding),
    other_token=wine_token,
)
@settings(max_examples=100)
def test_cross_reference_partial_disagreement(text_with_token, other_token):
    """Property 4.4: _cross_reference correctly handles partial matches
    where some fields are confirmed and some are disagreed.

    **Validates: Requirements 4.4**
    """
    before, token, after = text_with_token
    assume(len(token.strip()) >= 4)
    assume(len(other_token.strip()) >= 4)
    # Assume other_token is not in the text
    assume(not _token_found(before + " " + token + " " + after, other_token))

    raw_text = before + " " + token + " " + after

    fingerprint = Fingerprint(
        producer=token,  # Should be found
        appellation=other_token,  # Should not be found
        cru_vineyard=None,
        vintage=None,
    )

    result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

    assert result.producer_confirmed is True
    assert result.appellation_confirmed is False
    assert "producer" not in result.fields_disagreed
    assert "appellation" in result.fields_disagreed
    # cru_vineyard is None, should not be in disagreements
    assert "cru" not in result.fields_disagreed
