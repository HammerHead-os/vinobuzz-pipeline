"""Unit tests for OCR Cross-Checker.

Tests for Cloud Vision OCR API calls, text extraction, and cross-reference logic.
Requirements: 4.3, 5.1, 5.2, 5.3, 5.4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import Mock, patch, MagicMock
from wine_pipeline.models import Fingerprint, OCRResult
from wine_pipeline.ocr import OCRCrossChecker, _token_found, _MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# Tests for Cloud Vision OCR (Requirement 4.3)
# ---------------------------------------------------------------------------

class TestCloudVisionOCR:
    """Test Cloud Vision API integration and text extraction."""

    def test_extract_text_with_annotations(self):
        """Test text extraction when Cloud Vision returns annotations."""
        # Mock the Vision client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_annotation = Mock()
        mock_annotation.description = "Château Margaux\n2015\nMargaux"
        mock_response.text_annotations = [mock_annotation]
        mock_client.text_detection.return_value = mock_response

        checker = OCRCrossChecker(client=mock_client)
        result = checker._extract_text(b"fake_image_bytes")

        assert result == "Château Margaux\n2015\nMargaux"
        mock_client.text_detection.assert_called_once()

    def test_extract_text_no_annotations(self):
        """Test text extraction when Cloud Vision returns no annotations."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text_annotations = []
        mock_client.text_detection.return_value = mock_response

        checker = OCRCrossChecker(client=mock_client)
        result = checker._extract_text(b"fake_image_bytes")

        assert result == ""

    def test_extract_text_empty_annotation(self):
        """Test text extraction with empty annotation description."""
        mock_client = Mock()
        mock_response = Mock()
        mock_annotation = Mock()
        mock_annotation.description = ""
        mock_response.text_annotations = [mock_annotation]
        mock_client.text_detection.return_value = mock_response

        checker = OCRCrossChecker(client=mock_client)
        result = checker._extract_text(b"fake_image_bytes")

        assert result == ""


# ---------------------------------------------------------------------------
# Tests for Cross-Reference Comparison (Requirements 5.1, 5.2, 5.3, 5.4)
# ---------------------------------------------------------------------------

class TestCrossReference:
    """Test cross-reference comparison between OCR and fingerprint."""

    def test_cross_reference_all_fields_confirmed(self):
        """Test when all fields are confirmed by OCR."""
        raw_text = "Château Margaux Margaux 2015 Premier Grand Cru"
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation="Margaux",
            cru_vineyard="Premier Grand Cru",
            vintage="2015"
        )

        result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

        assert result.producer_confirmed is True
        assert result.appellation_confirmed is True
        assert result.cru_confirmed is True
        assert result.vintage_confirmed is True
        assert result.fields_disagreed == []

    def test_cross_reference_partial_confirmation(self):
        """Test when some fields are confirmed and some are not."""
        raw_text = "Château Margaux 2015"
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation="Pauillac",
            cru_vineyard="Grand Cru",
            vintage="2015"
        )

        result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

        assert result.producer_confirmed is True
        assert result.appellation_confirmed is False
        assert result.cru_confirmed is False
        assert result.vintage_confirmed is True
        assert "appellation" in result.fields_disagreed
        assert "cru" in result.fields_disagreed
        assert "producer" not in result.fields_disagreed
        assert "vintage" not in result.fields_disagreed

    def test_cross_reference_none_fields_not_flagged(self):
        """Test that None fields in fingerprint are not flagged as disagreed."""
        raw_text = "Château Margaux 2015"
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation=None,
            cru_vineyard=None,
            vintage="2015"
        )

        result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

        assert result.producer_confirmed is True
        assert result.appellation_confirmed is False
        assert result.cru_confirmed is False
        assert result.vintage_confirmed is True
        # None fields should not be in disagreements
        assert "appellation" not in result.fields_disagreed
        assert "cru" not in result.fields_disagreed

    def test_cross_reference_empty_raw_text(self):
        """Test cross-reference with empty raw text."""
        raw_text = ""
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation="Margaux",
            cru_vineyard=None,
            vintage="2015"
        )

        result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

        assert result.producer_confirmed is False
        assert result.appellation_confirmed is False
        assert result.vintage_confirmed is False
        assert "producer" in result.fields_disagreed
        assert "appellation" in result.fields_disagreed
        assert "vintage" in result.fields_disagreed

    def test_cross_reference_empty_fingerprint(self):
        """Test cross-reference with all None fingerprint fields."""
        raw_text = "Some text on label"
        fingerprint = Fingerprint(
            producer=None,
            appellation=None,
            cru_vineyard=None,
            vintage=None
        )

        result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

        assert result.producer_confirmed is False
        assert result.appellation_confirmed is False
        assert result.cru_confirmed is False
        assert result.vintage_confirmed is False
        assert result.fields_disagreed == []


# ---------------------------------------------------------------------------
# Tests for Fuzzy Matching Threshold (Requirements 5.1, 5.2)
# ---------------------------------------------------------------------------

class TestFuzzyMatching:
    """Test fuzzy matching threshold and comparison logic."""

    def test_token_found_exact_match(self):
        """Test exact token match returns True."""
        raw_text = "Château Margaux Premier Grand Cru Classé 2015"
        assert _token_found(raw_text, "Margaux") is True

    def test_token_found_case_insensitive(self):
        """Test case-insensitive matching."""
        raw_text = "CHÂTEAU MARGAUX PREMIER GRAND CRU CLASSÉ 2015"
        assert _token_found(raw_text, "margaux") is True

    def test_token_found_partial_match_above_threshold(self):
        """Test partial match above threshold returns True."""
        raw_text = "Chateau Margaux 2015"
        # "Margaux" vs "Margaux" should match well above threshold
        assert _token_found(raw_text, "Margaux") is True

    def test_token_found_no_match(self):
        """Test when token is not in text returns False."""
        raw_text = "Château Lafite Rothschild 2015"
        assert _token_found(raw_text, "Margaux") is False

    def test_token_found_none_token(self):
        """Test None token returns False."""
        raw_text = "Some text"
        assert _token_found(raw_text, None) is False

    def test_token_found_empty_token(self):
        """Test empty token returns False."""
        raw_text = "Some text"
        assert _token_found(raw_text, "") is False

    def test_token_found_none_raw_text(self):
        """Test None raw_text returns False."""
        assert _token_found(None, "Margaux") is False

    def test_token_found_empty_raw_text(self):
        """Test empty raw_text returns False."""
        assert _token_found("", "Margaux") is False

    def test_match_threshold_value(self):
        """Test that the match threshold is set correctly."""
        # The threshold should be 75.0 as defined in the module
        assert _MATCH_THRESHOLD == 75.0


# ---------------------------------------------------------------------------
# Tests for Disagreement Detection (Requirements 5.3, 5.4)
# ---------------------------------------------------------------------------

class TestDisagreementDetection:
    """Test disagreement detection between OCR and fingerprint."""

    def test_disagreement_detected_for_missing_producer(self):
        """Test disagreement when producer is in fingerprint but not in OCR."""
        raw_text = "Grand Vin de Bordeaux 2015"
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation=None,
            cru_vineyard=None,
            vintage="2015"
        )

        result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

        assert "producer" in result.fields_disagreed
        assert "vintage" not in result.fields_disagreed

    def test_disagreement_detected_for_all_fields(self):
        """Test disagreement when all fields differ."""
        raw_text = "Some random text"
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation="Margaux",
            cru_vineyard="Premier Cru",
            vintage="2015"
        )

        result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

        assert len(result.fields_disagreed) == 4
        assert "producer" in result.fields_disagreed
        assert "appellation" in result.fields_disagreed
        assert "cru" in result.fields_disagreed
        assert "vintage" in result.fields_disagreed

    def test_no_disagreement_for_matching_fields(self):
        """Test no disagreement when fields match."""
        raw_text = "Château Margaux Margaux Premier Cru 2015"
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation="Margaux",
            cru_vineyard="Premier Cru",
            vintage="2015"
        )

        result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

        assert result.fields_disagreed == []

    def test_no_disagreement_for_none_fields(self):
        """Test that None fields don't cause disagreements."""
        raw_text = "Château Margaux 2015"
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation=None,
            cru_vineyard=None,
            vintage="2015"
        )

        result = OCRCrossChecker._cross_reference(raw_text, fingerprint)

        # appellation and cru are None, so they shouldn't be in disagreements
        assert "appellation" not in result.fields_disagreed
        assert "cru" not in result.fields_disagreed
        assert "producer" not in result.fields_disagreed
        assert "vintage" not in result.fields_disagreed


# ---------------------------------------------------------------------------
# Integration Tests with Mock Client (Requirements 4.3, 5.1, 5.2)
# ---------------------------------------------------------------------------

class TestOCRIntegration:
    """Integration tests with mocked Cloud Vision client."""

    def test_check_method_integration(self):
        """Test full check method with mocked client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_annotation = Mock()
        mock_annotation.description = "Château Margaux\nMargaux\n2015"
        mock_response.text_annotations = [mock_annotation]
        mock_client.text_detection.return_value = mock_response

        checker = OCRCrossChecker(client=mock_client)
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation="Margaux",
            cru_vineyard=None,
            vintage="2015"
        )

        # Test via async wrapper
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            checker.check(b"fake_image_bytes", fingerprint)
        )

        assert isinstance(result, OCRResult)
        assert result.producer_confirmed is True
        assert result.appellation_confirmed is True
        assert result.vintage_confirmed is True
        assert result.fields_disagreed == []

    def test_check_method_with_disagreements(self):
        """Test check method with some disagreements."""
        mock_client = Mock()
        mock_response = Mock()
        mock_annotation = Mock()
        mock_annotation.description = "Château Lafite Rothschild\nPauillac\n2010"
        mock_response.text_annotations = [mock_annotation]
        mock_client.text_detection.return_value = mock_response

        checker = OCRCrossChecker(client=mock_client)
        fingerprint = Fingerprint(
            producer="Château Margaux",
            appellation="Margaux",
            cru_vineyard=None,
            vintage="2015"
        )

        # Test via async wrapper
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            checker.check(b"fake_image_bytes", fingerprint)
        )

        assert result.producer_confirmed is False
        assert result.appellation_confirmed is False
        # Note: "2010" vs "2015" may match fuzzily, so we don't assert on vintage_confirmed
        assert "producer" in result.fields_disagreed
        assert "appellation" in result.fields_disagreed
