"""Unit tests for the Fingerprint Verifier.

Tests Gemini extraction, JSON parsing, and field comparison logic.
Validates: Requirements 4.1, 4.2, 6.1, 6.2, 6.3, 6.4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wine_pipeline.models import Fingerprint, SKU, FieldScores
from wine_pipeline.verifier import FingerprintVerifier, VerificationResult


# ---------------------------------------------------------------------------
# Sub-task 4.1: Test Gemini extraction and JSON parsing
# Validates: Requirements 4.1, 4.2
# ---------------------------------------------------------------------------

class TestParseFingerprint:
    """Test JSON parsing from Gemini response."""

    def test_parse_valid_json(self):
        """Valid JSON with all fields produces correct Fingerprint."""
        raw_text = '{"producer": "Domaine Test", "appellation": "Burgundy", "cru_vineyard": "Les Cru", "vintage": "2020"}'
        fp = FingerprintVerifier._parse_fingerprint(raw_text)

        assert fp.producer == "Domaine Test"
        assert fp.appellation == "Burgundy"
        assert fp.cru_vineyard == "Les Cru"
        assert fp.vintage == "2020"

    def test_parse_json_with_nulls(self):
        """JSON with null values produces Fingerprint with None fields."""
        raw_text = '{"producer": "Test Winery", "appellation": null, "cru_vineyard": null, "vintage": null}'
        fp = FingerprintVerifier._parse_fingerprint(raw_text)

        assert fp.producer == "Test Winery"
        assert fp.appellation is None
        assert fp.cru_vineyard is None
        assert fp.vintage is None

    def test_parse_json_with_missing_keys(self):
        """JSON missing expected keys produces Fingerprint with None fields."""
        raw_text = '{"producer": "Test"}'
        fp = FingerprintVerifier._parse_fingerprint(raw_text)

        assert fp.producer == "Test"
        assert fp.appellation is None
        assert fp.cru_vineyard is None
        assert fp.vintage is None

    def test_parse_json_with_markdown_fences(self):
        """JSON wrapped in markdown code fences is handled correctly."""
        raw_text = '''```json
{"producer": "Winery", "appellation": "Region", "cru_vineyard": "Vineyard", "vintage": "2021"}
```'''
        fp = FingerprintVerifier._parse_fingerprint(raw_text)

        assert fp.producer == "Winery"
        assert fp.appellation == "Region"
        assert fp.cru_vineyard == "Vineyard"
        assert fp.vintage == "2021"

    def test_parse_empty_string(self):
        """Empty string produces Fingerprint with all None fields."""
        fp = FingerprintVerifier._parse_fingerprint("")

        assert fp.producer is None
        assert fp.appellation is None
        assert fp.cru_vineyard is None
        assert fp.vintage is None

    def test_parse_invalid_json(self):
        """Invalid JSON produces Fingerprint with all None fields."""
        raw_text = "This is not JSON at all"
        fp = FingerprintVerifier._parse_fingerprint(raw_text)

        assert fp.producer is None
        assert fp.appellation is None
        assert fp.cru_vineyard is None
        assert fp.vintage is None

    def test_parse_json_with_numeric_vintage(self):
        """JSON with numeric vintage converts to string."""
        raw_text = '{"producer": "Test", "appellation": "Test", "cru_vineyard": null, "vintage": 2020}'
        fp = FingerprintVerifier._parse_fingerprint(raw_text)

        assert fp.vintage == "2020"

    def test_parse_json_with_empty_strings(self):
        """JSON with empty strings produces None fields."""
        raw_text = '{"producer": "", "appellation": "  ", "cru_vineyard": null, "vintage": "2020"}'
        fp = FingerprintVerifier._parse_fingerprint(raw_text)

        assert fp.producer is None
        assert fp.appellation is None
        assert fp.cru_vineyard is None
        assert fp.vintage == "2020"

    def test_parse_non_dict_json(self):
        """Non-dict JSON produces Fingerprint with all None fields."""
        raw_text = '["not", "a", "dict"]'
        fp = FingerprintVerifier._parse_fingerprint(raw_text)

        assert fp.producer is None
        assert fp.appellation is None
        assert fp.cru_vineyard is None
        assert fp.vintage is None


class TestGeminiExtraction:
    """Test Gemini API call with label image."""

    @pytest.mark.asyncio
    async def test_extract_fingerprint_calls_gemini(self):
        """Gemini API is called with correct parameters."""
        fake_image = b"fake_image_bytes"

        # Mock the Gemini client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"producer": "Test Winery", "appellation": "Test Region", "cru_vineyard": null, "vintage": "2020"}'
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        verifier = FingerprintVerifier(client=mock_client)
        fp = await verifier._extract_fingerprint(fake_image)

        assert fp.producer == "Test Winery"
        assert fp.appellation == "Test Region"
        mock_client.aio.models.generate_content.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extract_fingerprint_handles_empty_response(self):
        """Empty Gemini response produces Fingerprint with all None fields."""
        fake_image = b"fake_image_bytes"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        verifier = FingerprintVerifier(client=mock_client)
        fp = await verifier._extract_fingerprint(fake_image)

        assert fp.producer is None
        assert fp.appellation is None
        assert fp.cru_vineyard is None
        assert fp.vintage is None


# ---------------------------------------------------------------------------
# Sub-task 4.2: Test field comparison logic
# Validates: Requirements 6.1, 6.2, 6.3, 6.4
# ---------------------------------------------------------------------------

class TestFieldComparison:
    """Test direct and cross-field matching."""

    def test_direct_field_matching(self):
        """Direct field matching produces correct scores."""
        fingerprint = Fingerprint(
            producer="Domaine Test",
            appellation="Burgundy",
            cru_vineyard="Les Cru",
            vintage="2020",
        )
        sku = SKU(
            id="test-001",
            producer="Domaine Test",
            appellation="Burgundy",
            cru_vineyard="Les Cru",
            vintage=2020,
            format="750ml",
            region="France",
        )

        scores = FingerprintVerifier._compare_fields(fingerprint, sku)

        assert scores.producer_match == 1.0
        assert scores.appellation_match == 1.0
        assert scores.cru_match == 1.0
        assert scores.vintage_match == 1.0

    def test_fuzzy_field_matching(self):
        """Fuzzy matching handles minor differences."""
        fingerprint = Fingerprint(
            producer="Domaine Test Winery",
            appellation="Burgundy Region",
            cru_vineyard="Les Cru Vineyard",
            vintage="2020",
        )
        sku = SKU(
            id="test-002",
            producer="Domaine Test",
            appellation="Burgundy",
            cru_vineyard="Les Cru",
            vintage=2020,
            format="750ml",
            region="France",
        )

        scores = FingerprintVerifier._compare_fields(fingerprint, sku)

        # Scores should be high but not perfect due to fuzzy matching
        assert scores.producer_match > 0.5
        assert scores.appellation_match > 0.5
        assert scores.cru_match > 0.5
        assert scores.vintage_match == 1.0

    def test_missing_field_scores_zero(self):
        """Missing fingerprint fields score 0.0."""
        fingerprint = Fingerprint(
            producer=None,
            appellation=None,
            cru_vineyard=None,
            vintage=None,
        )
        sku = SKU(
            id="test-003",
            producer="Domaine Test",
            appellation="Burgundy",
            cru_vineyard="Les Cru",
            vintage=2020,
            format="750ml",
            region="France",
        )

        scores = FingerprintVerifier._compare_fields(fingerprint, sku)

        assert scores.producer_match == 0.0
        assert scores.appellation_match == 0.0
        assert scores.cru_match == 0.0
        assert scores.vintage_match == 0.0

    def test_cross_field_matching_producer_in_appellation(self):
        """Cross-field matching finds producer in appellation field."""
        # This simulates OCR putting the cru name in the appellation field
        fingerprint = Fingerprint(
            producer="Domaine Test",
            appellation="Les Cru",  # cru ended up in appellation
            cru_vineyard=None,
            vintage="2020",
        )
        sku = SKU(
            id="test-004",
            producer="Domaine Test",
            appellation="Burgundy",
            cru_vineyard="Les Cru",  # We're looking for this in fingerprint
            vintage=2020,
            format="750ml",
            region="France",
        )

        scores = FingerprintVerifier._compare_fields(fingerprint, sku)

        # Cross-field matching should find "Les Cru" in all_extracted
        assert scores.cru_match > 0.0

    def test_cross_field_matching_appellation_in_cru(self):
        """Cross-field matching finds appellation in cru field."""
        fingerprint = Fingerprint(
            producer="Domaine Test",
            appellation=None,
            cru_vineyard="Burgundy",  # appellation ended up in cru
            vintage="2020",
        )
        sku = SKU(
            id="test-005",
            producer="Domaine Test",
            appellation="Burgundy",
            cru_vineyard=None,
            vintage=2020,
            format="750ml",
            region="France",
        )

        scores = FingerprintVerifier._compare_fields(fingerprint, sku)

        # Cross-field matching should find "Burgundy" in all_extracted
        assert scores.appellation_match > 0.0


# ---------------------------------------------------------------------------
# Sub-task 4.3: Test NV wine handling
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------

class TestNVWineHandling:
    """Test vintage score assignment for NV wines."""

    def test_nv_wine_vintage_score_zero(self):
        """NV wines always get vintage score of 0.0."""
        fingerprint = Fingerprint(
            producer="Champagne House",
            appellation="Champagne",
            cru_vineyard=None,
            vintage="NV",  # Some OCR might extract this
        )
        sku = SKU(
            id="nv-001",
            producer="Champagne House",
            appellation="Champagne",
            cru_vineyard=None,
            vintage=None,  # NV wine
            format="750ml",
            region="France",
        )

        scores = FingerprintVerifier._compare_fields(fingerprint, sku)

        # Even if fingerprint has vintage, NV wine gets 0.0
        assert scores.vintage_match == 0.0

    def test_nv_wine_with_null_fingerprint_vintage(self):
        """NV wine with null fingerprint vintage gets 0.0 vintage score."""
        fingerprint = Fingerprint(
            producer="Champagne House",
            appellation="Champagne",
            cru_vineyard=None,
            vintage=None,
        )
        sku = SKU(
            id="nv-002",
            producer="Champagne House",
            appellation="Champagne",
            cru_vineyard=None,
            vintage=None,
            format="750ml",
            region="France",
        )

        scores = FingerprintVerifier._compare_fields(fingerprint, sku)

        assert scores.vintage_match == 0.0

    def test_verify_returns_is_nv_flag(self):
        """verify() returns is_nv=True for NV wines."""
        fake_image = b"fake_image_bytes"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"producer": "Champagne House", "appellation": "Champagne", "cru_vineyard": null, "vintage": null}'
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        verifier = FingerprintVerifier(client=mock_client)
        sku = SKU(
            id="nv-003",
            producer="Champagne House",
            appellation="Champagne",
            cru_vineyard=None,
            vintage=None,
            format="750ml",
            region="France",
        )

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(verifier.verify(fake_image, sku))

        assert result.is_nv is True
        assert result.field_scores.vintage_match == 0.0


# ---------------------------------------------------------------------------
# Sub-task 4.5: Additional unit tests for cross-field matching
# Validates: Requirements 4.1, 4.2, 6.1, 6.2, 6.3, 6.4
# ---------------------------------------------------------------------------

class TestCrossFieldMatching:
    """Test various cross-field matching scenarios."""

    def test_cross_field_uses_max_score(self):
        """Cross-field matching takes max of direct and cross-field scores."""
        # Appellation appears in cru_vineyard field
        fingerprint = Fingerprint(
            producer="Domaine Test",
            appellation="Wrong Appellation",  # Low direct match
            cru_vineyard="Burgundy",  # Actual appellation is here
            vintage="2020",
        )
        sku = SKU(
            id="cross-001",
            producer="Domaine Test",
            appellation="Burgundy",
            cru_vineyard=None,
            vintage=2020,
            format="750ml",
            region="France",
        )

        scores = FingerprintVerifier._compare_fields(fingerprint, sku)

        # Cross-field match should boost appellation score
        assert scores.appellation_match > 0.0

    def test_all_fields_combined_for_cross_match(self):
        """All extracted text is combined for cross-field matching."""
        fingerprint = Fingerprint(
            producer="Producer Name Les Cru",  # cru embedded in producer
            appellation="Burgundy",
            cru_vineyard=None,
            vintage="2020",
        )
        sku = SKU(
            id="cross-002",
            producer="Producer Name",
            appellation="Burgundy",
            cru_vineyard="Les Cru",
            vintage=2020,
            format="750ml",
            region="France",
        )

        scores = FingerprintVerifier._compare_fields(fingerprint, sku)

        # Cross-field should find "Les Cru" in producer text
        assert scores.cru_match > 0.0
