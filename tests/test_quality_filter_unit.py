"""Unit tests for Quality Filter.

Tests Gemini API call with label image and JSON parsing from Gemini response.
Validates: Requirements 7.1
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import AsyncMock, MagicMock

from wine_pipeline.quality_filter import QualityFilter, QUALITY_PROMPT
from wine_pipeline.models import QualityResult


class TestQualityFilterUnit:
    """Unit tests for QualityFilter."""

    def test_parse_response_all_pass(self):
        """Test parsing when all criteria pass."""
        raw_text = """
        {
            "watermarks": "pass",
            "full_bottle_visible": "pass",
            "sharp_legible_label": "pass",
            "single_bottle_only": "pass",
            "clean_background": "pass",
            "no_lifestyle_props": "pass",
            "not_ai_generated": "pass"
        }
        """
        result = QualityFilter._parse_response(raw_text)

        assert result.passed is True
        assert result.image_quality_score == 1.0
        assert len(result.rejection_reasons) == 0

    def test_parse_response_some_fail(self):
        """Test parsing when some criteria fail."""
        raw_text = """
        {
            "watermarks": "pass",
            "full_bottle_visible": "pass",
            "sharp_legible_label": "fail",
            "single_bottle_only": "pass",
            "clean_background": "fail",
            "no_lifestyle_props": "pass",
            "not_ai_generated": "pass"
        }
        """
        result = QualityFilter._parse_response(raw_text)

        assert result.passed is False
        assert result.image_quality_score == pytest.approx(5 / 7, rel=0.01)
        assert len(result.rejection_reasons) == 2
        assert "Image is blurry / label text not legible" in result.rejection_reasons
        assert "Background is cluttered / not a clean product shot" in result.rejection_reasons

    def test_parse_response_all_fail(self):
        """Test parsing when all criteria fail."""
        raw_text = """
        {
            "watermarks": "fail",
            "full_bottle_visible": "fail",
            "sharp_legible_label": "fail",
            "single_bottle_only": "fail",
            "clean_background": "fail",
            "no_lifestyle_props": "fail",
            "not_ai_generated": "fail"
        }
        """
        result = QualityFilter._parse_response(raw_text)

        assert result.passed is False
        assert result.image_quality_score == 0.0
        assert len(result.rejection_reasons) == 7

    def test_parse_response_with_markdown_fences(self):
        """Test parsing when response includes markdown fences."""
        raw_text = """
        ```json
        {
            "watermarks": "pass",
            "full_bottle_visible": "pass",
            "sharp_legible_label": "pass",
            "single_bottle_only": "pass",
            "clean_background": "pass",
            "no_lifestyle_props": "pass",
            "not_ai_generated": "pass"
        }
        ```
        """
        result = QualityFilter._parse_response(raw_text)

        assert result.passed is True
        assert result.image_quality_score == 1.0
        assert len(result.rejection_reasons) == 0

    def test_parse_response_invalid_json(self):
        """Test parsing when response is invalid JSON."""
        raw_text = "This is not valid JSON"

        result = QualityFilter._parse_response(raw_text)

        assert result.passed is False
        assert result.image_quality_score == 0.0
        assert len(result.rejection_reasons) == 1
        assert "Unable to parse" in result.rejection_reasons[0]

    def test_parse_response_missing_criteria(self):
        """Test parsing when some criteria are missing (defaults to fail)."""
        raw_text = """
        {
            "watermarks": "pass",
            "full_bottle_visible": "pass"
        }
        """
        result = QualityFilter._parse_response(raw_text)

        assert result.passed is False
        assert len(result.rejection_reasons) == 5  # Missing criteria default to fail

    def test_parse_response_non_dict(self):
        """Test parsing when response is not a dictionary."""
        raw_text = '["watermarks", "blur_glare"]'

        result = QualityFilter._parse_response(raw_text)

        assert result.passed is False
        assert result.image_quality_score == 0.0
        assert "Unable to parse" in result.rejection_reasons[0]

    @pytest.mark.asyncio
    async def test_evaluate_calls_gemini_api(self):
        """Test that evaluate() calls Gemini API with correct parameters."""
        # Create a mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """
        {
            "watermarks": "pass",
            "full_bottle_visible": "pass",
            "sharp_legible_label": "pass",
            "single_bottle_only": "pass",
            "clean_background": "pass",
            "no_lifestyle_props": "pass",
            "not_ai_generated": "pass"
        }
        """

        # Set up async mock
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        # Create filter with mock client
        quality_filter = QualityFilter(client=mock_client, model="test-model")

        # Call evaluate
        test_image = b"fake image bytes"
        result = await quality_filter.evaluate(test_image)

        # Verify API was called
        mock_client.aio.models.generate_content.assert_called_once()
        call_args = mock_client.aio.models.generate_content.call_args
        assert call_args.kwargs["model"] == "test-model"
        assert len(call_args.kwargs["contents"]) == 2

        # Verify result
        assert result.passed is True
        assert result.image_quality_score == 1.0


    def test_rejection_reason_mapping(self):
        """Test that each criterion maps to correct human-readable description.
        
        Validates: Requirements 7.2
        """
        # Test each criterion individually
        criteria_and_descriptions = [
            ("watermarks", "Image contains visible watermarks"),
            ("full_bottle_visible", "Bottle is cropped / not fully visible"),
            ("sharp_legible_label", "Image is blurry / label text not legible"),
            ("single_bottle_only", "Image contains multiple bottles / group shot"),
            ("clean_background", "Background is cluttered / not a clean product shot"),
            ("no_lifestyle_props", "Image contains lifestyle props or non-product elements"),
            ("not_ai_generated", "Image appears to be AI-generated"),
        ]
        
        for criterion, expected_description in criteria_and_descriptions:
            # Create a response with only this criterion failing
            pass_criteria = [
                "watermarks",
                "full_bottle_visible",
                "sharp_legible_label",
                "single_bottle_only",
                "clean_background",
                "no_lifestyle_props",
                "not_ai_generated",
            ]
            response_dict = {c: "pass" for c in pass_criteria}
            response_dict[criterion] = "fail"
            
            import json
            raw_text = json.dumps(response_dict)
            result = QualityFilter._parse_response(raw_text)
            
            assert result.passed is False
            assert expected_description in result.rejection_reasons, \
                f"Expected '{expected_description}' for criterion '{criterion}'"

    def test_quality_score_calculation(self):
        """Test quality score calculation based on pass/fail ratio.
        
        Validates: Requirements 7.3
        """
        # 7/7 pass = 1.0
        result = QualityFilter._parse_response(
            '{"watermarks":"pass","full_bottle_visible":"pass","sharp_legible_label":"pass",'
            '"single_bottle_only":"pass","clean_background":"pass","no_lifestyle_props":"pass",'
            '"not_ai_generated":"pass"}'
        )
        assert result.image_quality_score == pytest.approx(1.0, rel=0.01)
        
        # 6/7 pass
        result = QualityFilter._parse_response(
            '{"watermarks":"pass","full_bottle_visible":"pass","sharp_legible_label":"pass",'
            '"single_bottle_only":"pass","clean_background":"pass","no_lifestyle_props":"pass",'
            '"not_ai_generated":"fail"}'
        )
        assert result.image_quality_score == pytest.approx(6/7, rel=0.01)
        
        # 3/7 pass
        result = QualityFilter._parse_response(
            '{"watermarks":"pass","full_bottle_visible":"fail","sharp_legible_label":"pass",'
            '"single_bottle_only":"fail","clean_background":"fail","no_lifestyle_props":"pass",'
            '"not_ai_generated":"fail"}'
        )
        assert result.image_quality_score == pytest.approx(3/7, rel=0.01)
        
        # 0/7 pass = 0.0
        result = QualityFilter._parse_response(
            '{"watermarks":"fail","full_bottle_visible":"fail","sharp_legible_label":"fail",'
            '"single_bottle_only":"fail","clean_background":"fail","no_lifestyle_props":"fail",'
            '"not_ai_generated":"fail"}'
        )
        assert result.image_quality_score == pytest.approx(0.0, rel=0.01)

    def test_rejection_reasons_below_threshold(self):
        """Test that images below threshold are marked for rejection.
        
        Validates: Requirements 7.4
        """
        # Any failing criterion should cause rejection
        result = QualityFilter._parse_response(
            '{"watermarks":"pass","full_bottle_visible":"pass","sharp_legible_label":"pass",'
            '"single_bottle_only":"pass","clean_background":"pass","no_lifestyle_props":"fail",'
            '"not_ai_generated":"pass"}'
        )
        assert result.passed is False
        assert len(result.rejection_reasons) > 0
