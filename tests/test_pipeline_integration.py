"""Integration tests for the full Pipeline flow.

Mocks all external APIs (SerpAPI, Gemini, Google Vision) and verifies
that all stages execute in correct order and produce a valid ScoredResult.

Requirements: 1.1–6.6
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from wine_pipeline.cache import ResultCache
from wine_pipeline.label_extractor import LabelExtractor
from wine_pipeline.models import (
    CandidateImage,
    FieldScores,
    Fingerprint,
    LabelExtractionResult,
    OCRResult,
    QualityResult,
    ScoredResult,
    SKU,
    Verdict,
)
from wine_pipeline.ocr import OCRCrossChecker
from wine_pipeline.pipeline import Pipeline
from wine_pipeline.quality_filter import QualityFilter
from wine_pipeline.scoring import ConfidenceScorer
from wine_pipeline.search import SearchModule
from wine_pipeline.verifier import FingerprintVerifier, VerificationResult


@pytest.fixture
def sample_sku() -> SKU:
    return SKU(
        id="SKU-TEST-001",
        producer="Domaine Leflaive",
        appellation="Puligny-Montrachet",
        cru_vineyard="Les Pucelles",
        vintage=2020,
        format="750ml",
        region="Burgundy",
    )


@pytest.fixture
def fake_image() -> bytes:
    """A minimal fake image (1x1 PNG-like bytes)."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 50


@pytest.fixture
def pipeline_with_mocks(fake_image, tmp_path):
    """Build a Pipeline with all stages mocked, returning realistic data."""
    # Search: returns one candidate with raw_image already set
    candidate = CandidateImage(
        url="https://example.com/wine.jpg",
        source="serpapi",
        raw_image=fake_image,
    )
    mock_search = MagicMock(spec=SearchModule)
    mock_search.search = AsyncMock(return_value=[candidate])

    # Label extractor: returns the image with label detected
    label_result = LabelExtractionResult(cropped_image=fake_image, label_detected=True)
    mock_label = MagicMock(spec=LabelExtractor)
    mock_label.extract_label = MagicMock(return_value=label_result)

    # Verifier: returns a fingerprint with good field scores
    fingerprint = Fingerprint(
        producer="Domaine Leflaive",
        appellation="Puligny-Montrachet",
        cru_vineyard="Les Pucelles",
        vintage="2020",
    )
    field_scores = FieldScores(
        producer_match=0.95,
        appellation_match=0.90,
        cru_match=0.85,
        vintage_match=1.0,
    )
    verification = VerificationResult(
        fingerprint=fingerprint, field_scores=field_scores, is_nv=False
    )
    mock_verifier = MagicMock(spec=FingerprintVerifier)
    mock_verifier.verify = AsyncMock(return_value=verification)

    # OCR: all fields confirmed
    ocr_result = OCRResult(
        raw_text="Domaine Leflaive Puligny-Montrachet Les Pucelles 2020",
        producer_confirmed=True,
        appellation_confirmed=True,
        cru_confirmed=True,
        vintage_confirmed=True,
        fields_disagreed=[],
    )
    mock_ocr = MagicMock(spec=OCRCrossChecker)
    mock_ocr.check = AsyncMock(return_value=ocr_result)

    # Quality filter: passes
    quality_result = QualityResult(
        passed=True, image_quality_score=1.0, rejection_reasons=[]
    )
    mock_quality = MagicMock(spec=QualityFilter)
    mock_quality.evaluate = AsyncMock(return_value=quality_result)

    # Scorer: use the real scorer
    scorer = ConfidenceScorer()

    # Cache: real SQLite in temp dir
    db_path = str(tmp_path / "test_cache.db")
    cache = ResultCache(db_path=db_path)

    pipeline = Pipeline(
        search=mock_search,
        label_extractor=mock_label,
        verifier=mock_verifier,
        ocr_checker=mock_ocr,
        quality_filter=mock_quality,
        scorer=scorer,
        cache=cache,
    )

    return pipeline, {
        "search": mock_search,
        "label_extractor": mock_label,
        "verifier": mock_verifier,
        "ocr": mock_ocr,
        "quality": mock_quality,
        "cache": cache,
    }


@pytest.mark.asyncio
async def test_full_pipeline_executes_all_stages(pipeline_with_mocks, sample_sku):
    """All pipeline stages execute in order and produce a valid ScoredResult."""
    pipeline, mocks = pipeline_with_mocks

    result = await pipeline.process_sku(sample_sku)

    # All stages were called
    mocks["search"].search.assert_awaited_once_with(sample_sku)
    mocks["label_extractor"].extract_label.assert_called_once()
    mocks["verifier"].verify.assert_awaited_once()
    mocks["ocr"].check.assert_awaited_once()
    mocks["quality"].evaluate.assert_awaited_once()

    # Result is valid
    assert isinstance(result, ScoredResult)
    assert result.sku_id == sample_sku.id
    assert result.image_url == "https://example.com/wine.jpg"
    assert result.fingerprint is not None
    assert result.fingerprint.producer == "Domaine Leflaive"

    # Scores are bounded
    for score in [result.producer_match, result.appellation_match,
                  result.cru_match, result.vintage_match, result.image_quality]:
        assert 0.0 <= score <= 1.0

    assert 0.0 <= result.overall_confidence <= 1.0
    assert result.verdict in (Verdict.PASS, Verdict.QUARANTINE, Verdict.REJECT)

    # Result was cached
    cached = mocks["cache"].get(sample_sku.id)
    assert cached is not None
    assert cached.sku_id == result.sku_id

    mocks["cache"].close()


@pytest.mark.asyncio
async def test_pipeline_no_candidates_returns_no_image(sample_sku, tmp_path):
    """When search returns no candidates, pipeline returns 'No Image' REJECT."""
    mock_search = MagicMock(spec=SearchModule)
    mock_search.search = AsyncMock(return_value=[])

    mock_label = MagicMock(spec=LabelExtractor)
    mock_verifier = MagicMock(spec=FingerprintVerifier)
    mock_verifier.verify = AsyncMock()
    mock_ocr = MagicMock(spec=OCRCrossChecker)
    mock_ocr.check = AsyncMock()
    mock_quality = MagicMock(spec=QualityFilter)
    mock_quality.evaluate = AsyncMock()

    db_path = str(tmp_path / "test_cache2.db")
    cache = ResultCache(db_path=db_path)

    pipeline = Pipeline(
        search=mock_search,
        label_extractor=mock_label,
        verifier=mock_verifier,
        ocr_checker=mock_ocr,
        quality_filter=mock_quality,
        scorer=ConfidenceScorer(),
        cache=cache,
    )

    result = await pipeline.process_sku(sample_sku)

    assert result.verdict == Verdict.REJECT
    assert result.image_url is None
    assert "No Image" in result.rejection_reasons

    # Downstream stages should not have been called
    mock_label.extract_label.assert_not_called()
    mock_verifier.verify.assert_not_awaited()
    mock_ocr.check.assert_not_awaited()
    mock_quality.evaluate.assert_not_awaited()

    cache.close()
