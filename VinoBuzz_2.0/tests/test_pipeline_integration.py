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


@pytest.mark.asyncio
async def test_pipeline_cache_bypass_reprocesses_sku(sample_sku, fake_image, tmp_path):
    """When bypass_cache=True, pipeline ignores cached result and reprocesses."""
    # Set up a cached result
    cached_result = ScoredResult(
        sku_id=sample_sku.id,
        image_url="https://old-url.com/wine.jpg",
        producer_match=0.5,
        appellation_match=0.5,
        cru_match=0.5,
        vintage_match=0.5,
        image_quality=0.5,
        overall_confidence=0.5,
        verdict=Verdict.QUARANTINE,
        fingerprint=None,
        rejection_reasons=[],
    )

    db_path = str(tmp_path / "test_cache_bypass.db")
    cache = ResultCache(db_path=db_path)
    cache.put(sample_sku.id, cached_result)

    # Set up mocks that will return new data
    candidate = CandidateImage(
        url="https://new-url.com/wine.jpg",
        source="test",
        raw_image=fake_image,
    )
    mock_search = MagicMock(spec=SearchModule)
    mock_search.search = AsyncMock(return_value=[candidate])

    label_result = LabelExtractionResult(cropped_image=fake_image, label_detected=True)
    mock_label = MagicMock(spec=LabelExtractor)
    mock_label.extract_label = MagicMock(return_value=label_result)

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

    ocr_result = OCRResult(
        raw_text="Domaine Leflaive",
        producer_confirmed=True,
        appellation_confirmed=True,
        cru_confirmed=True,
        vintage_confirmed=True,
        fields_disagreed=[],
    )
    mock_ocr = MagicMock(spec=OCRCrossChecker)
    mock_ocr.check = AsyncMock(return_value=ocr_result)

    quality_result = QualityResult(
        passed=True, image_quality_score=1.0, rejection_reasons=[]
    )
    mock_quality = MagicMock(spec=QualityFilter)
    mock_quality.evaluate = AsyncMock(return_value=quality_result)

    pipeline = Pipeline(
        search=mock_search,
        label_extractor=mock_label,
        verifier=mock_verifier,
        ocr_checker=mock_ocr,
        quality_filter=mock_quality,
        scorer=ConfidenceScorer(),
        cache=cache,
    )

    # Process with bypass=True
    result = await pipeline.process_sku(sample_sku, bypass_cache=True)

    # Search should have been called (bypassed cache)
    mock_search.search.assert_awaited_once_with(sample_sku)

    # Result should be the new one, not the cached one
    assert result.image_url == "https://new-url.com/wine.jpg"
    assert result.overall_confidence > cached_result.overall_confidence

    # Verify cache was updated with new result
    updated_cache = cache.get(sample_sku.id)
    assert updated_cache.image_url == "https://new-url.com/wine.jpg"

    cache.close()


# ---------------------------------------------------------------------------
# Task 9.2: Batch processing tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_processing_preserves_order(fake_image, tmp_path):
    """Batch processing returns results in the same order as input SKUs."""
    # Create multiple SKUs
    skus = [
        SKU(
            id=f"SKU-{i:03d}",
            producer=f"Producer {i}",
            appellation=f"Appellation {i}",
            cru_vineyard=f"Cru {i}" if i % 2 == 0 else None,
            vintage=2020 + (i % 5),
            format="750ml",
            region="Burgundy",
        )
        for i in range(5)
    ]

    # Set up mocks
    mock_search = MagicMock(spec=SearchModule)
    # Each SKU gets a different candidate to distinguish results
    async def mock_search_impl(sku):
        idx = int(sku.id.split("-")[1])
        return [CandidateImage(
            url=f"https://example.com/wine_{idx}.jpg",
            source="test",
            raw_image=fake_image,
        )]
    mock_search.search = AsyncMock(side_effect=mock_search_impl)

    label_result = LabelExtractionResult(cropped_image=fake_image, label_detected=True)
    mock_label = MagicMock(spec=LabelExtractor)
    mock_label.extract_label = MagicMock(return_value=label_result)

    async def mock_verify_impl(image, sku):
        idx = int(sku.id.split("-")[1])
        return VerificationResult(
            fingerprint=Fingerprint(
                producer=f"Producer {idx}",
                appellation=f"Appellation {idx}",
                cru_vineyard=f"Cru {idx}" if idx % 2 == 0 else None,
                vintage=str(2020 + (idx % 5)),
            ),
            field_scores=FieldScores(
                producer_match=0.9,
                appellation_match=0.9,
                cru_match=0.9,
                vintage_match=0.9,
            ),
            is_nv=False,
        )
    mock_verifier = MagicMock(spec=FingerprintVerifier)
    mock_verifier.verify = AsyncMock(side_effect=mock_verify_impl)

    mock_ocr = MagicMock(spec=OCRCrossChecker)
    mock_ocr.check = AsyncMock(return_value=OCRResult(
        raw_text="text",
        producer_confirmed=True,
        appellation_confirmed=True,
        cru_confirmed=True,
        vintage_confirmed=True,
        fields_disagreed=[],
    ))

    mock_quality = MagicMock(spec=QualityFilter)
    mock_quality.evaluate = AsyncMock(return_value=QualityResult(
        passed=True, image_quality_score=1.0, rejection_reasons=[]
    ))

    db_path = str(tmp_path / "test_batch_order.db")
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

    results = await pipeline.process_batch(skus, bypass_cache=True, max_concurrency=2)

    # Verify order is preserved
    assert len(results) == len(skus)
    for i, (sku, result) in enumerate(zip(skus, results)):
        assert result.sku_id == sku.id, f"Order mismatch at index {i}"
        expected_idx = i
        assert result.image_url == f"https://example.com/wine_{expected_idx}.jpg"

    cache.close()


@pytest.mark.asyncio
async def test_batch_processing_with_semaphore_limits_concurrency(fake_image, tmp_path):
    """Batch processing respects max_concurrency limit using semaphore."""
    import asyncio
    from collections import defaultdict
    import time

    concurrent_count = 0
    max_observed_concurrency = 0
    lock = asyncio.Lock()

    skus = [SKU(
        id=f"SKU-{i}",
        producer=f"Producer {i}",
        appellation=f"Appellation {i}",
        cru_vineyard=None,
        vintage=2020,
        format="750ml",
        region="Burgundy",
    ) for i in range(10)]

    mock_search = MagicMock(spec=SearchModule)
    async def track_concurrency(sku):
        nonlocal concurrent_count, max_observed_concurrency
        async with lock:
            concurrent_count += 1
            max_observed_concurrency = max(max_observed_concurrency, concurrent_count)

        # Simulate some async work
        await asyncio.sleep(0.01)

        async with lock:
            concurrent_count -= 1

        return [CandidateImage(url=f"https://example.com/{sku.id}.jpg", source="test", raw_image=fake_image)]
    mock_search.search = AsyncMock(side_effect=track_concurrency)

    mock_label = MagicMock(spec=LabelExtractor)
    mock_label.extract_label = MagicMock(return_value=LabelExtractionResult(
        cropped_image=fake_image, label_detected=True
    ))

    mock_verifier = MagicMock(spec=FingerprintVerifier)
    mock_verifier.verify = AsyncMock(return_value=VerificationResult(
        fingerprint=Fingerprint(producer="P", appellation="A", cru_vineyard=None, vintage="2020"),
        field_scores=FieldScores(producer_match=0.9, appellation_match=0.9, cru_match=0.9, vintage_match=0.9),
        is_nv=False,
    ))

    mock_ocr = MagicMock(spec=OCRCrossChecker)
    mock_ocr.check = AsyncMock(return_value=OCRResult(
        raw_text="text", producer_confirmed=True, appellation_confirmed=True,
        cru_confirmed=True, vintage_confirmed=True, fields_disagreed=[],
    ))

    mock_quality = MagicMock(spec=QualityFilter)
    mock_quality.evaluate = AsyncMock(return_value=QualityResult(
        passed=True, image_quality_score=1.0, rejection_reasons=[],
    ))

    db_path = str(tmp_path / "test_semaphore.db")
    cache = ResultCache(db_path=db_path)

    pipeline = Pipeline(
        search=mock_search, label_extractor=mock_label, verifier=mock_verifier,
        ocr_checker=mock_ocr, quality_filter=mock_quality, scorer=ConfidenceScorer(), cache=cache,
    )

    await pipeline.process_batch(skus, bypass_cache=True, max_concurrency=3)

    # Max observed concurrency should not exceed 3
    assert max_observed_concurrency <= 3, f"Observed concurrency {max_observed_concurrency} exceeded limit 3"

    cache.close()


# ---------------------------------------------------------------------------
# Task 9.3: Error handling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_handles_image_download_failure(sample_sku, fake_image, tmp_path):
    """When image download fails, pipeline continues with next candidate or returns No Image."""
    # Two candidates: first will fail download, second will succeed
    candidates = [
        CandidateImage(url="https://fail-url.com/wine.jpg", source="test", raw_image=None),
        CandidateImage(url="https://success-url.com/wine.jpg", source="test", raw_image=None),
    ]

    mock_search = MagicMock(spec=SearchModule)
    mock_search.search = AsyncMock(return_value=candidates)

    mock_label = MagicMock(spec=LabelExtractor)
    mock_label.extract_label = MagicMock(return_value=LabelExtractionResult(
        cropped_image=fake_image, label_detected=True
    ))

    mock_verifier = MagicMock(spec=FingerprintVerifier)
    mock_verifier.verify = AsyncMock(return_value=VerificationResult(
        fingerprint=Fingerprint(producer="P", appellation="A", cru_vineyard=None, vintage="2020"),
        field_scores=FieldScores(producer_match=0.9, appellation_match=0.9, cru_match=0.9, vintage_match=0.9),
        is_nv=False,
    ))

    mock_ocr = MagicMock(spec=OCRCrossChecker)
    mock_ocr.check = AsyncMock(return_value=OCRResult(
        raw_text="text", producer_confirmed=True, appellation_confirmed=True,
        cru_confirmed=True, vintage_confirmed=True, fields_disagreed=[],
    ))

    mock_quality = MagicMock(spec=QualityFilter)
    mock_quality.evaluate = AsyncMock(return_value=QualityResult(
        passed=True, image_quality_score=1.0, rejection_reasons=[],
    ))

    db_path = str(tmp_path / "test_download_fail.db")
    cache = ResultCache(db_path=db_path)

    pipeline = Pipeline(
        search=mock_search, label_extractor=mock_label, verifier=mock_verifier,
        ocr_checker=mock_ocr, quality_filter=mock_quality, scorer=ConfidenceScorer(), cache=cache,
    )

    # Mock _download_image to fail for first URL, succeed for second
    async def mock_download(url):
        if "fail-url" in url:
            return None  # Simulate download failure
        return fake_image

    with patch.object(pipeline, '_download_image', side_effect=mock_download):
        result = await pipeline.process_sku(sample_sku, bypass_cache=True)

    # Should have fallen back to the second candidate
    assert result.sku_id == sample_sku.id
    assert result.image_url == "https://success-url.com/wine.jpg"

    cache.close()


@pytest.mark.asyncio
async def test_pipeline_handles_ocr_failure(sample_sku, fake_image, tmp_path):
    """When OCR fails, pipeline continues and still returns a result."""
    mock_search = MagicMock(spec=SearchModule)
    mock_search.search = AsyncMock(return_value=[
        CandidateImage(url="https://example.com/wine.jpg", source="test", raw_image=fake_image)
    ])

    mock_label = MagicMock(spec=LabelExtractor)
    mock_label.extract_label = MagicMock(return_value=LabelExtractionResult(
        cropped_image=fake_image, label_detected=True
    ))

    mock_verifier = MagicMock(spec=FingerprintVerifier)
    mock_verifier.verify = AsyncMock(return_value=VerificationResult(
        fingerprint=Fingerprint(producer="P", appellation="A", cru_vineyard=None, vintage="2020"),
        field_scores=FieldScores(producer_match=0.9, appellation_match=0.9, cru_match=0.9, vintage_match=0.9),
        is_nv=False,
    ))

    # OCR raises an exception
    mock_ocr = MagicMock(spec=OCRCrossChecker)
    mock_ocr.check = AsyncMock(side_effect=Exception("OCR API error"))

    mock_quality = MagicMock(spec=QualityFilter)
    mock_quality.evaluate = AsyncMock(return_value=QualityResult(
        passed=True, image_quality_score=1.0, rejection_reasons=[],
    ))

    db_path = str(tmp_path / "test_ocr_fail.db")
    cache = ResultCache(db_path=db_path)

    pipeline = Pipeline(
        search=mock_search, label_extractor=mock_label, verifier=mock_verifier,
        ocr_checker=mock_ocr, quality_filter=mock_quality, scorer=ConfidenceScorer(), cache=cache,
    )

    # Pipeline should catch the error and return "No Image" since all candidates failed
    result = await pipeline.process_sku(sample_sku, bypass_cache=True)

    # Should return a result (not raise)
    assert isinstance(result, ScoredResult)
    assert result.sku_id == sample_sku.id
    # Since the only candidate failed, should be "No Image"
    assert result.image_url is None
    assert "No Image" in result.rejection_reasons

    cache.close()


@pytest.mark.asyncio
async def test_pipeline_handles_quality_filter_failure(sample_sku, fake_image, tmp_path):
    """When quality filter fails, pipeline continues and returns a result."""
    mock_search = MagicMock(spec=SearchModule)
    mock_search.search = AsyncMock(return_value=[
        CandidateImage(url="https://example.com/wine.jpg", source="test", raw_image=fake_image)
    ])

    mock_label = MagicMock(spec=LabelExtractor)
    mock_label.extract_label = MagicMock(return_value=LabelExtractionResult(
        cropped_image=fake_image, label_detected=True
    ))

    mock_verifier = MagicMock(spec=FingerprintVerifier)
    mock_verifier.verify = AsyncMock(return_value=VerificationResult(
        fingerprint=Fingerprint(producer="P", appellation="A", cru_vineyard=None, vintage="2020"),
        field_scores=FieldScores(producer_match=0.9, appellation_match=0.9, cru_match=0.9, vintage_match=0.9),
        is_nv=False,
    ))

    mock_ocr = MagicMock(spec=OCRCrossChecker)
    mock_ocr.check = AsyncMock(return_value=OCRResult(
        raw_text="text", producer_confirmed=True, appellation_confirmed=True,
        cru_confirmed=True, vintage_confirmed=True, fields_disagreed=[],
    ))

    # Quality filter raises an exception
    mock_quality = MagicMock(spec=QualityFilter)
    mock_quality.evaluate = AsyncMock(side_effect=Exception("Quality API error"))

    db_path = str(tmp_path / "test_quality_fail.db")
    cache = ResultCache(db_path=db_path)

    pipeline = Pipeline(
        search=mock_search, label_extractor=mock_label, verifier=mock_verifier,
        ocr_checker=mock_ocr, quality_filter=mock_quality, scorer=ConfidenceScorer(), cache=cache,
    )

    # Pipeline should catch the error and return "No Image"
    result = await pipeline.process_sku(sample_sku, bypass_cache=True)

    # Should return a result (not raise)
    assert isinstance(result, ScoredResult)
    assert result.sku_id == sample_sku.id
    # Since the only candidate failed, should be "No Image"
    assert result.image_url is None

    cache.close()


@pytest.mark.asyncio
async def test_batch_processing_returns_no_image_when_search_fails(fake_image, tmp_path):
    """When search returns empty list, pipeline returns 'No Image' result."""
    sku = SKU(id="SKU-BAD", producer="Bad", appellation="A", cru_vineyard=None, vintage=2020, format="750ml", region="R")

    mock_search = MagicMock(spec=SearchModule)
    # Search returns empty list (simulating no results found)
    mock_search.search = AsyncMock(return_value=[])

    db_path = str(tmp_path / "test_search_empty.db")
    cache = ResultCache(db_path=db_path)

    pipeline = Pipeline(
        search=mock_search,
        label_extractor=MagicMock(spec=LabelExtractor),
        verifier=MagicMock(spec=FingerprintVerifier),
        ocr_checker=MagicMock(spec=OCRCrossChecker),
        quality_filter=MagicMock(spec=QualityFilter),
        scorer=ConfidenceScorer(),
        cache=cache,
    )

    results = await pipeline.process_batch([sku], bypass_cache=True, max_concurrency=1)

    # Should have a result
    assert len(results) == 1
    assert results[0].sku_id == "SKU-BAD"
    assert results[0].image_url is None
    assert "No Image" in results[0].rejection_reasons

    cache.close()


@pytest.mark.asyncio
async def test_pipeline_returns_cached_result_when_available(sample_sku, tmp_path):
    """When cache has result and bypass_cache=False, pipeline returns cached result."""
    cached_result = ScoredResult(
        sku_id=sample_sku.id,
        image_url="https://cached-url.com/wine.jpg",
        producer_match=0.95,
        appellation_match=0.90,
        cru_match=0.85,
        vintage_match=1.0,
        image_quality=0.9,
        overall_confidence=0.92,
        verdict=Verdict.PASS,
        fingerprint=Fingerprint(
            producer="Domaine Leflaive",
            appellation="Puligny-Montrachet",
            cru_vineyard="Les Pucelles",
            vintage="2020",
        ),
        rejection_reasons=[],
    )

    db_path = str(tmp_path / "test_cache_hit.db")
    cache = ResultCache(db_path=db_path)
    cache.put(sample_sku.id, cached_result)

    # These mocks should NOT be called
    mock_search = MagicMock(spec=SearchModule)
    mock_search.search = AsyncMock()
    mock_label = MagicMock(spec=LabelExtractor)
    mock_verifier = MagicMock(spec=FingerprintVerifier)
    mock_verifier.verify = AsyncMock()
    mock_ocr = MagicMock(spec=OCRCrossChecker)
    mock_ocr.check = AsyncMock()
    mock_quality = MagicMock(spec=QualityFilter)
    mock_quality.evaluate = AsyncMock()

    pipeline = Pipeline(
        search=mock_search,
        label_extractor=mock_label,
        verifier=mock_verifier,
        ocr_checker=mock_ocr,
        quality_filter=mock_quality,
        scorer=ConfidenceScorer(),
        cache=cache,
    )

    result = await pipeline.process_sku(sample_sku, bypass_cache=False)

    # Verify no pipeline stages were invoked
    mock_search.search.assert_not_awaited()
    mock_label.extract_label.assert_not_called()
    mock_verifier.verify.assert_not_awaited()
    mock_ocr.check.assert_not_awaited()
    mock_quality.evaluate.assert_not_awaited()

    # Result should match cached result
    assert result.sku_id == cached_result.sku_id
    assert result.image_url == cached_result.image_url
    assert result.overall_confidence == cached_result.overall_confidence
    assert result.verdict == cached_result.verdict

    cache.close()
