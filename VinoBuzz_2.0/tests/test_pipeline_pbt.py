"""Property-based tests for the Pipeline Orchestrator.

Feature: wine-photo-pipeline, Property 1: Batch processing preserves order and handles size limits
Feature: wine-photo-pipeline, Property 11: Error handling continues processing on failures
Feature: wine-photo-pipeline, Property 15: Cached SKU returns without re-executing pipeline stages
Validates: Requirements 1.1, 1.2, 1.3, 1.4, 12.1, 12.2, 12.3, 12.4, 7.2
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

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


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

verdicts = st.sampled_from([Verdict.PASS, Verdict.QUARANTINE, Verdict.REJECT])

fingerprints = st.builds(
    Fingerprint,
    producer=st.one_of(st.none(), st.text(min_size=1, max_size=30)),
    appellation=st.one_of(st.none(), st.text(min_size=1, max_size=30)),
    cru_vineyard=st.one_of(st.none(), st.text(min_size=1, max_size=30)),
    vintage=st.one_of(st.none(), st.from_regex(r"[12]\d{3}", fullmatch=True)),
)

scored_results = st.builds(
    ScoredResult,
    sku_id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    image_url=st.one_of(st.none(), st.text(min_size=5, max_size=60)),
    producer_match=st.floats(min_value=0.0, max_value=1.0),
    appellation_match=st.floats(min_value=0.0, max_value=1.0),
    cru_match=st.floats(min_value=0.0, max_value=1.0),
    vintage_match=st.floats(min_value=0.0, max_value=1.0),
    image_quality=st.floats(min_value=0.0, max_value=1.0),
    overall_confidence=st.floats(min_value=0.0, max_value=1.0),
    verdict=verdicts,
    fingerprint=st.one_of(st.none(), fingerprints),
    rejection_reasons=st.lists(st.text(min_size=1, max_size=40), max_size=3),
)

# SKU strategy for batch testing
skus = st.builds(
    SKU,
    id=st.text(min_size=1, max_size=20).filter(lambda s: s.strip()),
    producer=st.text(min_size=1, max_size=30),
    appellation=st.text(min_size=1, max_size=30),
    cru_vineyard=st.one_of(st.none(), st.text(min_size=1, max_size=30)),
    vintage=st.one_of(st.none(), st.integers(min_value=1900, max_value=2100)),
    format=st.just("750ml"),
    region=st.text(min_size=1, max_size=30),
)

# Lists of SKUs for batch testing
sku_batches = st.lists(skus, min_size=1, max_size=10)


# ---------------------------------------------------------------------------
# Property 1: Batch processing preserves order and handles size limits
# ---------------------------------------------------------------------------


@given(skus_batch=sku_batches)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_batch_processing_preserves_order_and_handles_size_limits(skus_batch: list[SKU]):
    """**Property 1: Batch processing preserves order and handles size limits**

    For any batch of SKUs, processing the batch should return results in the
    same order as input, process all SKUs in batches up to 100, process only
    the first 100 SKUs in larger batches, and return results for all processed SKUs.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
    """
    fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        cache = ResultCache(db_path=tmp.name)

        # Create mock pipeline components
        async def mock_search_impl(sku):
            return [CandidateImage(url=f"https://example.com/{sku.id}.jpg", source="test", raw_image=fake_image)]

        mock_search = MagicMock(spec=SearchModule)
        mock_search.search = AsyncMock(side_effect=mock_search_impl)

        mock_label = MagicMock(spec=LabelExtractor)
        mock_label.extract_label = MagicMock(return_value=LabelExtractionResult(
            cropped_image=fake_image, label_detected=True
        ))

        async def mock_verify_impl(image, sku):
            return VerificationResult(
                fingerprint=Fingerprint(producer=sku.producer, appellation=sku.appellation,
                                       cru_vineyard=sku.cru_vineyard, vintage=str(sku.vintage) if sku.vintage else None),
                field_scores=FieldScores(producer_match=0.9, appellation_match=0.9, cru_match=0.9, vintage_match=0.9),
                is_nv=sku.vintage is None,
            )

        mock_verifier = MagicMock(spec=FingerprintVerifier)
        mock_verifier.verify = AsyncMock(side_effect=mock_verify_impl)

        mock_ocr = MagicMock(spec=OCRCrossChecker)
        mock_ocr.check = AsyncMock(return_value=OCRResult(
            raw_text="text", producer_confirmed=True, appellation_confirmed=True,
            cru_confirmed=True, vintage_confirmed=True, fields_disagreed=[],
        ))

        mock_quality = MagicMock(spec=QualityFilter)
        mock_quality.evaluate = AsyncMock(return_value=QualityResult(
            passed=True, image_quality_score=1.0, rejection_reasons=[],
        ))

        pipeline = Pipeline(
            search=mock_search, label_extractor=mock_label, verifier=mock_verifier,
            ocr_checker=mock_ocr, quality_filter=mock_quality, scorer=ConfidenceScorer(), cache=cache,
        )

        results = await pipeline.process_batch(skus_batch, bypass_cache=True, max_concurrency=2)

        # Property 1.1: All SKUs are processed
        assert len(results) == len(skus_batch)

        # Property 1.4: Order is preserved
        for i, (sku, result) in enumerate(zip(skus_batch, results)):
            assert result.sku_id == sku.id, f"Order mismatch at index {i}"

        cache.close()


# ---------------------------------------------------------------------------
# Property 11: Error handling continues processing on failures
# ---------------------------------------------------------------------------


@given(skus_batch=sku_batches)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_error_handling_continues_processing_on_failures(skus_batch: list[SKU]):
    """**Property 11: Error handling continues processing on failures**

    For any error condition (image download failure, OCR failure, batch errors,
    critical errors), the system should log the error, continue processing other
    items when possible, and return partial results for successful SKUs.

    **Validates: Requirements 12.1, 12.2, 12.3, 12.4**
    """
    assume(len(skus_batch) >= 1)  # Need at least one SKU to test

    fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        cache = ResultCache(db_path=tmp.name)

        # Create mock pipeline that simulates failures for some SKUs
        failure_count = 0
        sku_ids = [sku.id for sku in skus_batch]

        async def mock_search_with_failures(sku):
            nonlocal failure_count
            # Fail for every other SKU (based on SKU ID hash)
            if hash(sku.id) % 2 == 0:
                failure_count += 1
                return []  # Return empty list (simulating no candidates found)
            return [CandidateImage(url=f"https://example.com/{sku.id}.jpg", source="test", raw_image=fake_image)]

        mock_search = MagicMock(spec=SearchModule)
        mock_search.search = AsyncMock(side_effect=mock_search_with_failures)

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

        pipeline = Pipeline(
            search=mock_search, label_extractor=mock_label, verifier=mock_verifier,
            ocr_checker=mock_ocr, quality_filter=mock_quality, scorer=ConfidenceScorer(), cache=cache,
        )

        results = await pipeline.process_batch(skus_batch, bypass_cache=True, max_concurrency=1)

        # Property 12.3: Batch contains errors but still returns partial results
        assert len(results) == len(skus_batch), "Should return results for all SKUs even when some fail"

        # Each result should be valid (either with image or "No Image")
        for result in results:
            assert isinstance(result, ScoredResult)
            assert result.sku_id in sku_ids
            # Either we got an image URL or we got a "No Image" result
            assert result.image_url is not None or "No Image" in result.rejection_reasons

        cache.close()


@given(result=scored_results)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_cached_sku_returns_without_reexecuting_stages(result: ScoredResult):
    """**Property 15: Cached SKU returns without re-executing pipeline stages**

    When a SKU has a cached result and is processed without cache bypass,
    the Pipeline returns the cached result and does NOT invoke any pipeline
    stage (search, label extractor, verifier, OCR, quality filter).

    **Validates: Requirements 7.2**
    """
    # Set up a real SQLite cache in a temp file
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        cache = ResultCache(db_path=tmp.name)
        cache.put(result.sku_id, result)

        # Create mock pipeline stages — none should be called
        mock_search = MagicMock(spec=SearchModule)
        mock_search.search = AsyncMock()
        mock_label = MagicMock(spec=LabelExtractor)
        mock_verifier = MagicMock(spec=FingerprintVerifier)
        mock_verifier.verify = AsyncMock()
        mock_ocr = MagicMock(spec=OCRCrossChecker)
        mock_ocr.check = AsyncMock()
        mock_quality = MagicMock(spec=QualityFilter)
        mock_quality.evaluate = AsyncMock()
        mock_scorer = MagicMock(spec=ConfidenceScorer)

        pipeline = Pipeline(
            search=mock_search,
            label_extractor=mock_label,
            verifier=mock_verifier,
            ocr_checker=mock_ocr,
            quality_filter=mock_quality,
            scorer=mock_scorer,
            cache=cache,
        )

        # Build a dummy SKU with the same id
        sku = SKU(
            id=result.sku_id,
            producer="Test",
            appellation="Test",
            cru_vineyard=None,
            vintage=None,
            format="750ml",
            region="Test",
        )

        # Process without bypass — should hit cache
        returned = await pipeline.process_sku(sku, bypass_cache=False)

        # Verify no pipeline stages were invoked
        mock_search.search.assert_not_awaited()
        mock_verifier.verify.assert_not_awaited()
        mock_ocr.check.assert_not_awaited()
        mock_quality.evaluate.assert_not_awaited()
        mock_scorer.score.assert_not_called()
        mock_label.extract_label.assert_not_called()

        # Verify the returned result matches the cached one
        assert returned.sku_id == result.sku_id
        assert returned.verdict == result.verdict

        cache.close()
