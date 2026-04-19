"""Property-based tests for the Pipeline Orchestrator.

Feature: wine-photo-pipeline, Property 15: Cached SKU returns without re-executing pipeline stages
Validates: Requirements 7.2
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wine_pipeline.cache import ResultCache
from wine_pipeline.label_extractor import LabelExtractor
from wine_pipeline.models import (
    Fingerprint,
    ScoredResult,
    SKU,
    Verdict,
)
from wine_pipeline.ocr import OCRCrossChecker
from wine_pipeline.pipeline import Pipeline
from wine_pipeline.quality_filter import QualityFilter
from wine_pipeline.scoring import ConfidenceScorer
from wine_pipeline.search import SearchModule
from wine_pipeline.verifier import FingerprintVerifier


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


# ---------------------------------------------------------------------------
# Property 15: Cached SKU returns without re-executing pipeline stages
# ---------------------------------------------------------------------------


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
