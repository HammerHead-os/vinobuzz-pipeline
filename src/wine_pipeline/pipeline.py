"""Pipeline Orchestrator for the Wine Photo Pipeline.

Wires all components together: cache check → search → label extraction →
fingerprint verification → OCR cross-check → quality filter → confidence
scoring → cache store.

Requirements: 1.1–1.5, 2.1–2.3, 3.1–3.4, 4.1–4.4, 5.1–5.7, 6.1–6.6, 7.1–7.3
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from wine_pipeline.cache import ResultCache
from wine_pipeline.label_extractor import LabelExtractor
from wine_pipeline.models import (
    CandidateImage,
    FieldScores,
    OCRResult,
    QualityResult,
    ScoredResult,
    SKU,
    Verdict,
)
from wine_pipeline.ocr import OCRCrossChecker
from wine_pipeline.quality_filter import QualityFilter
from wine_pipeline.scoring import ConfidenceScorer
from wine_pipeline.search import SearchModule
from wine_pipeline.verifier import FingerprintVerifier

logger = logging.getLogger(__name__)


def _no_image_result(sku_id: str) -> ScoredResult:
    """Return a 'No Image' ScoredResult for a SKU."""
    return ScoredResult(
        sku_id=sku_id,
        image_url=None,
        producer_match=0.0,
        appellation_match=0.0,
        cru_match=0.0,
        vintage_match=0.0,
        image_quality=0.0,
        overall_confidence=0.0,
        verdict=Verdict.REJECT,
        fingerprint=None,
        rejection_reasons=["No Image"],
    )


class Pipeline:
    """Orchestrates the full wine photo pipeline for a single SKU or batch."""

    def __init__(
        self,
        search: SearchModule,
        label_extractor: LabelExtractor,
        verifier: FingerprintVerifier,
        ocr_checker: OCRCrossChecker,
        quality_filter: QualityFilter,
        scorer: ConfidenceScorer,
        cache: ResultCache,
    ) -> None:
        self.search = search
        self.label_extractor = label_extractor
        self.verifier = verifier
        self.ocr_checker = ocr_checker
        self.quality_filter = quality_filter
        self.scorer = scorer
        self.cache = cache

    # ------------------------------------------------------------------
    # Single SKU processing
    # ------------------------------------------------------------------

    async def process_sku(
        self, sku: SKU, bypass_cache: bool = False
    ) -> ScoredResult:
        """Run the full pipeline for a single SKU.

        Flow:
        1. Check cache (unless bypass requested)
        2. Search for candidate images
        3. For each candidate: download → extract label → verify → OCR → quality filter → score
        4. Pick the best scoring candidate
        5. Store result in cache
        6. Return result
        """
        # 1. Cache check
        if not bypass_cache:
            cached = self.cache.get(sku.id)
            if cached is not None:
                logger.info("Cache hit for SKU %s", sku.id)
                return cached

        # 2. Search for candidates
        candidates = await self.search.search(sku)
        if not candidates:
            logger.warning("No candidates found for SKU %s", sku.id)
            result = _no_image_result(sku.id)
            self.cache.put(sku.id, result)
            return result

        # Limit to top 3 candidates to keep runtime reasonable
        candidates = candidates[:3]

        # 3. Process each candidate through the pipeline stages
        best_result: Optional[ScoredResult] = None

        for candidate in candidates:
            try:
                scored = await self._process_candidate(candidate, sku)
            except Exception as exc:
                logger.error(
                    "Error processing candidate %s for SKU %s: %s",
                    candidate.url, sku.id, exc,
                )
                continue

            if best_result is None or scored.overall_confidence > best_result.overall_confidence:
                best_result = scored

        # 4. If no candidate survived, return "No Image"
        if best_result is None:
            result = _no_image_result(sku.id)
            self.cache.put(sku.id, result)
            return result

        # 5. Store and return
        self.cache.put(sku.id, best_result)
        return best_result

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    async def process_batch(
        self,
        skus: list[SKU],
        bypass_cache: bool = False,
        max_concurrency: int = 1,
    ) -> list[ScoredResult]:
        """Process multiple SKUs concurrently with a concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _limited(sku: SKU) -> ScoredResult:
            async with semaphore:
                return await self.process_sku(sku, bypass_cache=bypass_cache)

        return list(await asyncio.gather(*[_limited(sku) for sku in skus]))

    # ------------------------------------------------------------------
    # Internal: per-candidate pipeline
    # ------------------------------------------------------------------

    async def _process_candidate(
        self, candidate: CandidateImage, sku: SKU
    ) -> ScoredResult:
        """Run a single candidate through label extraction → verification →
        OCR → quality filter → scoring."""

        # Download image if not already present
        image_bytes = candidate.raw_image
        if image_bytes is None:
            image_bytes = await self._download_image(candidate.url)
        if image_bytes is None:
            raise ValueError(f"Failed to download image: {candidate.url}")

        # Label extraction
        label_result = self.label_extractor.extract_label(image_bytes)

        # Fingerprint verification
        verification = await self.verifier.verify(label_result.cropped_image, sku)

        # OCR cross-check
        ocr_result = await self.ocr_checker.check(
            label_result.cropped_image, verification.fingerprint
        )

        # Quality filter
        quality_result = await self.quality_filter.evaluate(image_bytes)

        # Confidence scoring
        scored = self.scorer.score(
            field_scores=verification.field_scores,
            ocr_result=ocr_result,
            quality_result=quality_result,
            label_detected=label_result.label_detected,
            is_nv=verification.is_nv,
        )

        # Fill in SKU-specific fields
        scored.sku_id = sku.id
        scored.image_url = candidate.url
        scored.fingerprint = verification.fingerprint

        # Save image locally for the Streamlit UI
        self._save_image_locally(sku.id, image_bytes)

        return scored

    @staticmethod
    def _save_image_locally(sku_id: str, image_bytes: bytes) -> None:
        """Save a candidate image to disk so the UI can display it."""
        import os
        img_dir = os.path.join("data", "images")
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(img_dir, f"{sku_id}.jpg")
        with open(path, "wb") as f:
            f.write(image_bytes)

    @staticmethod
    async def _download_image(url: str) -> Optional[bytes]:
        """Download an image from a URL. Returns None on failure."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/*,*/*;q=0.8",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except Exception as exc:
            logger.error("Image download failed for %s: %s", url, exc)
            return None
