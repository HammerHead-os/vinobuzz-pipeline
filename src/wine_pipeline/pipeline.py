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
from wine_pipeline.image_processor import ProductImageProcessor, is_vivino_product_bottle
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
        self._image_processor = ProductImageProcessor()
        self._strict_quality = True
        self._require_pass_verdict = False
        self._strict_match = False

    def _is_accurate_match(self, scored: ScoredResult, sku: SKU, *, strict_match: bool) -> bool:
        """Return True when quality passed and label metadata matches the SKU."""
        if scored.rejection_reasons:
            return False
        if strict_match:
            min_producer = getattr(self, "_min_producer_match", 0.35)
            min_vintage = getattr(self, "_min_vintage_match", 0.85)
            if scored.producer_match < min_producer:
                return False
            if sku.vintage is not None and scored.vintage_match < min_vintage:
                return False
        return True

    # ------------------------------------------------------------------
    # Single SKU processing
    # ------------------------------------------------------------------

    async def process_sku(
        self, sku: SKU, bypass_cache: bool = False, max_candidates: int = 5
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
        candidates = await self.search.search(sku, limit=max_candidates)
        if not candidates:
            logger.warning("No candidates found for SKU %s", sku.id)
            result = _no_image_result(sku.id)
            self.cache.put(sku.id, result)
            return result

        # Limit candidates for faster runtime
        candidates = candidates[:max_candidates]

        # 3. Process each candidate through the pipeline stages.
        #
        # Only accept candidates that pass all quality checks; pick highest confidence.
        strict_quality = getattr(self, "_strict_quality", True)
        require_pass = getattr(self, "_require_pass_verdict", False)
        strict_match = getattr(self, "_strict_match", False)
        best_passed: Optional[ScoredResult] = None
        best_any: Optional[ScoredResult] = None
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

            if best_any is None or scored.overall_confidence > best_any.overall_confidence:
                best_any = scored

            if not self._is_accurate_match(scored, sku, strict_match=strict_match):
                continue
            if require_pass and scored.verdict != Verdict.PASS:
                continue

            if best_passed is None or scored.overall_confidence > best_passed.overall_confidence:
                best_passed = scored

        # 4. Strict mode: only quality-passing candidates; never fall back to low-quality picks
        if best_result is None:
            if strict_quality:
                best_result = best_passed
            else:
                best_result = best_passed or best_any
        if best_result is None:
            result = _no_image_result(sku.id)
            self.cache.put(sku.id, result)
            return result

        # Ensure the locally saved image corresponds to the selected best result.
        # _process_candidate saves each candidate for UI convenience, so the file
        # could otherwise end up containing the last processed candidate.
        if best_result.image_url:
            try:
                image_bytes = await self._download_image(best_result.image_url)
                if image_bytes:
                    image_bytes = self._normalize_image_bytes(image_bytes)
                if image_bytes:
                    skip_removebg = getattr(self, "_skip_removebg", False)
                    if skip_removebg:
                        self._save_image_locally(sku.id, image_bytes)
                    else:
                        processed = self._image_processor.prepare_for_delivery(image_bytes)
                        if processed:
                            self._save_image_locally(sku.id, processed)
                        elif strict_quality:
                            result = _no_image_result(sku.id)
                            self.cache.put(sku.id, result)
                            return result
            except Exception:
                pass

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
        max_concurrency: int = 3,
        max_candidates: int = 1,
        skip_ocr: bool = True,
        skip_quality: bool = False,
    ) -> list[ScoredResult]:
        """Process multiple SKUs concurrently with a concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _limited(sku: SKU) -> ScoredResult:
            async with semaphore:
                return await self.process_sku(sku, bypass_cache=bypass_cache)

        # Store optimization flags for process_sku to use
        self._max_candidates = max_candidates
        self._skip_ocr = skip_ocr
        self._skip_quality = skip_quality

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

        # Some sources return HTML or invalid bytes with a 200 status.
        # Validate and normalize to JPEG so Gemini accepts it.
        image_bytes = self._normalize_image_bytes(image_bytes)
        if image_bytes is None:
            raise ValueError(f"Downloaded bytes were not a valid image: {candidate.url}")

        if not getattr(self, "_skip_margin_check", False):
            if not is_vivino_product_bottle(candidate.url):
                if not ProductImageProcessor.has_full_body_margins(image_bytes):
                    raise ValueError("Bottle is cropped — cap or base not fully visible")

        # Label extraction
        label_result = self.label_extractor.extract_label(image_bytes)

        # Fingerprint verification
        verification = await self.verifier.verify(label_result.cropped_image, sku)
        
        # Early rejection if not a wine bottle image
        if not verification.fingerprint.is_wine_bottle:
            raise ValueError("Image does not contain a wine bottle with a label")

        # OCR cross-check (optional, can be skipped for speed)
        skip_ocr = getattr(self, "_skip_ocr", False)
        if skip_ocr:
            ocr_result = OCRResult(
                raw_text="",
                producer_confirmed=False,
                appellation_confirmed=False,
                cru_confirmed=False,
                vintage_confirmed=False,
                fields_disagreed=[],
            )
        else:
            try:
                ocr_result = await self.ocr_checker.check(
                    label_result.cropped_image, verification.fingerprint
                )
            except Exception as exc:
                # Treat OCR failure as a candidate failure (tests expect this behavior).
                raise ValueError(f"OCR failed: {exc}") from exc

        # Quality filter (optional, can be skipped for speed)
        skip_quality = getattr(self, '_skip_quality', False)
        if skip_quality:
            quality_result = QualityResult(
                passed=True,
                image_quality_score=1.0,
                rejection_reasons=[],
            )
        else:
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
        # Sanitize SKU ID to avoid invalid path characters
        safe_id = sku_id.replace("/", "_").replace("\\", "_")
        path = os.path.join(img_dir, f"{safe_id}.jpg")
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
                # Quick reject: avoid spending time on non-image payloads.
                ctype = (resp.headers.get("content-type") or "").lower()
                if ctype and not ctype.startswith("image/"):
                    return None
                return resp.content
        except Exception as exc:
            logger.error("Image download failed for %s: %s", url, exc)
            return None

    @staticmethod
    def _normalize_image_bytes(image_bytes: bytes) -> Optional[bytes]:
        """Return JPEG-encoded bytes if decodable; else None."""
        try:
            import numpy as np
            import cv2

            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                return None
            return buf.tobytes()
        except Exception:
            return None
