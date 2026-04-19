# Implementation Plan: Wine Photo Pipeline

## Overview

Incremental implementation of the wine photo pipeline, starting with core data models and scoring logic, then building outward through verification, search, quality filtering, caching, and finally the Streamlit demo UI. Each stage is validated with tests before moving to the next.

## Tasks

- [x] 1. Set up project structure and core data models
  - [x] 1.1 Create project directory structure and install dependencies
    - Create `src/wine_pipeline/` package with `__init__.py`
    - Create `pyproject.toml` or `requirements.txt` with: httpx, openai, google-cloud-vision, opencv-python-headless, playwright, rapidfuzz, streamlit, hypothesis, pytest
    - Create `tests/` directory with `conftest.py` (hypothesis settings: min 100 examples)
    - _Requirements: 1.5_
  - [x] 1.2 Implement core data models
    - Create `src/wine_pipeline/models.py` with: SKU, CandidateImage, LabelExtractionResult, Fingerprint, FieldScores, OCRResult, QualityResult, ScoredResult, Verdict enum
    - Implement `ScoredResult.to_json()` and `ScoredResult.from_json()` methods
    - _Requirements: 6.6_
  - [x] 1.3 Write property test for ScoredResult JSON round-trip
    - **Property 13: ScoredResult JSON round-trip**
    - Generate random ScoredResult instances with hypothesis, verify `from_json(to_json(x)) == x`
    - **Validates: Requirements 6.6**

- [x] 2. Implement Confidence Scorer
  - [x] 2.1 Implement field comparison function
    - Create `src/wine_pipeline/scoring.py`
    - Implement `compare_field(extracted: str, expected: str) -> float` using rapidfuzz normalized Levenshtein distance
    - Return 0.0 for None/empty inputs, 1.0 for identical strings, fuzzy score otherwise
    - _Requirements: 3.2_
  - [x] 2.2 Write property test for field comparison
    - **Property 5: Field comparison scores are bounded and identity-correct**
    - Generate random string pairs, verify score in [0.0, 1.0]; generate random strings, verify self-comparison == 1.0
    - **Validates: Requirements 3.2**
  - [x] 2.3 Implement Confidence Scorer with weighted scoring and verdict assignment
    - Implement `ConfidenceScorer.score()` that computes per-dimension scores, weighted overall confidence, and verdict
    - Implement NV wine weight redistribution (skip vintage, redistribute proportionally)
    - Implement OCR agreement/disagreement score adjustment
    - Implement verdict thresholds: ≥0.85 PASS, [0.60, 0.85) QUARANTINE, <0.60 REJECT
    - _Requirements: 3.3, 4.3, 4.4, 6.1, 6.2, 6.3, 6.4, 6.5_
  - [x] 2.4 Write property tests for Confidence Scorer
    - **Property 6: NV wine scoring redistributes vintage weight** — generate random NV SKU scores, verify remaining weights sum to 1.0
    - **Property 7: OCR agreement boosts and disagreement reduces field scores** — generate random field scores with controlled OCR results, verify directional adjustment
    - **Property 10: All dimension scores are bounded in [0.0, 1.0]** — generate random inputs, verify all output scores in range
    - **Property 11: Overall confidence equals weighted sum** — generate random dimension scores, verify overall == weighted sum within 1e-9
    - **Property 12: Verdict thresholds correctly applied** — generate random overall_confidence floats in [0.0, 1.0], verify correct verdict
    - **Validates: Requirements 3.3, 4.3, 4.4, 6.1, 6.2, 6.3, 6.4, 6.5**
  - [x] 2.5 Write unit tests for Confidence Scorer edge cases
    - Test verdict boundary values: 0.0, 0.59, 0.60, 0.849, 0.85, 1.0
    - Test NV wine with all fields matching perfectly
    - Test all fields disagreed by OCR
    - _Requirements: 6.3, 6.4, 6.5_

- [x] 3. Checkpoint — Ensure scoring tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement Fingerprint Verifier
  - [x] 4.1 Implement GPT-4o fingerprint extraction
    - Create `src/wine_pipeline/verifier.py`
    - Implement `FingerprintVerifier._extract_fingerprint()` that calls OpenAI GPT-4o vision API with a structured prompt to extract producer, appellation, cru_vineyard, vintage from a label image
    - Parse GPT-4o JSON response into a Fingerprint object, defaulting missing fields to None
    - _Requirements: 3.1_
  - [x] 4.2 Implement field-by-field comparison against SKU metadata
    - Implement `FingerprintVerifier._compare_fields()` using the `compare_field` function from scoring.py
    - Handle NV wines by setting vintage_match to 0.0 and flagging for weight redistribution
    - Implement `FingerprintVerifier.verify()` that orchestrates extraction and comparison
    - _Requirements: 3.2, 3.3, 3.4_
  - [x] 4.3 Write property test for fingerprint structure
    - **Property 4: Fingerprint extraction always produces all four fields**
    - Mock GPT-4o responses with random JSON structures, verify parsed Fingerprint always has all four fields (string or None)
    - **Validates: Requirements 3.1**

- [x] 5. Implement OCR Cross-Checker
  - [x] 5.1 Implement Google Vision OCR integration
    - Create `src/wine_pipeline/ocr.py`
    - Implement `OCRCrossChecker.check()` that sends image to Google Cloud Vision (asia-east2), extracts raw text, and parses tokens
    - Implement token matching logic: search raw OCR text for producer, appellation, cru, vintage substrings using fuzzy matching
    - Return OCRResult with per-field confirmation/contradiction flags
    - _Requirements: 4.1, 4.2_
  - [x] 5.2 Write property test for OCR text parsing
    - **Property 8: OCR text parsing extracts known embedded tokens**
    - Generate random text strings with embedded known wine terms, verify the parser identifies them
    - **Validates: Requirements 4.2**

- [x] 6. Implement Label Extractor
  - [x] 6.1 Implement OpenCV label cropping
    - Create `src/wine_pipeline/label_extractor.py`
    - Implement `LabelExtractor.extract_label()`: grayscale conversion, Gaussian blur, adaptive thresholding, contour detection, filter by area/aspect ratio, crop largest qualifying contour
    - Implement fallback: if no qualifying contour, return full image with `label_detected=False`
    - _Requirements: 2.1, 2.2, 2.3_
  - [x] 6.2 Write unit tests for Label Extractor
    - Test with a synthetic image containing a rectangle (should detect and crop)
    - Test with a uniform color image (no contour — should return full image with label_detected=False)
    - _Requirements: 2.1, 2.2_

- [x] 7. Implement Quality Filter
  - [x] 7.1 Implement GPT-4o-based quality evaluation
    - Create `src/wine_pipeline/quality_filter.py`
    - Implement `QualityFilter.evaluate()` that sends image to GPT-4o with a structured prompt checking: watermarks, blur/glare, single upright bottle, white/grey background, no lifestyle props, not AI-generated
    - Parse response into QualityResult with pass/fail and specific rejection reasons
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  - [x] 7.2 Write property test for quality filter rejection reasons
    - **Property 9: Rejected images always have non-empty rejection reasons**
    - Generate random QualityResult objects where passed=False, verify rejection_reasons is non-empty
    - **Validates: Requirements 5.7**

- [x] 8. Checkpoint — Ensure all component tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Search Module
  - [x] 9.1 Implement SerpAPI search
    - Create `src/wine_pipeline/search.py`
    - Implement `SearchModule._search_serpapi()` using httpx async client to query SerpAPI with constructed wine search query
    - Parse response into list of CandidateImage objects
    - _Requirements: 1.1_
  - [x] 9.2 Implement Playwright retailer scraping
    - Implement `SearchModule._search_retailers()` using Playwright async API to scrape Vivino and Wine-Searcher
    - Extract product image URLs from page content
    - _Requirements: 1.2_
  - [x] 9.3 Implement producer site search and fallback chain orchestration
    - Implement `SearchModule._search_producer_site()` for producer website image search
    - Implement `SearchModule.search()` that executes the full fallback chain: SerpAPI → retailers → producer site → "No Image"
    - _Requirements: 1.3, 1.4_
  - [x] 9.4 Write unit tests for Search Module fallback chain
    - Mock all sources, verify SerpAPI is called first
    - Mock SerpAPI returning empty, verify retailers are called
    - Mock all sources returning empty, verify "No Image" result
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 10. Implement Cache Layer
  - [x] 10.1 Implement SQLite cache
    - Create `src/wine_pipeline/cache.py`
    - Implement `ResultCache` with `get()`, `put()`, `invalidate()` methods
    - Create table schema on initialization if not exists
    - Serialize/deserialize Fingerprint and rejection_reasons as JSON strings
    - _Requirements: 7.1, 7.2, 7.3_
  - [x] 10.2 Write property test for cache round-trip
    - **Property 14: Cache store/retrieve round-trip**
    - Generate random ScoredResult objects, store in SQLite, retrieve by SKU ID, verify equivalence
    - **Validates: Requirements 7.1**

- [x] 11. Implement Pipeline Orchestrator
  - [x] 11.1 Implement Pipeline.process_sku()
    - Create `src/wine_pipeline/pipeline.py`
    - Wire all components: cache check → search → label extraction → fingerprint verification → OCR cross-check → quality filter → confidence scoring → cache store
    - Handle cache bypass flag
    - Handle "No Image" at each stage (no candidates, all rejected, low confidence)
    - _Requirements: 1.1–1.5, 2.1–2.3, 3.1–3.4, 4.1–4.4, 5.1–5.7, 6.1–6.6, 7.1–7.3_
  - [x] 11.2 Implement Pipeline.process_batch()
    - Implement concurrent batch processing using asyncio.gather with concurrency limits
    - _Requirements: 1.5_
  - [x] 11.3 Write property test for cached SKU bypass
    - **Property 15: Cached SKU returns without re-executing pipeline stages**
    - Mock pipeline stages, store a cached result, process SKU without bypass, verify no stage methods called
    - **Validates: Requirements 7.2**
  - [x] 11.4 Write integration test for full pipeline flow
    - Mock all external APIs (SerpAPI, GPT-4o, Google Vision)
    - Process a test SKU through the full pipeline
    - Verify all stages execute in correct order and produce a valid ScoredResult
    - _Requirements: 1.1–6.6_

- [x] 12. Checkpoint — Ensure all pipeline tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement Streamlit Demo UI
  - [x] 13.1 Build Streamlit app with SKU list and detail views
    - Create `src/wine_pipeline/app.py`
    - Implement SKU list table showing verdict, overall_confidence, and image URL per SKU
    - Implement detail view: raw image, cropped label, fingerprint fields, per-dimension scores
    - Add buttons to trigger pipeline on test SKUs and reference SKUs
    - Implement summary accuracy metrics (PASS/QUARANTINE/REJECT counts and percentages)
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 13.2 Write property test for summary metrics computation
    - **Property 16: Summary metrics correctly count verdicts**
    - Generate random lists of ScoredResults, verify PASS + QUARANTINE + REJECT counts == total, percentages == count/total
    - **Validates: Requirements 8.4**

- [x] 14. Define test SKU data and run full benchmark
  - [x] 14.1 Create test SKU and reference SKU data files
    - Create `data/test_skus.json` with the 10 test SKUs from the assignment
    - Create `data/reference_skus.json` with the 10 reference SKUs
    - _Requirements: 9.1_
  - [x] 14.2 Implement benchmark runner script
    - Create `scripts/benchmark.py` that loads test SKUs, runs pipeline, outputs per-SKU results and overall accuracy
    - _Requirements: 9.1, 9.2_

- [x] 15. Final checkpoint — Ensure all tests pass and pipeline runs end-to-end
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required — no optional tasks
- Each task references specific requirements for traceability
- Property tests use `hypothesis` library with minimum 100 iterations
- External API calls (GPT-4o, SerpAPI, Google Vision) are mocked in tests
- Checkpoints at tasks 3, 8, 12, and 15 ensure incremental validation
