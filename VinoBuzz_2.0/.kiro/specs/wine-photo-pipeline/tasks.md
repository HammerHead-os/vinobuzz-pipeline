# Implementation Plan: Wine Photo Pipeline

## Overview

This implementation plan breaks down the wine photo pipeline feature into discrete coding steps. The pipeline processes wine SKUs by searching for images, extracting labels, reading text via OCR, and cross-checking against SKU metadata. All components are already implemented in the existing codebase, so this plan focuses on testing and validation.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Verify existing codebase structure in src/wine_pipeline/
  - Verify test structure in tests/
  - Install required dependencies: ddgs, opencv-python, google-cloud-vision, google-genai, rapidfuzz, httpx, playwright, streamlit, python-dotenv
  - Set up environment variables for Google Cloud credentials
  - _Requirements: 11.3_

- [x] 2. Verify search module implementation
  - [x] 2.1 Verify DuckDuckGo search integration
    - Test query construction from SKU metadata
    - Test candidate image URL extraction
    - _Requirements: 2.1, 2.2_
  
  - [x] 2.2 Verify retailer scraping fallback
    - Test Vivino scraping with Playwright
    - Test Wine-Searcher scraping with Playwright
    - _Requirements: 2.1_
  
  - [x] 2.3 Verify producer site search fallback
    - Test site-restricted DuckDuckGo queries
    - _Requirements: 2.1_
  
  - [x] 2.4 Write property test for search fallback chain
    - **Property 2: Search fallback chain returns candidates or empty list**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
  
  - [x] 2.5 Write unit tests for search module
    - Test query construction edge cases
    - Test image URL filtering heuristics
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. Verify label extraction implementation
  - [x] 3.1 Verify OpenCV contour detection
    - Test grayscale conversion
    - Test Gaussian blur
    - Test adaptive thresholding
    - _Requirements: 3.1_
  
  - [x] 3.2 Verify label cropping logic
    - Test contour filtering by area ratio
    - Test contour filtering by aspect ratio
    - _Requirements: 3.2_
  
  - [x] 3.3 Verify fallback to full image
    - Test behavior when no label detected
    - _Requirements: 3.3_
  
  - [x] 3.4 Write property test for label extraction
    - **Property 3: Label extraction returns cropped label or full image fallback**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
  
  - [x] 3.5 Write unit tests for label extractor
    - Test edge cases (no contours, all filtered)
    - Test image decoding failures
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 4. Verify fingerprint verifier implementation
  - [x] 4.1 Verify Gemini extraction
    - Test Gemini API call with label image
    - Test JSON parsing from Gemini response
    - _Requirements: 4.1, 4.2_
  
  - [x] 4.2 Verify field comparison logic
    - Test direct field matching
    - Test cross-field matching fallback
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [x] 4.3 Verify NV wine handling
    - Test vintage score assignment for NV wines
    - _Requirements: 6.3_
  
  - [x] 4.4 Write property test for field comparison
    - **Property 5: Field comparison uses fuzzy matching with cross-field fallback**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
  
  - [x] 4.5 Write unit tests for fingerprint verifier
    - Test Gemini JSON parsing edge cases
    - Test cross-field matching scenarios
    - _Requirements: 4.1, 4.2, 6.1, 6.2, 6.3, 6.4_

- [x] 5. Verify OCR cross-check implementation
  - [x] 5.1 Verify Cloud Vision OCR
    - Test Cloud Vision API call
    - Test text extraction from response
    - _Requirements: 4.3_
  
  - [x] 5.2 Verify cross-reference comparison
    - Test field matching with fuzzy comparison
    - Test confirmed field marking
    - Test disagreed field marking
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 5.3 Write property test for OCR cross-check
    - **Property 4: OCR extraction returns structured fields with cross-check validation**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
  
  - [x] 5.4 Write unit tests for OCR cross-checker
    - Test fuzzy matching threshold
    - Test disagreement detection
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 6. Verify quality filter implementation
  - [x] 6.1 Verify Gemini quality assessment
    - Test Gemini API call with label image
    - Test JSON parsing from Gemini response
    - _Requirements: 7.1_
  
  - [x] 6.2 Verify rejection reason generation
    - Test rejection reason mapping
    - Test quality score calculation
    - _Requirements: 7.2, 7.3, 7.4_
  
  - [x] 6.3 Write property test for quality evaluation
    - **Property 6: Quality evaluation checks all criteria and returns rejection reasons**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
  
  - [x] 6.4 Write unit tests for quality filter
    - Test JSON parsing edge cases
    - Test rejection reason generation
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 7. Verify confidence scorer implementation
  - [x] 7.1 Verify weighted scoring
    - Test weighted combination of field scores
    - Test weight redistribution for NV wines
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
  
  - [x] 7.2 Verify OCR adjustment
    - Test boost for confirmed fields
    - Test penalty for disagreements
    - _Requirements: 8.7_
  
  - [x] 7.3 Verify verdict assignment
    - Test PASS threshold (≥70%)
    - Test QUARANTINE threshold (50-70%)
    - Test REJECT threshold (<50%)
    - _Requirements: 10.1, 10.2, 10.3_
  
  - [x] 7.4 Write property test for confidence scoring
    - **Property 7: Confidence scoring combines weighted fields with OCR adjustment**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7**
  
  - [x] 7.5 Write unit tests for confidence scorer
    - Test weight redistribution edge cases
    - Test OCR boost/penalty calculations
    - Test verdict thresholds
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 10.1, 10.2, 10.3_

- [x] 8. Verify result cache implementation
  - [x] 8.1 Verify SQLite storage
    - Test result serialization
    - Test result storage in database
    - _Requirements: 9.1_
  
  - [x] 8.2 Verify cache retrieval
    - Test result retrieval by SKU ID
    - Test cache miss handling
    - _Requirements: 9.2_
  
  - [x] 8.3 Verify cache bypass
    - Test bypass_cache parameter
    - Test reprocessing when bypassed
    - _Requirements: 9.3_
  
  - [x] 8.4 Write property test for caching
    - **Property 8: Caching stores and retrieves results correctly**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4**
  
  - [x] 8.5 Write unit tests for result cache
    - Test serialization/deserialization round trip
    - Test cache invalidation
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 9. Verify pipeline orchestrator implementation
  - [x] 9.1 Verify single SKU processing
    - Test full pipeline flow
    - Test cache check and bypass
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1_
  
  - [x] 9.2 Verify batch processing
    - Test concurrent processing with semaphore
    - Test order preservation
    - _Requirements: 1.1, 1.4_
  
  - [x] 9.3 Verify error handling
    - Test image download failures
    - Test OCR failures
    - Test quality filter failures
    - _Requirements: 12.1, 12.2, 12.3, 12.4_
  
  - [x] 9.4 Write property test for batch processing
    - **Property 1: Batch processing preserves order and handles size limits**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
  
  - [x] 9.5 Write property test for error handling
    - **Property 11: Error handling continues processing on failures**
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4**
  
  - [x] 9.6 Write integration tests for pipeline
    - Test full pipeline with mocked components
    - Test batch processing with errors
    - _Requirements: 1.1, 1.4, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 12.1, 12.2, 12.3, 12.4_

- [x] 10. Verify output generation
  - [x] 10.1 Verify ScoredResult serialization
    - Test to_json() method
    - Test from_json() method
    - _Requirements: 10.4_
  
  - [x] 10.2 Verify no-image result handling
    - Test _no_image_result() function
    - Test placeholder result generation
    - _Requirements: 10.2_
  
  - [x] 10.3 Verify local image storage
    - Test image saving to data/images/
    - Test filename generation
    - _Requirements: 10.1_
  
  - [x] 10.4 Write property test for output generation
    - **Property 9: Output generation includes all required fields**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4**
  
  - [x] 10.5 Write unit tests for output generation
    - Test JSON serialization round trip
    - Test no-image result fields
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 11. Verify CLI implementation
  - [x] 11.1 Verify JSON input loading
    - Test SKU loading from JSON file
    - Test error handling for missing files
    - _Requirements: 11.1_
  
  - [x] 11.2 Verify JSON output generation
    - Test results export to JSON
    - Test file path configuration
    - _Requirements: 11.2_
  
  - [x] 11.3 Verify command-line argument parsing
    - Test argument parsing for configuration
    - Test default values
    - _Requirements: 11.3_
  
  - [x] 11.4 Verify logging to stdout
    - Test progress logging
    - Test error logging
    - _Requirements: 11.4_
  
  - [x] 11.5 Write property test for CLI execution
    - **Property 10: CLI execution processes JSON files and logs progress**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4**
  
  - [x] 11.6 Write unit tests for CLI
    - Test argument parsing edge cases
    - Test file loading errors
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 12. Run all tests and verify coverage
  - [x] 12.1 Run property-based tests
    - Run all property tests with 100+ iterations
    - Verify all properties pass
    - _Requirements: All properties_
  
  - [x] 12.2 Run unit tests
    - Run all unit tests
    - Verify all tests pass
    - _Requirements: All requirements_
  
  - [x] 12.3 Run integration tests
    - Run pipeline integration tests
    - Verify end-to-end flow
    - _Requirements: All requirements_
  
  - [x] 12.4 Check test coverage
    - Verify coverage for all components
    - Identify missing test coverage
    - _Requirements: All requirements_

- [x] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The existing codebase provides a solid foundation; this plan focuses on testing and validation
- All components are already implemented; tasks focus on verifying correctness through testing
