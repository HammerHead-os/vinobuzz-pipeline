# Requirements Document

## Introduction

The VinoBuzz Wine Photo Pipeline is an automated system that sources, verifies, and scores wine product photos from the web for VinoBuzz's 4,000+ SKU catalog. The pipeline replaces a ~50% accuracy manual/semi-automated process with a structured multi-stage pipeline targeting 90%+ accuracy. It is designed for Hong Kong deployment with appropriate API endpoint selection and handles the nuances of wine label identification, particularly Burgundy naming complexity.

## Glossary

- **Pipeline**: The end-to-end automated system that searches, extracts, verifies, filters, and scores wine product photos
- **SKU**: A Stock Keeping Unit representing a single wine product with metadata (producer, appellation, cru/vineyard, vintage, format, region)
- **Search_Module**: The component responsible for finding candidate wine images from multiple web sources using a fallback chain
- **Label_Extractor**: The OpenCV-based component that crops the label region from a candidate wine bottle image
- **Fingerprint_Verifier**: The component that uses GPT-4o vision to extract structured label data and compares it field-by-field against SKU metadata
- **OCR_Cross_Checker**: The Google Cloud Vision OCR component that provides independent text extraction for cross-referencing against GPT-4o results
- **Quality_Filter**: The component that evaluates image quality against defined standards (no watermarks, no blur, single bottle, etc.)
- **Confidence_Scorer**: The component that computes per-dimension and overall confidence scores and renders a three-tier verdict
- **Candidate_Image**: A single image URL retrieved during the search stage, before verification
- **Fingerprint**: A structured JSON object containing producer, appellation, cru_vineyard, and vintage fields extracted from a wine label
- **Verdict**: The final decision for a SKU photo: PASS (auto-accept), QUARANTINE (human review), or REJECT (auto-reject)
- **NV_Wine**: A Non-Vintage wine where no vintage year appears on the label
- **Fallback_Chain**: The ordered sequence of image sources tried when earlier sources fail to produce a verified result
- **Demo_UI**: The Streamlit-based interface that displays pipeline results per SKU with visual before/after comparisons
- **Cache**: The SQLite database storing previously processed results to avoid redundant API calls

## Requirements

### Requirement 1: Multi-Source Image Search

**User Story:** As a pipeline operator, I want the system to search multiple image sources in a defined fallback order, so that candidate images are found even for wines with limited web presence.

#### Acceptance Criteria

1. WHEN a SKU is submitted for processing, THE Search_Module SHALL query Google Images via SerpAPI as the first source and return candidate image URLs
2. WHEN SerpAPI returns zero usable candidate images, THE Search_Module SHALL scrape wine retailer sites (Vivino, Wine-Searcher) using Playwright as the second source
3. WHEN retailer scraping returns zero usable candidate images, THE Search_Module SHALL attempt to find images on the producer's own website as the third source
4. WHEN all sources in the Fallback_Chain return zero usable candidate images, THE Search_Module SHALL return a "No Image" result for that SKU
5. THE Search_Module SHALL execute searches asynchronously using httpx to enable concurrent processing of multiple SKUs

### Requirement 2: Label Region Extraction

**User Story:** As a pipeline operator, I want the system to crop the label region from candidate bottle images, so that verification accuracy improves by removing background noise.

#### Acceptance Criteria

1. WHEN a Candidate_Image is retrieved, THE Label_Extractor SHALL use OpenCV contour detection to identify and crop the label region from the bottle image
2. WHEN the Label_Extractor cannot detect a label contour in a Candidate_Image, THE Label_Extractor SHALL pass the full uncropped image to the verification stage with a reduced confidence modifier
3. THE Label_Extractor SHALL produce a cropped image that contains the primary label text (producer, appellation, vintage) without truncation

### Requirement 3: Structured Fingerprint Verification

**User Story:** As a pipeline operator, I want the system to extract structured label data and compare it field-by-field against SKU metadata, so that verification is explicit and auditable rather than a vague similarity check.

#### Acceptance Criteria

1. WHEN a cropped label image is provided, THE Fingerprint_Verifier SHALL use GPT-4o vision to extract a Fingerprint containing producer, appellation, cru_vineyard, and vintage fields
2. THE Fingerprint_Verifier SHALL compare each extracted Fingerprint field independently against the corresponding SKU metadata field and produce a per-field match score between 0.0 and 1.0
3. WHEN the SKU represents an NV_Wine, THE Fingerprint_Verifier SHALL skip the vintage field comparison and redistribute its scoring weight across the remaining fields
4. WHEN GPT-4o extraction and OCR_Cross_Checker extraction disagree on any field value, THE Fingerprint_Verifier SHALL flag that field for review and reduce its match score

### Requirement 4: OCR Cross-Reference

**User Story:** As a pipeline operator, I want an independent OCR check against the GPT-4o extraction, so that verification has a secondary signal to catch vision model errors.

#### Acceptance Criteria

1. WHEN a cropped label image is provided, THE OCR_Cross_Checker SHALL send the image to Google Cloud Vision OCR (asia-east2 endpoint) and return extracted text
2. THE OCR_Cross_Checker SHALL parse the raw OCR text to identify producer, appellation, cru_vineyard, and vintage tokens for cross-referencing against the GPT-4o Fingerprint
3. WHEN the OCR_Cross_Checker result confirms all fields of the GPT-4o Fingerprint, THE Confidence_Scorer SHALL apply a positive confidence boost to the overall score
4. WHEN the OCR_Cross_Checker result contradicts one or more fields of the GPT-4o Fingerprint, THE Confidence_Scorer SHALL reduce the confidence score for the contradicted fields

### Requirement 5: Image Quality Filtering

**User Story:** As a pipeline operator, I want the system to reject images that do not meet VinoBuzz's photo standards, so that only clean, professional product photos are accepted.

#### Acceptance Criteria

1. THE Quality_Filter SHALL reject any Candidate_Image that contains visible watermarks
2. THE Quality_Filter SHALL reject any Candidate_Image that exhibits blur or glare that obscures label text
3. THE Quality_Filter SHALL reject any Candidate_Image that does not show a single upright bottle
4. THE Quality_Filter SHALL reject any Candidate_Image that has a background other than white or neutral grey
5. THE Quality_Filter SHALL reject any Candidate_Image that contains lifestyle props, table settings, or non-product elements
6. THE Quality_Filter SHALL reject any Candidate_Image that appears to be AI-generated rather than a real product photo
7. WHEN a Candidate_Image is rejected by the Quality_Filter, THE Quality_Filter SHALL record the specific rejection reason for auditability

### Requirement 6: Confidence Scoring and Verdict

**User Story:** As a pipeline operator, I want per-dimension confidence scores and a three-tier verdict for each SKU, so that I can trust auto-accepted results and efficiently review borderline cases.

#### Acceptance Criteria

1. THE Confidence_Scorer SHALL compute individual scores for producer_match, appellation_match, cru_match, vintage_match, and image_quality, each between 0.0 and 1.0
2. THE Confidence_Scorer SHALL compute an overall_confidence score as a weighted combination of the individual dimension scores
3. WHEN overall_confidence is 0.85 or above, THE Confidence_Scorer SHALL assign a PASS Verdict (auto-accept)
4. WHEN overall_confidence is between 0.60 and 0.85 (exclusive of 0.85), THE Confidence_Scorer SHALL assign a QUARANTINE Verdict (human review)
5. WHEN overall_confidence is below 0.60, THE Confidence_Scorer SHALL assign a REJECT Verdict (auto-reject) and output "No Image" for that SKU
6. THE Confidence_Scorer SHALL serialize each SKU result as a JSON object containing all per-dimension scores, overall_confidence, and verdict

### Requirement 7: Result Caching

**User Story:** As a pipeline operator, I want processed results cached in SQLite, so that reprocessing the same SKU does not incur redundant API calls.

#### Acceptance Criteria

1. WHEN a SKU has been fully processed, THE Cache SHALL store the SKU identifier, all per-dimension scores, the verdict, the selected image URL, and a timestamp in SQLite
2. WHEN a SKU is submitted for processing and a cached result exists, THE Pipeline SHALL return the cached result without re-executing the search and verification stages
3. WHEN a pipeline operator requests a cache bypass for a SKU, THE Pipeline SHALL re-execute the full pipeline and update the cached result

### Requirement 8: Streamlit Demo UI

**User Story:** As a pipeline operator, I want a visual demo interface that displays per-SKU results with before/after comparisons, so that pipeline accuracy is demonstrable and auditable.

#### Acceptance Criteria

1. THE Demo_UI SHALL display a list of all processed SKUs with their Verdict (PASS, QUARANTINE, REJECT) and overall_confidence score
2. WHEN a user selects a SKU in the Demo_UI, THE Demo_UI SHALL display the raw Candidate_Image, the cropped label image, the extracted Fingerprint fields, and the per-dimension scores
3. THE Demo_UI SHALL allow a user to trigger pipeline processing for the 10 test SKUs and the 10 reference SKUs
4. THE Demo_UI SHALL display a summary accuracy metric showing the count and percentage of PASS, QUARANTINE, and REJECT verdicts across the processed set

### Requirement 9: Pipeline Accuracy Target

**User Story:** As a pipeline operator, I want the pipeline to achieve 90% accuracy on the 10 test SKUs, so that the system meets VinoBuzz's quality threshold for production deployment.

#### Acceptance Criteria

1. WHEN the Pipeline processes the 10 test SKUs, THE Pipeline SHALL correctly identify and verify the right wine photo (or correctly return "No Image") for at least 9 out of 10 SKUs
2. THE Pipeline SHALL treat a correct "No Image" result for a wine with near-zero web presence as a correct outcome, not a failure
