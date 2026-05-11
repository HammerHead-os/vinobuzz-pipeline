# Requirements Document

## Introduction

Build an automated wine label matching system that processes wine SKUs by searching for images, extracting labels, reading text via OCR, and cross-checking against the SKU metadata. The system should handle batch processing of ~100 SKUs, cache results, and return matched images with confidence scores or placeholders when no match is found.

## Glossary

- **SKU**: Stock Keeping Unit - a unique identifier for a wine product with metadata (producer, appellation, cru, vintage)
- **Candidate Image**: A wine bottle image found through search that may match the target SKU
- **Label Extraction**: The process of cropping a wine label from a full bottle image using computer vision
- **Fingerprint**: Extracted text fields (producer, appellation, cru, vintage) from a label image
- **OCR**: Optical Character Recognition - converting image text to machine-readable text
- **Cross-Check**: Comparing OCR results from multiple sources (Google Gemini 2.5 Flash and Google Cloud Vision) to verify accuracy
- **Quality Filter**: Evaluating image quality to ensure labels are readable and not distorted
- **Confidence Score**: A weighted combination of field matches and quality metrics indicating match reliability
- **Verdict**: Final classification (PASS, QUARANTINE, REJECT) for each SKU result
- **Result Cache**: SQLite database storing processed results to avoid redundant computation

## Requirements

### Requirement 1: Batch Input Processing

**User Story:** As a data engineer, I want to process wine SKUs in batches, so that I can handle large product catalogs efficiently.

#### Acceptance Criteria

1. WHEN a batch of wine SKUs is provided, THE System SHALL process all SKUs in the batch
2. WHERE a batch contains up to 100 SKUs, THE System SHALL process the entire batch without manual intervention
3. IF a batch exceeds 100 SKUs, THEN THE System SHALL process the first 100 SKUs and log a warning for the remainder
4. WHEN a batch is processed, THE System SHALL return results for each SKU in the same order as input

### Requirement 2: Image Search

**User Story:** As a user, I want the system to search for wine images using DuckDuckGo, so that I can find candidate images for matching.

#### Acceptance Criteria

1. WHEN a wine SKU is provided, THE System SHALL search DuckDuckGo for candidate wine images
2. WHEN candidate images are found, THE System SHALL return up to 3 candidate images with URLs and source information
3. IF no candidate images are found, THEN THE System SHALL return an empty candidate list
4. WHEN image search fails, THEN THE System SHALL log the error and return an empty candidate list

### Requirement 3: Label Extraction

**User Story:** As a user, I want the system to extract wine labels from images using OpenCV, so that I can isolate the label for text reading.

#### Acceptance Criteria

1. WHEN a candidate image is provided, THE System SHALL attempt to extract the wine label
2. IF a label is successfully extracted, THEN THE System SHALL return a cropped label image
3. IF no label is detected, THEN THE System SHALL return the full image as fallback
4. WHEN label extraction fails, THEN THE System SHALL log the error and return a failure indicator

### Requirement 4: Text Reading with Multiple OCR Sources

**User Story:** As a user, I want the system to read text from wine labels using Google Gemini 2.5 Flash via Vertex AI, so that I can extract label metadata.

#### Acceptance Criteria

1. WHEN a label image is provided, THE System SHALL send it to Google Gemini 2.5 Flash via Vertex AI for text extraction
2. WHEN Gemini OCR returns text, THE System SHALL parse it into field candidates (producer, appellation, cru, vintage)
3. WHEN Gemini OCR fails, THEN THE System SHALL attempt Google Cloud Vision OCR as fallback
4. WHEN both OCR sources fail, THEN THE System SHALL return empty field candidates

### Requirement 5: Cross-Check Verification

**User Story:** As a user, I want the system to cross-check OCR results between Google Gemini 2.5 Flash and Google Cloud Vision, so that I can verify field accuracy.

#### Acceptance Criteria

1. WHEN both OCR sources return results, THE System SHALL compare field values
2. WHEN fields agree between sources, THE System SHALL mark them as confirmed
3. WHEN fields disagree between sources, THEN THE System SHALL flag them as disagreed
4. WHEN only one OCR source returns results, THE System SHALL use those results without cross-check

### Requirement 6: Field-by-Field Fuzzy Matching

**User Story:** As a user, I want the system to perform fuzzy matching on each field (producer, appellation, cru, vintage), so that I can match labels even with OCR errors.

#### Acceptance Criteria

1. WHEN a fingerprint is extracted, THE System SHALL compare each field against the SKU metadata
2. FOR EACH field, THE System SHALL calculate a match score from 0.0 to 1.0
3. WHERE a field is not present in the fingerprint, THE System SHALL assign a score of 0.0
4. WHERE a field is present in both fingerprint and SKU, THE System SHALL use fuzzy string matching

### Requirement 7: Quality Assessment

**User Story:** As a user, I want the system to evaluate image quality, so that I can filter out poor-quality labels.

#### Acceptance Criteria

1. WHEN a candidate image is processed, THE System SHALL evaluate its quality
2. WHEN image quality is below threshold, THEN THE System SHALL mark the result for rejection
3. WHEN image quality is above threshold, THE System SHALL include it in scoring
4. WHEN quality evaluation fails, THEN THE System SHALL log the error and assign minimum quality score

### Requirement 8: Confidence Scoring

**User Story:** As a user, I want the system to calculate an overall confidence score, so that I can rank matches by reliability.

#### Acceptance Criteria

1. WHEN all scores are available, THE System SHALL calculate overall confidence as a weighted combination
2. WHERE producer match is high, THE System SHALL increase overall confidence
3. WHERE appellation match is high, THE System SHALL increase overall confidence
4. WHERE cru match is high, THE System SHALL increase overall confidence
5. WHERE vintage match is high, THE System SHALL increase overall confidence
6. WHERE image quality is high, THE System SHALL increase overall confidence
7. WHEN fields disagree between OCR sources, THEN THE System SHALL reduce overall confidence

### Requirement 9: Result Caching

**User Story:** As a data engineer, I want results cached in SQLite, so that I can avoid redundant processing.

#### Acceptance Criteria

1. WHEN a SKU result is computed, THE System SHALL store it in the SQLite cache
2. WHEN a SKU is processed again, THE System SHALL return the cached result
3. WHERE bypass_cache is enabled, THE System SHALL skip cache and reprocess
4. WHEN cache storage fails, THEN THE System SHALL log the error but continue processing

### Requirement 10: Output Generation

**User Story:** As a user, I want the system to output matched images or placeholders with confidence scores, so that I can review results.

#### Acceptance Criteria

1. WHEN a match is found, THE System SHALL return the matched image URL with confidence scores
2. WHEN no match is found, THE System SHALL return a placeholder result with zero confidence
3. WHEN a result is quarantined, THE System SHALL return the image with rejection reasons
4. WHEN results are exported, THE System SHALL include all fields: SKU ID, image URL, field scores, overall confidence, verdict, and rejection reasons

### Requirement 11: CLI Execution

**User Story:** As a developer, I want to run the pipeline as a CLI script, so that I can integrate it into automation workflows.

#### Acceptance Criteria

1. WHEN the pipeline script is executed, THE System SHALL process SKUs from a JSON file
2. WHEN the pipeline script is executed, THE System SHALL output results to a JSON file
3. WHEN the pipeline script is executed, THE System SHALL accept command-line arguments for configuration
4. WHEN the pipeline script is executed, THE System SHALL log progress to stdout

### Requirement 12: Error Handling

**User Story:** As a user, I want the system to handle errors gracefully, so that I can identify processing issues.

#### Acceptance Criteria

1. WHEN an image download fails, THEN THE System SHALL log the error and continue with next candidate
2. WHEN OCR processing fails, THEN THE System SHALL log the error and return empty fields
3. WHEN a batch contains errors, THE System SHALL return partial results for successful SKUs
4. WHEN critical errors occur, THEN THE System SHALL log the error and terminate gracefully
