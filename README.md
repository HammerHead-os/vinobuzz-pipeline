# VinoBuzz Wine Photo Pipeline 🍷

An automated pipeline that finds, verifies, and scores wine product photos from the web. Built for Hong Kong deployment, targeting 70% minimum accuracy across VinoBuzz's catalog.

**Current Status**: 40% PASS on 5 SKU test (with DuckDuckGo search). Wine bottle detection added to filter non-wine images. Google Custom Search integration ready (requires valid API key).

---

## How It Works

```
Wine SKU → Search → Download → Crop Label → Read Label → Cross-check OCR → Quality Check → Score → Pass/Fail
```

1. **Search** — Finds candidate bottle photos using Google Custom Search API (highest quality) with fallback to DuckDuckGo Images. Falls back to retailer scraping (Vivino, Wine-Searcher) and producer websites.

2. **Label Extraction** — OpenCV crops just the label region from the bottle photo, removing background noise.

3. **Fingerprint Verification** — Gemini 2.5 Flash reads the cropped label and extracts producer, appellation, cru/vineyard, and vintage. Compares each field against the SKU metadata using fuzzy matching.

4. **OCR Cross-check** — Google Cloud Vision OCR independently reads the label text. If it agrees with Gemini, small confidence boost. If it disagrees, small penalty.

5. **Quality Filter** — Gemini checks the image for watermarks, blur, single upright bottle, clean background, no lifestyle props, and not AI-generated.

6. **Confidence Scoring** — Weighted score across all dimensions. Pass (≥70%), Quarantine (50-70%), or Reject (<50%).

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/HammerHead-os/vinobuzz-pipeline.git
cd vinobuzz-pipeline
pip install -r requirements.txt
```

### 2. Set up credentials

You need a Google Cloud project with two APIs enabled:
- **Vertex AI API** (for Gemini vision)
- **Cloud Vision API** (for OCR)

Create a service account with the "Vertex AI User" role, download the JSON key, and drop it in the project root:

```bash
mv ~/Downloads/your-service-account.json ./credentials.json
```

Create a `.env` file:

```
GOOGLE_API_KEY=your-google-api-key
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
GOOGLE_CX=your-custom-search-engine-id
```

Get your Gemini API key at https://aistudio.google.com/apikey

For Google Custom Search (recommended for better accuracy):
1. Create a Programmable Search Engine at https://programmablesearchengine.google.com/
2. Get your Search Engine ID (CX)
3. Enable Custom Search API in Google Cloud Console
4. Create an API key with Custom Search API access

### 3. Run the benchmark

```bash
python scripts/benchmark.py --skus export_sku_20260427_062154.xlsx --bypass-cache
```

Or use the test SKUs:

```bash
python scripts/benchmark.py --skus data/test_skus.json --bypass-cache
```

### 4. Launch the demo UI

```bash
PYTHONPATH=src streamlit run src/wine_pipeline/app.py
```

Open http://localhost:8501 to see results with images, scores, and verdicts.

---

## Project Structure

```
├── src/wine_pipeline/
│   ├── app.py              # Streamlit demo UI
│   ├── cache.py            # SQLite result caching
│   ├── label_extractor.py  # OpenCV label cropping
│   ├── models.py           # Data models (SKU, Fingerprint, ScoredResult, etc.)
│   ├── ocr.py              # Google Cloud Vision OCR cross-checker
│   ├── pipeline.py         # Main orchestrator wiring all stages together
│   ├── quality_filter.py   # Gemini-based image quality evaluation
│   ├── scoring.py          # Confidence scoring and verdict assignment
│   ├── search.py           # DuckDuckGo image search with fallback chain
│   └── verifier.py         # Gemini-based label fingerprint extraction
├── tests/                  # 212 tests (unit + property-based)
├── scripts/
│   ├── benchmark.py        # Benchmark runner with formatted output
│   └── test_single_sku.py  # Single SKU testing utility
├── data/
│   ├── test_skus.json      # 10 test wines (pipeline must find these)
│   ├── reference_skus.json # 10 reference wines (live on VinoBuzz)
│   ├── production_skus.json # Production SKU dataset
│   └── images/             # Locally saved candidate images
├── export_sku_20260427_062154.xlsx  # Main SKU Excel file (298 wines)
├── charts/                 # Visualization and analysis charts
├── .kiro/specs/wine-photo-pipeline/
│   ├── requirements.md     # Full requirements specification
│   ├── design.md           # Design document with architecture
│   └── tasks.md            # Implementation task list
├── WRITEUP.txt             # Detailed write-up (approach, decisions, results)
├── MEETING_PREP.md         # Meeting preparation notes
├── REQUIREMENTS.md         # Full requirements spec
└── requirements.txt        # Python dependencies
```

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Vision AI | Google Gemini 2.5 Flash (Vertex AI) | Available in HK, strong label reading |
| OCR | Google Cloud Vision (asia-east2) | HK endpoint, independent cross-reference |
| Image Search | Google Custom Search (primary) + DuckDuckGo (fallback) | Google = high quality, DDG = free backup |
| Label Cropping | OpenCV | Simple contour detection, works well |
| String Matching | rapidfuzz | Fast fuzzy matching with Levenshtein distance |
| Cache | SQLite | Zero config, single file |
| Demo UI | Streamlit | Quick visual demo |
| Testing | pytest + Hypothesis | Unit tests + property-based testing |

### Why not GPT-4o?
OpenAI is not available in Hong Kong.

### Why Google Custom Search?
DuckDuckGo returns mixed results (random images, stock photos) which hurts accuracy. Google Custom Search provides higher quality, more relevant wine bottle images.

---

## Recent Improvements

### Wine Bottle Detection
Added `is_wine_bottle` field to fingerprint extraction. GPT-4o/Gemini now explicitly checks if the downloaded image is actually a wine bottle before processing. This filters out:
- Stock photos
- Random images (motorcycles, flowers, etc.)
- Non-product lifestyle shots

### Google Custom Search Integration
Added Google Custom Search API as primary search source for higher quality results. Falls back to DuckDuckGo if Google quota exceeded or API key missing.

### Field Mapping Fix
Fixed Excel SKU loading - `sub_region` column contains the actual cru/appellation (e.g., "Clos de Vougeot", "Richebourg"), not the generic `region` column ("Burgundy").

---

## Test Results

### 5 SKU Benchmark (DuckDuckGo Search)
```
SKU ID       Verdict      Confidence
S000001      PASS             0.80    (Domaine A.F. Gros Clos de Vougeot)
S000002      QUARANTINE       0.61    (Domaine A.F. Gros Moulin-à-Vent 2022)
S000003      REJECT           0.48    (Domaine A.F. Gros Moulin-à-Vent 2021)
S000004      PASS             0.96    (Domaine A.F. Gros Richebourg 2019)
S000005      QUARANTINE       0.63    (Domaine A.F. Gros Richebourg 2020)

Accuracy: 2/5 PASS (40%), 2/5 QUARANTINE (40%), 1/5 REJECT (20%)
```

### Historical Results (Test SKUs)
```
Accuracy: 8/10 PASS (80%), best run 9/10 (90%)
```

**Note**: Accuracy varies based on search quality. Google Custom Search expected to significantly improve results.

---

## Cost Estimate (4,000 SKUs)

| Service | Cost |
|---------|------|
| Gemini 2.5 Flash | ~$3.60 |
| Cloud Vision OCR | ~$16.50 |
| DuckDuckGo search | Free |
| **Total** | **~$20** |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 212 tests pass:
- Unit tests for each component
- Property-based tests with Hypothesis
- Integration tests for the full pipeline

---

## Documentation

- **Requirements**: See `REQUIREMENTS.md` and `.kiro/specs/wine-photo-pipeline/requirements.md`
- **Design**: See `.kiro/specs/wine-photo-pipeline/design.md`
- **Implementation Plan**: See `.kiro/specs/wine-photo-pipeline/tasks.md`
- **Write-up**: See `WRITEUP.txt`

---

## License

Built for the VinoBuzz internship assignment.
