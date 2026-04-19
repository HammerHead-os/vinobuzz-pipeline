# VinoBuzz Wine Photo Pipeline — Requirements

## Objective
Build an automated pipeline that sources and verifies wine product photos at 90%+ accuracy (up from ~50% baseline). Designed for Hong Kong deployment, targeting VinoBuzz's 4,000+ SKU catalog.

---

## Tech Stack (HK-Optimized)

| Component | Tool | Why |
|---|---|---|
| Language | Python (async with httpx) | First-class SDK support, parallelism for speed |
| Image Search | SerpAPI / Google Custom Search API | No geo-restrictions from HK, image metadata included |
| Fallback Scraping | Playwright | Wine retailer sites (Vivino, Wine-Searcher, producer sites) |
| Primary Verification | Google Gemini 2.5 Flash (vision) | Available in HK, strong label reading, cost-effective |
| Secondary OCR | Google Cloud Vision OCR (asia-east2) | Local HK endpoint, independent cross-reference |
| Label Cropping | OpenCV | Crop label region before verification |
| Demo UI | Streamlit | Zero-config, visual demo |
| Cache | SQLite | Avoid reprocessing at scale |

### Cost Estimates
- Gemini 2.5 Flash vision: ~$0.15/1K images (input) — significantly cheaper than GPT-4o
- SerpAPI: 100 free searches/month
- Google Vision OCR: 1,000 free units/month

---

## Pipeline Architecture

### Stage 1: Multi-Source Search (Fallback Chain)
1. Google Images via SerpAPI
2. Wine retailer sites (Vivino, Wine-Searcher)
3. Producer's own website
4. All fail → "No Image" (intentional, not a failure)

### Stage 2: Label Region Extraction
- Use OpenCV contour detection to crop just the label area from each candidate image
- Removes background noise, shelf clutter, other bottles
- Cropped label fed to verification stage for higher accuracy
- Visually demonstrable: show raw image → cropped label → extracted data in demo

### Stage 3: Structured Fingerprint Verification
Via Gemini vision on the cropped label, extract:
```json
{
  "producer": "...",
  "appellation": "...",
  "cru_vineyard": "...",
  "vintage": "..."
}
```
Field-by-field comparison against SKU metadata. No vague "is this the right wine?" — explicit matching on each dimension independently.

Cross-reference with Google Vision OCR as secondary check. If Gemini and OCR disagree, flag for review.

### Stage 4: Image Quality Filter
- No watermarks
- No blur or glare
- Single bottle, upright
- White/neutral grey background
- No lifestyle props
- No AI-generated images

### Stage 5: Confidence Scoring (Per-Dimension)
Output per SKU:
```json
{
  "producer_match": 0.98,
  "appellation_match": 0.95,
  "cru_match": 0.92,
  "vintage_match": 0.85,
  "image_quality": 0.90,
  "overall_confidence": 0.92,
  "verdict": "PASS"
}
```

### Three-Tier Decision
- ≥85% → Auto-accept
- 60–85% → Quarantine (human review queue)
- <60% → Auto-reject → "No Image"

---

## Edge Cases

- **NV (Non-Vintage) wines** — Skip vintage matching, adjust scoring weights
- **Near-zero web presence** — Fallback chain handles this, "No Image" is valid
- **Burgundy name confusion** — Field-by-field fingerprint matching is critical (Olivier Leflaive ≠ Domaine Leflaive)
- **Lifestyle/marketing images** — Quality filter rejects these
- **Watermarked images** — Detect and reject

---

## Deliverables

### 1. Working Demo (Streamlit UI)
- Runs on all 10 test SKUs
- Per-SKU output: photo URL or "No Image", per-dimension confidence scores, pass/fail
- Visual before/after: raw image → cropped label → extracted fingerprint
- Also benchmark on 10 reference SKUs against VinoBuzz's approved photos

### 2. Write-Up (Max 2 Pages or 5 Slides)
- Lead with accuracy number: "X/10 correct. Here's the one it missed and why."
- Pipeline architecture diagram
- Hard case walkthrough (e.g., Arnot-Roberts Trousseau Gris step-by-step)
- Failure analysis: 2-3 SKUs that struggled, exactly why
- Cost/latency table: time per SKU, API calls, estimated cost for 4,000 SKUs
- Prompt versioning: how verification prompts evolved
- Total time spent (honest)

### 3. Decision Log
- Why each tech choice was made
- Alternatives considered (CLIP, SAM, Claude vision) and why deferred
- HK-specific considerations

---

## Future Improvements (Mention in Write-Up, Don't Build)
- CLIP-based semantic search for visual matching when OCR fails
- Producer website photo anchoring as ground truth verification
- Active learning loop: analyze failure logs to tune thresholds per region
- SAM for more precise label segmentation
- Feedback loop from human review queue back into pipeline tuning

---

## Development Sequence

1. Basic search → Gemini verify loop on 2-3 easy SKUs
2. Add OpenCV label cropping
3. Add structured fingerprint extraction + field-by-field matching
4. Add Google Vision OCR cross-reference
5. Add image quality filter
6. Tune confidence thresholds
7. Run full test set + reference set benchmark
8. Build Streamlit demo
9. Write up results + decision log

Track time per phase. Report honestly.
