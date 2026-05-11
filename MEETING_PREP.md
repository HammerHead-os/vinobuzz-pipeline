# Meeting Prep — Call with Rev
**Date:** April 27, 2026  
**Purpose:** Get technical clarity before building the pipeline

---

## Questions for Rev

### APIs & Access (Must-Ask — These Unblock Me)
1. What APIs do we have access to or budget for?
   - Image search: SerpAPI? Google Custom Search? Or am I sourcing my own?
   - Vision/OCR: Google Cloud Vision? Gemini? OpenAI?
2. Are there existing API keys, or do I provision my own?
3. Any rate limits or monthly spend caps I should design around?
4. Are there GCP credentials or a service account I should use? (Relevant because of the HK deployment — direct Gemini API is geo-blocked, need Vertex AI)

### Input Data
5. What format does the wine list come in? (CSV, JSON, Google Sheet, database?) database (list will be given of 100 SKUS will also have image url )
design upto me 


6. What fields are guaranteed per SKU? (Producer, appellation, cru/vineyard, vintage, format — anything else?)
7. How big is a typical batch? 10 wines? 100? 1000?
8. Can I get a sample batch beyond the 10 test SKUs to stress-test?

### Output & Delivery
9. Where do finished images go? (S3 bucket, local ZIP, API endpoint, shared drive?)
10. Expected image format/resolution/dimensions?
11. Is there a database or system where I log results (matched, placeholder, confidence)?
12. Does the output feed directly into the store, or does Jacky review first?

### Infrastructure & Scope
13. Tech stack preference? (Python seems obvious, just confirming)
14. Is this a CLI script, a web service, or does it need a UI?
15. Any existing codebase or repo I should build on top of?
16. Where should this run? (My machine, a cloud VM, Lambda?)
17. Is caching expected? (So we don't re-search wines we've already matched)

### Timeline & Expectations
18. Beyond the 48hr assignment — is this meant to become production software?
19. What does "done" look like for the internship submission vs. the real product?

---

## Strategy I'm Going With (and Why)

### Image Search: DuckDuckGo (via `ddgs` Python library)

**Why this:**
- Zero API keys, zero signup, completely free
- No geo-restrictions from Hong Kong
- Returns good enough results for wine bottle photos
- Gets us moving immediately without waiting on API provisioning

**Why not the alternatives:**
| Option | Why Not |
|--------|---------|
| SerpAPI | Phone verification didn't work during testing. $15/1K searches. |
| Serper.dev | Signup issues. Another paid dependency. |
| Google Custom Search API | Being deprecated (sunsetting Jan 2027). Bad to build on a dying service. |
| Direct Vivino/Wine-Searcher scraping | Both enforce aggressive CAPTCHA blocking. Jacky explicitly said don't do this. |

**The play:** Search Google/DuckDuckGo for wine names. Vivino and winery sites have the best photos, but they block bots. Google already indexes those images — so we get them through the back door via search thumbnails and cached links, without ever hitting those sites directly. This is exactly what Jacky's Manus prompt does manually.

---

### Vision & Label Verification: Google Gemini 2.5 Flash (via Vertex AI)

**Why this:**
- Google launched Gemini in HK in March 2026 — timing is perfect
- Strong at reading wine label text (producer, appellation, cru, vintage)
- Cost-effective: ~$0.15 per 1M input tokens → roughly $3.60 for 4,000 SKUs
- Structured output: extracts a JSON fingerprint per label for field-by-field comparison

**Why Vertex AI specifically:**
- Direct Gemini API (Google AI Studio) returns "User location is not supported" from HK
- Vertex AI uses a GCP service account and connects to us-central1 — bypasses the geo-block completely

**Why not the alternatives:**
| Option | Why Not |
|--------|---------|
| GPT-4o / OpenAI | Banned in Hong Kong. Non-starter. |
| Claude (Anthropic) | HK not listed as supported country. |
| Local OCR only (Tesseract) | Can't handle fancy wine label fonts. Needs a vision model that understands context. |

---

### OCR Cross-Check: Google Cloud Vision (asia-east2)

**Why this:**
- Independent second opinion on what the label says
- Has a Hong Kong endpoint (asia-east2) — low latency, no geo issues
- 1,000 free units/month
- If Gemini and OCR disagree on a field, we flag it — reduces false positives

**Why not skip it:**
- Gemini alone is ~80% reliable on label text. Adding OCR as a cross-reference catches the cases where Gemini hallucinates or misreads a word. Two independent readers > one.

---

### Label Cropping: OpenCV

**Why this:**
- Crops just the label area from the bottle photo before sending to Gemini
- Removes background noise, shelf clutter, other bottles
- Simple contour detection — no ML model needed
- Falls back to full image if it can't find a label region

**Why not SAM (Segment Anything):**
- Overkill. OpenCV contour detection works fine for rectangular wine labels.
- SAM adds a heavy model dependency for marginal improvement.
- Worth revisiting later if label cropping becomes a bottleneck.

---

### Scoring: Weighted Field-by-Field Matching

**Why this approach:**
- No vague "does this look right?" — explicit match on each dimension independently
- Producer (35%), Appellation (25%), Cru (15%), Vintage (15%), Image Quality (10%)
- Uses fuzzy string matching (rapidfuzz) + token overlap to handle OCR noise
- Three-tier decision: PASS (≥70%) / QUARANTINE (50-70%) / REJECT (<50%)

**Why not a single similarity score:**
- Wine names are tricky. "Domaine Leflaive Puligny-Montrachet" and "Olivier Leflaive Puligny-Montrachet" would score high on overall similarity but are completely different wines. Field-by-field catches this.

**Why 70% threshold (not 85%):**
- 85% was too strict — only 40% of test wines passed
- 70% balances accuracy vs. coverage, consistently hits 80-90% on the test set
- The QUARANTINE tier (50-70%) catches borderline cases for human review rather than auto-rejecting

---

### Fallback: "No Image" Placeholder

**Why:**
- Jacky was explicit: a wrong photo is worse than no photo
- If the pipeline can't find a verified match, it generates a grey placeholder named [SKU].jpg
- No SKU gets skipped — backend expects sequential processing
- This is a feature, not a failure

---

### Caching: SQLite

**Why this:**
- Zero config, single file, no server needed
- Avoids reprocessing wines we've already matched
- Perfect for a demo/prototype
- Easy to upgrade to Postgres later if this goes to production

---

## Summary: The Pipeline in 30 Seconds

```
Wine SKU → Build search query → DuckDuckGo Images → Top 3 candidates
    → OpenCV crops label region
    → Gemini reads label → extracts {producer, appellation, cru, vintage}
    → Google Vision OCR cross-checks
    → Field-by-field fuzzy matching against SKU metadata
    → Quality filter (watermarks, blur, background)
    → Confidence score → PASS / QUARANTINE / REJECT
    → Save as [SKU].jpg or placeholder
```

Cost for 4,000 SKUs: ~$20  
Time per SKU: ~25 seconds  
Accuracy on test set: 80-90%
