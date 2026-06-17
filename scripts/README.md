# Scripts

| Script | Purpose |
|--------|---------|
| `process_pricelist.py` | Main batch: Excel → search → verify → JPG + ZIP |
| `rerun_corrected.py` | Re-run orange/yellow flagged SKUs from CORRECTED.xlsx |
| `rerun_low_quality.py` | Re-run low confidence or missing SKUs |
| `review_pricelist.py` | Streamlit review UI (port 8501) |
| `benchmark.py` | Accuracy benchmark on test/reference SKUs |
| `evaluate_pricelist_results.py` | Score saved images without re-searching |
| `build_image_gallery.py` | Build HTML image gallery from a folder |
| `watch_live_gallery.py` | Auto-refresh gallery during batch runs |

## Examples

```bash
# Full pricelist
python scripts/process_pricelist.py "08_06_2026.xlsx" artifacts/Goldgate_wine artifacts/Goldgate_wine.zip

# Resume from SKU
python scripts/process_pricelist.py "08_06_2026.xlsx" artifacts/Goldgate_wine artifacts/Goldgate_wine.zip --from S004725

# CORRECTED.xlsx flagged rerun
python scripts/rerun_corrected.py --excel CORRECTED.xlsx --output-dir artifacts/Goldgate_wine --zip artifacts/Goldgate_wine.zip

# Review site
streamlit run scripts/review_pricelist.py
```
