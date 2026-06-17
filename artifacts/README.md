# Artifacts

Runtime outputs from batch runs. Do not commit images or ZIP files (see root `.gitignore`).

## Folder layout

| Path | Contents |
|------|----------|
| `Goldgate_wine/` | Client deliverable JPGs named by SKU (e.g. `S004481.jpg`) |
| `Goldgate_wine.zip` | ZIP of deliverable images |
| `notes/` | Meeting prep, assignment briefs, project notes |
| `test_run_100/` | Pilot run timing and gallery for 100 SKU test |
| `08_06_2026_run.log` | Main 283 SKU batch log |
| `corrected_rerun.log` | CORRECTED.xlsx flagged SKU rerun log |
| `sku_time_cost_breakdown.csv` | Per SKU time and API cost estimates |

## Regenerating deliverables

```bash
python scripts/process_pricelist.py "08_06_2026.xlsx" artifacts/Goldgate_wine artifacts/Goldgate_wine.zip
```

## Review UI

```bash
streamlit run scripts/review_pricelist.py
```
