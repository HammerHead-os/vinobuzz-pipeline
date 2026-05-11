"""Streamlit Demo UI for the Wine Photo Pipeline.

Displays per-SKU results with verdict, confidence scores, and detail views.
Requirements: 8.1, 8.2, 8.3, 8.4
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from wine_pipeline.models import SKU, ScoredResult, Verdict


# ------------------------------------------------------------------
# Summary metrics computation (pure function, tested via PBT)
# ------------------------------------------------------------------


def compute_summary_metrics(results: list[ScoredResult]) -> dict:
    """Compute verdict counts and percentages from a list of ScoredResults.

    Returns a dict with keys: total, pass_count, quarantine_count, reject_count,
    pass_pct, quarantine_pct, reject_pct.
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "pass_count": 0,
            "quarantine_count": 0,
            "reject_count": 0,
            "pass_pct": 0.0,
            "quarantine_pct": 0.0,
            "reject_pct": 0.0,
        }

    pass_count = sum(1 for r in results if r.verdict == Verdict.PASS)
    quarantine_count = sum(1 for r in results if r.verdict == Verdict.QUARANTINE)
    reject_count = sum(1 for r in results if r.verdict == Verdict.REJECT)

    return {
        "total": total,
        "pass_count": pass_count,
        "quarantine_count": quarantine_count,
        "reject_count": reject_count,
        "pass_pct": pass_count / total,
        "quarantine_pct": quarantine_count / total,
        "reject_pct": reject_count / total,
    }


# ------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------


def _load_skus(path: str) -> list[SKU]:
    """Load SKU list from a JSON file."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return [
        SKU(
            id=item["id"],
            producer=item["producer"],
            appellation=item["appellation"],
            cru_vineyard=item.get("cru_vineyard"),
            vintage=item.get("vintage"),
            format=item.get("format", "750ml"),
            region=item.get("region", ""),
        )
        for item in data
    ]


def _load_cached_results() -> list[ScoredResult]:
    """Load all cached results from the SQLite database."""
    from wine_pipeline.cache import ResultCache

    db_path = os.environ.get("PIPELINE_CACHE_DB", "benchmark_cache.db")
    if not Path(db_path).exists():
        return []
    cache = ResultCache(db_path)
    try:
        conn = cache._conn
        rows = conn.execute("SELECT * FROM pipeline_results").fetchall()
        return [cache._row_to_result(row) for row in rows]
    finally:
        cache.close()


# ------------------------------------------------------------------
# Pipeline runner (async bridge)
# ------------------------------------------------------------------


def _run_pipeline_on_skus(skus: list[SKU]) -> list[ScoredResult]:
    """Instantiate the pipeline and process SKUs. Returns results."""
    from wine_pipeline.cache import ResultCache
    from wine_pipeline.label_extractor import LabelExtractor
    from wine_pipeline.ocr import OCRCrossChecker
    from wine_pipeline.pipeline import Pipeline
    from wine_pipeline.quality_filter import QualityFilter
    from wine_pipeline.scoring import ConfidenceScorer
    from wine_pipeline.search import SearchModule
    from wine_pipeline.verifier import FingerprintVerifier

    db_path = os.environ.get("PIPELINE_CACHE_DB", "benchmark_cache.db")
    pipeline = Pipeline(
        search=SearchModule(),
        label_extractor=LabelExtractor(),
        verifier=FingerprintVerifier(),
        ocr_checker=OCRCrossChecker(),
        quality_filter=QualityFilter(),
        scorer=ConfidenceScorer(),
        cache=ResultCache(db_path),
    )

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(pipeline.process_batch(skus))
    finally:
        loop.close()


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------


def _verdict_color(verdict: Verdict) -> str:
    return {
        Verdict.PASS: "🟢",
        Verdict.QUARANTINE: "🟡",
        Verdict.REJECT: "🔴",
    }.get(verdict, "⚪")


def main() -> None:
    st.set_page_config(page_title="VinoBuzz Wine Photo Pipeline", layout="wide")
    st.title("VinoBuzz Wine Photo Pipeline")

    # --- Sidebar: trigger pipeline ---
    st.sidebar.header("Run Pipeline")

    test_sku_path = st.sidebar.text_input(
        "Test SKUs JSON path", value="data/test_skus.json"
    )
    ref_sku_path = st.sidebar.text_input(
        "Reference SKUs JSON path", value="data/reference_skus.json"
    )

    if st.sidebar.button("Process Test SKUs"):
        skus = _load_skus(test_sku_path)
        if skus:
            with st.spinner("Processing test SKUs..."):
                _run_pipeline_on_skus(skus)
            st.sidebar.success(f"Processed {len(skus)} test SKUs")
        else:
            st.sidebar.warning("No test SKUs found at the given path.")

    if st.sidebar.button("Process Reference SKUs"):
        skus = _load_skus(ref_sku_path)
        if skus:
            with st.spinner("Processing reference SKUs..."):
                _run_pipeline_on_skus(skus)
            st.sidebar.success(f"Processed {len(skus)} reference SKUs")
        else:
            st.sidebar.warning("No reference SKUs found at the given path.")

    # --- Load cached results ---
    results = _load_cached_results()

    if not results:
        st.info("No processed results yet. Use the sidebar to run the pipeline.")
        return

    # --- Summary metrics ---
    metrics = compute_summary_metrics(results)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SKUs", metrics["total"])
    col2.metric("PASS", f"{metrics['pass_count']} ({metrics['pass_pct']:.0%})")
    col3.metric(
        "QUARANTINE",
        f"{metrics['quarantine_count']} ({metrics['quarantine_pct']:.0%})",
    )
    col4.metric("REJECT", f"{metrics['reject_count']} ({metrics['reject_pct']:.0%})")

    # --- SKU list table ---
    st.subheader("Processed SKUs")

    table_data = []
    for r in results:
        table_data.append(
            {
                "SKU ID": r.sku_id,
                "Verdict": f"{_verdict_color(r.verdict)} {r.verdict.value}",
                "Confidence": f"{r.overall_confidence:.2f}",
                "Image URL": r.image_url or "No Image",
            }
        )
    st.dataframe(table_data, use_container_width=True)

    # --- Detail view ---
    st.subheader("SKU Detail View")
    sku_ids = [r.sku_id for r in results]
    selected_id = st.selectbox("Select a SKU", sku_ids)

    if selected_id:
        selected = next((r for r in results if r.sku_id == selected_id), None)
        if selected:
            _render_detail(selected)


def _render_detail(result: ScoredResult) -> None:
    """Render the detail view for a single SKU result."""
    left, right = st.columns(2)

    with left:
        st.markdown("**Raw Candidate Image**")
        local_path = Path("data") / "images" / f"{result.sku_id}.jpg"
        if local_path.exists():
            st.image(str(local_path), use_container_width=True)
        elif result.image_url:
            st.image(result.image_url, use_container_width=True)
        else:
            st.write("No image available")

    with right:
        st.markdown("**Fingerprint Fields**")
        if result.fingerprint:
            st.json(
                {
                    "producer": result.fingerprint.producer,
                    "appellation": result.fingerprint.appellation,
                    "cru_vineyard": result.fingerprint.cru_vineyard,
                    "vintage": result.fingerprint.vintage,
                }
            )
        else:
            st.write("No fingerprint extracted")

    st.markdown("**Per-Dimension Scores**")
    score_cols = st.columns(5)
    score_cols[0].metric("Producer", f"{result.producer_match:.2f}")
    score_cols[1].metric("Appellation", f"{result.appellation_match:.2f}")
    score_cols[2].metric("Cru", f"{result.cru_match:.2f}")
    score_cols[3].metric("Vintage", f"{result.vintage_match:.2f}")
    score_cols[4].metric("Image Quality", f"{result.image_quality:.2f}")

    st.metric("Overall Confidence", f"{result.overall_confidence:.2f}")
    st.metric(
        "Verdict",
        f"{_verdict_color(result.verdict)} {result.verdict.value}",
    )

    if result.rejection_reasons:
        st.markdown("**Rejection Reasons**")
        for reason in result.rejection_reasons:
            st.write(f"- {reason}")


if __name__ == "__main__":
    main()
