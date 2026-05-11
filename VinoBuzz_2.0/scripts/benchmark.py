#!/usr/bin/env python3
"""Benchmark runner for the Wine Photo Pipeline.

Loads test SKUs (and optionally reference SKUs), runs the pipeline,
and outputs per-SKU results plus overall accuracy metrics.

Requirements: 9.1, 9.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure the src package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wine_pipeline.cache import ResultCache
from wine_pipeline.label_extractor import LabelExtractor
from wine_pipeline.models import SKU, Verdict
from wine_pipeline.ocr import OCRCrossChecker
from wine_pipeline.pipeline import Pipeline
from wine_pipeline.quality_filter import QualityFilter
from wine_pipeline.scoring import ConfidenceScorer
from wine_pipeline.search import SearchModule
from wine_pipeline.verifier import FingerprintVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_skus(path: Path) -> list[SKU]:
    """Load SKU objects from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
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
        for item in raw
    ]


def build_pipeline(cache_path: str = ":memory:") -> Pipeline:
    """Construct a Pipeline with real components.

    API keys are read from environment variables:
      - GEMINI_API_KEY or GOOGLE_API_KEY (used by google-genai SDK)
      - GOOGLE_APPLICATION_CREDENTIALS (used implicitly by google-cloud-vision)
    """
    search = SearchModule()
    label_extractor = LabelExtractor()
    verifier = FingerprintVerifier()
    ocr_checker = OCRCrossChecker()
    quality_filter = QualityFilter()
    scorer = ConfidenceScorer()
    cache = ResultCache(db_path=cache_path)

    return Pipeline(
        search=search,
        label_extractor=label_extractor,
        verifier=verifier,
        ocr_checker=ocr_checker,
        quality_filter=quality_filter,
        scorer=scorer,
        cache=cache,
    )


def print_result_table(results: list) -> None:
    """Print a formatted table of per-SKU results."""
    header = f"{'SKU ID':<12} {'Verdict':<12} {'Confidence':>10}  {'Image URL'}"
    print("\n" + header)
    print("-" * len(header) + "-" * 40)
    for r in results:
        url = r.image_url or "No Image"
        if len(url) > 60:
            url = url[:57] + "..."
        print(f"{r.sku_id:<12} {r.verdict.value:<12} {r.overall_confidence:>10.4f}  {url}")


def print_summary(results: list) -> None:
    """Print summary accuracy metrics."""
    total = len(results)
    if total == 0:
        print("\nNo results to summarise.")
        return

    counts = {v: 0 for v in Verdict}
    for r in results:
        counts[r.verdict] += 1

    print("\n--- Summary ---")
    for v in Verdict:
        pct = counts[v] / total * 100
        print(f"  {v.value:<12} {counts[v]:>3} / {total}  ({pct:.1f}%)")

    accepted = counts[Verdict.PASS]
    print(f"\n  Accuracy (PASS): {accepted}/{total} = {accepted / total * 100:.1f}%")


async def run_benchmark(
    sku_file: Path,
    bypass_cache: bool = False,
    cache_path: str = "benchmark_cache.db",
) -> None:
    """Load SKUs, run the pipeline, and print results."""
    skus = load_skus(sku_file)
    logger.info("Loaded %d SKUs from %s", len(skus), sku_file)

    pipeline = build_pipeline(cache_path=cache_path)

    start = time.perf_counter()
    results = await pipeline.process_batch(skus, bypass_cache=bypass_cache)
    elapsed = time.perf_counter() - start

    print_result_table(results)
    print_summary(results)
    print(f"\n  Total time: {elapsed:.2f}s  ({elapsed / len(skus):.2f}s per SKU)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wine Photo Pipeline Benchmark")
    parser.add_argument(
        "--skus",
        type=str,
        default=str(DATA_DIR / "test_skus.json"),
        help="Path to SKU JSON file (default: data/test_skus.json)",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Run on reference SKUs instead of test SKUs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on both test and reference SKUs",
    )
    parser.add_argument(
        "--bypass-cache",
        action="store_true",
        help="Bypass the result cache and re-run the full pipeline",
    )
    parser.add_argument(
        "--cache-db",
        type=str,
        default="benchmark_cache.db",
        help="Path to SQLite cache database (default: benchmark_cache.db)",
    )
    args = parser.parse_args()

    if args.all:
        print("=== Test SKUs ===")
        asyncio.run(
            run_benchmark(
                DATA_DIR / "test_skus.json",
                bypass_cache=args.bypass_cache,
                cache_path=args.cache_db,
            )
        )
        print("\n=== Reference SKUs ===")
        asyncio.run(
            run_benchmark(
                DATA_DIR / "reference_skus.json",
                bypass_cache=args.bypass_cache,
                cache_path=args.cache_db,
            )
        )
    elif args.reference:
        asyncio.run(
            run_benchmark(
                DATA_DIR / "reference_skus.json",
                bypass_cache=args.bypass_cache,
                cache_path=args.cache_db,
            )
        )
    else:
        asyncio.run(
            run_benchmark(
                Path(args.skus),
                bypass_cache=args.bypass_cache,
                cache_path=args.cache_db,
            )
        )


if __name__ == "__main__":
    main()
