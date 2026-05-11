#!/usr/bin/env python3
"""Quick test: run the pipeline on a single easy SKU."""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from wine_pipeline.models import SKU
from wine_pipeline.pipeline import Pipeline
from wine_pipeline.search import SearchModule
from wine_pipeline.label_extractor import LabelExtractor
from wine_pipeline.verifier import FingerprintVerifier
from wine_pipeline.ocr import OCRCrossChecker
from wine_pipeline.quality_filter import QualityFilter
from wine_pipeline.scoring import ConfidenceScorer
from wine_pipeline.cache import ResultCache


async def main():
    sku = SKU(
        id="test-004",
        producer="Château Fonroque",
        appellation="Saint-Émilion Grand Cru Classé",
        cru_vineyard=None,
        vintage=2016,
        format="750ml",
        region="Bordeaux",
    )

    pipeline = Pipeline(
        search=SearchModule(),
        label_extractor=LabelExtractor(),
        verifier=FingerprintVerifier(),
        ocr_checker=OCRCrossChecker(),
        quality_filter=QualityFilter(),
        scorer=ConfidenceScorer(),
        cache=ResultCache(":memory:"),
    )

    result = await pipeline.process_sku(sku, bypass_cache=True)
    print()
    print("=== RESULT ===")
    print(json.dumps(result.to_json(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
