#!/usr/bin/env python3
"""Re-run image sourcing for low-confidence or missing SKUs.

Uses first-match selection (trusts Serper/Google search rank) and skips the
strict bottle-margin gate that was rejecting official producer photos.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import shutil
import sys
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from dotenv import load_dotenv

load_dotenv()

from process_pricelist import (  # noqa: E402
    _base_wine_key,
    _extract_producer,
    _safe_sku_filename,
    log_time,
)
from wine_pipeline.cache import ResultCache
from wine_pipeline.label_extractor import LabelExtractor
from wine_pipeline.models import SKU
from wine_pipeline.ocr import OCRCrossChecker
from wine_pipeline.pipeline import Pipeline
from wine_pipeline.quality_filter import QualityFilter
from wine_pipeline.scoring import ConfidenceScorer
from wine_pipeline.search import SearchModule
from wine_pipeline.verifier import FingerprintVerifier

import pandas as pd

logger = logging.getLogger("rerun_low_quality")

RESULT_RE = re.compile(
    r"  ✓ (S\d+) confidence=(\d+)% producer=(\d+)%"
)


def parse_last_results(log_path: Path) -> dict[str, dict[str, int]]:
    """Return last logged confidence/producer per SKU from a run log."""
    results: dict[str, dict[str, int]] = {}
    if not log_path.is_file():
        return results
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RESULT_RE.search(line)
        if m:
            sku_id, conf, producer = m.groups()
            results[sku_id] = {"confidence": int(conf), "producer": int(producer)}
    return results


def load_skus_from_excel(excel_path: Path) -> list[SKU]:
    df = pd.read_excel(excel_path)

    def _col(*names: str) -> str:
        for n in names:
            if n in df.columns:
                return n
        raise KeyError(f"Missing column; tried {names}")

    code_col = _col("Code", "wine_id", "sku_id", "id")
    name_col = _col("Name", "full_wine_name", "full_name", "wine_name", "name")
    vintage_col = _col("Vintage", "vintage")
    volume_col = next((c for c in ("Volume", "format") if c in df.columns), None)
    type_col = next((c for c in ("Type", "wine_type") if c in df.columns), None)
    country_col = next((c for c in ("Country", "country") if c in df.columns), None)
    grape_col = next((c for c in ("Grape", "grapes") if c in df.columns), None)
    region_col = next(
        (c for c in ("sub_region", "Region", "region") if c in df.columns),
        None,
    )

    skus: list[SKU] = []
    for _, row in df.iterrows():
        code = str(row[code_col])
        name = str(row[name_col])
        vintage_str = str(row[vintage_col]) if pd.notna(row[vintage_col]) else None
        vintage = None
        if vintage_str and vintage_str.upper() != "NV":
            try:
                vintage = int(vintage_str)
            except ValueError:
                pass
        region = ""
        if region_col is not None and pd.notna(row.get(region_col)):
            region = str(row[region_col]).strip()
        grape = ""
        if grape_col is not None and pd.notna(row.get(grape_col)):
            grape = str(row[grape_col])

        skus.append(
            SKU(
                id=code,
                producer=_extract_producer(name),
                appellation=region,
                cru_vineyard=name,
                vintage=vintage,
                format=str(row[volume_col])
                if (volume_col and pd.notna(row.get(volume_col)))
                else "750ml",
                region=region,
                full_name=name,
                wine_type=str(row[type_col])
                if (type_col and pd.notna(row.get(type_col)))
                else None,
                country=str(row[country_col])
                if (country_col and pd.notna(row.get(country_col)))
                else None,
                grapes=grape,
            )
        )
    return skus


def select_targets(
    skus: list[SKU],
    img_dir: Path,
    log_results: dict[str, dict[str, int]],
    min_confidence: int,
    min_producer: int,
    include_missing: bool,
) -> list[SKU]:
    targets: list[SKU] = []
    for sku in skus:
        safe = _safe_sku_filename(sku.id)
        path = img_dir / f"{safe}.jpg"
        has_image = path.is_file() and path.stat().st_size > 0
        stats = log_results.get(sku.id, {})
        conf = stats.get("confidence", 0 if not has_image else 100)
        producer = stats.get("producer", 0 if not has_image else 100)

        low_conf = conf < min_confidence
        low_producer = producer < min_producer
        missing = include_missing and not has_image

        if missing or (has_image and (low_conf or low_producer)):
            targets.append(sku)
    return targets


async def rerun_low_quality(
    excel_path: str,
    log_path: str,
    output_dir: str,
    zip_name: str,
    min_confidence: int = 80,
    min_producer: int = 70,
    max_candidates: int = 12,
    sku_timeout_sec: int = 120,
) -> None:
    excel = Path(excel_path)
    log_file = Path(log_path)
    out_dir = Path(output_dir)
    img_cache = Path("data/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_cache.mkdir(parents=True, exist_ok=True)

    skus = load_skus_from_excel(excel)
    log_results = parse_last_results(log_file)
    targets = select_targets(
        skus, img_cache, log_results, min_confidence, min_producer, True
    )

    logger.info(
        "Re-running %d SKUs (confidence<%d%% or producer<%d%% or missing)",
        len(targets),
        min_confidence,
        min_producer,
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
    pipeline._skip_ocr = True
    pipeline._skip_quality = False
    pipeline._strict_quality = False
    pipeline._require_pass_verdict = False
    pipeline._strict_match = False
    pipeline._skip_removebg = True
    pipeline.search._vivino_first_only = True

    improved = 0
    base_wine_images: dict[str, Path] = {}

    for idx, sku in enumerate(targets):
        safe_id = _safe_sku_filename(sku.id)
        src_path = img_cache / f"{safe_id}.jpg"
        base_key = _base_wine_key(sku.full_name or "")

        logger.info("[%d/%d] %s: %s", idx + 1, len(targets), sku.id, sku.full_name)

        if base_key and base_key in base_wine_images:
            donor = base_wine_images[base_key]
            shutil.copy(donor, src_path)
            logger.info("  ↻ Reused image from same wine")
            improved += 1
            continue

        if src_path.exists():
            src_path.unlink()

        try:
            result = await asyncio.wait_for(
                pipeline.process_sku(
                    sku, bypass_cache=True, max_candidates=max_candidates
                ),
                timeout=sku_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.error("  Timeout (%ds)", sku_timeout_sec)
            continue
        except Exception as exc:
            logger.error("  Error: %s", exc)
            continue

        if src_path.exists() and src_path.stat().st_size > 0:
            if base_key:
                base_wine_images[base_key] = src_path
            improved += 1
            logger.info(
                "  ✓ conf=%.0f%% producer=%.0f%% url=%s",
                result.overall_confidence * 100,
                result.producer_match * 100,
                (result.image_url or "")[:80],
            )
        else:
            logger.warning("  No image saved")

    logger.info("Copying images and rebuilding ZIP...")
    count = 0
    for sku in skus:
        safe_id = _safe_sku_filename(sku.id)
        src = img_cache / f"{safe_id}.jpg"
        if src.is_file() and src.stat().st_size > 0:
            shutil.copy(src, out_dir / f"{safe_id}.jpg")
            count += 1

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in out_dir.glob("*.jpg"):
            zf.write(img, img.name)

    logger.info("Done — %d/%d targets improved, %d total images in ZIP", improved, len(targets), count)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Re-run low-quality pricelist images")
    parser.add_argument("--excel", default="08_06_2026.xlsx")
    parser.add_argument("--log", default="artifacts/08_06_2026_run.log")
    parser.add_argument("--output-dir", default="artifacts/08_06_2026_images")
    parser.add_argument("--zip", default="artifacts/08_06_2026_images.zip")
    parser.add_argument("--min-confidence", type=int, default=80)
    parser.add_argument("--min-producer", type=int, default=70)
    args = parser.parse_args()
    asyncio.run(
        rerun_low_quality(
            args.excel,
            args.log,
            args.output_dir,
            args.zip,
            min_confidence=args.min_confidence,
            min_producer=args.min_producer,
        )
    )


if __name__ == "__main__":
    main()
