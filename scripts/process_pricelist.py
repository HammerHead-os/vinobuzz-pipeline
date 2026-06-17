#!/usr/bin/env python3
"""Process VinoBuzz pricelist - find images, rename by Code, create ZIP.

Workflow:
1. Load pricelist Excel file
2. For each wine, search for product images using the pipeline
3. Download the best matching image
4. Rename image using the Code column
5. Create ZIP file with all images
6. Log timing for each step
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pricelist_processor")

from wine_pipeline.cache import ResultCache
from wine_pipeline.label_extractor import LabelExtractor
from wine_pipeline.models import SKU, Verdict
from wine_pipeline.ocr import OCRCrossChecker
from wine_pipeline.pipeline import Pipeline
from wine_pipeline.quality_filter import QualityFilter
from wine_pipeline.scoring import ConfidenceScorer
from wine_pipeline.search import SearchModule
from wine_pipeline.verifier import FingerprintVerifier


# Timing log
TIMING_LOG = []

def log_time(step: str, start_time: float) -> float:
    """Log timing for a step."""
    elapsed = time.perf_counter() - start_time
    TIMING_LOG.append({"step": step, "seconds": round(elapsed, 2)})
    logger.info(f"[TIMING] {step}: {elapsed:.2f}s")
    return time.perf_counter()

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _base_wine_key(name: str) -> str:
    """Normalize a wine name for deduping across vintages.

    Removes obvious vintage tokens (e.g. 2019, 2021) and 'NV' markers.
    """
    s = _norm_text(name)
    s = re.sub(r"\b(19|20)\d{2}\b", "", s)  # remove 4-digit years
    s = re.sub(r"\bnv\b", "", s)
    s = re.sub(r"[^\w\s\-']", " ", s)  # normalize punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_producer(name: str) -> str:
    """Derive a meaningful producer name from the full wine name."""
    name = (name or "").strip()
    if not name:
        return name
    tokens = name.split()
    if tokens[0].lower() in ("domaine", "château", "chateau", "maison", "bodegas", "champagne"):
        if len(tokens) >= 3 and tokens[1].lower() in ("de", "du", "des", "la", "le"):
            return " ".join(tokens[:3])
        return " ".join(tokens[:2])
    if tokens[0] in ("M.", "M") and len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    if len(tokens) >= 2 and tokens[1].lower() in (
        "peak", "vineyards", "range", "sparks", "jaboulet", "chapoutier",
    ):
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0]


def _safe_sku_filename(sku_id: str) -> str:
    return sku_id.replace("/", "_").replace("\\", "_")


def _publish_image(sku_id: str, src_path: Path, output_dir: Path) -> None:
    """Copy a sourced image into the review/deliverable folder immediately."""
    if not src_path.is_file() or src_path.stat().st_size == 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    dst_path = output_dir / f"{_safe_sku_filename(sku_id)}.jpg"
    shutil.copy2(src_path, dst_path)


async def process_pricelist(
    excel_path: str,
    output_dir: str = "pricelist_images",
    zip_name: str = "vinobuzz_images.zip",
    limit: int | None = None,
    resume: bool = True,
    start_from_sku: str | None = None,
    max_candidates: int = 10,
    sku_timeout_sec: int = 120,
):
    """Process the pricelist and collect images."""
    total_start = time.perf_counter()
    t = total_start
    
    # Step 1: Load Excel
    logger.info(f"Loading pricelist from {excel_path}")
    df = pd.read_excel(excel_path)
    if limit is not None and limit > 0:
        df = df.head(limit)
    log_time("Load Excel file", t)
    t = time.perf_counter()
    
    logger.info(f"Found {len(df)} wines to process")
    
    os.makedirs(output_dir, exist_ok=True)
    img_cache = Path("data/images")
    img_cache.mkdir(parents=True, exist_ok=True)
    if not resume:
        for old in Path(output_dir).glob("*.jpg"):
            try:
                old.unlink()
            except Exception:
                pass
        for old in img_cache.glob("*.jpg"):
            try:
                old.unlink()
            except Exception:
                pass
    else:
        existing = len(list(img_cache.glob("*.jpg")))
        logger.info("Resume mode: keeping %d existing images", existing)
    
    # Step 2: Build pipeline
    logger.info("Initializing pipeline...")
    pipeline = Pipeline(
        search=SearchModule(),
        label_extractor=LabelExtractor(),
        verifier=FingerprintVerifier(),
        ocr_checker=OCRCrossChecker(),
        quality_filter=QualityFilter(),
        scorer=ConfidenceScorer(),
        cache=ResultCache(":memory:"),
    )
    # Plain search + quality check; skip heavy OCR / strict PASS gate for batch speed.
    pipeline._skip_ocr = True
    pipeline._skip_quality = False
    pipeline._strict_quality = False  # prefer quality pass; fall back to best candidate for coverage
    pipeline._require_pass_verdict = False
    pipeline._strict_match = False
    pipeline._skip_removebg = True  # batch speed; remove.bg was blocking timeouts
    log_time("Initialize pipeline", t)
    t = time.perf_counter()
    
    # Step 3: Create SKUs from pricelist
    def _col(*names: str) -> str:
        for n in names:
            if n in df.columns:
                return n
        raise KeyError(f"None of these columns exist in Excel: {names}")

    code_col = _col("Code", "wine_id", "sku_id", "id")
    name_col = _col("Name", "full_wine_name", "full_name", "wine_name", "name")
    vintage_col = _col("Vintage", "vintage")
    volume_col = next((c for c in ("Volume", "format") if c in df.columns), None)
    type_col = next((c for c in ("Type", "wine_type") if c in df.columns), None)
    country_col = next((c for c in ("Country", "country") if c in df.columns), None)
    grape_col = next((c for c in ("Grape", "grapes") if c in df.columns), None)

    # Prefer sub_region if present (often contains cru/appellation); otherwise fall back.
    region_col = next(
        (c for c in ("sub_region", "Region", "region") if c in df.columns),
        None,
    )
    if region_col is None:
        # Legacy sheet may have newline in header: "Re\ngion"
        region_candidates = [c for c in df.columns if "Re" in c and "ion" in c]
        region_col = region_candidates[0] if region_candidates else None

    skus = []
    for _, row in df.iterrows():
        code = str(row[code_col])
        name = str(row[name_col])
        vintage_str = str(row[vintage_col]) if pd.notna(row[vintage_col]) else None
        
        # Parse vintage (handle NV)
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
        
        producer = _extract_producer(name)
        
        sku = SKU(
            id=code,
            producer=producer,
            appellation=region,
            cru_vineyard=name,  # Use full wine name as cru for better search matching
            vintage=vintage,
            format=str(row[volume_col]) if (volume_col and pd.notna(row.get(volume_col))) else "750ml",
            region=region,
            full_name=name,
            wine_type=str(row[type_col]) if (type_col and pd.notna(row.get(type_col))) else None,
            country=str(row[country_col]) if (country_col and pd.notna(row.get(country_col))) else None,
            grapes=grape,
        )
        skus.append(sku)

    if start_from_sku:
        start_from_sku = start_from_sku.strip()
        start_idx = next(
            (i for i, s in enumerate(skus) if s.id == start_from_sku),
            None,
        )
        if start_idx is None:
            raise ValueError(f"SKU {start_from_sku!r} not found in Excel")
        skus = skus[start_idx:]
        logger.info(
            "Starting from SKU %s — %d wines to process",
            start_from_sku,
            len(skus),
        )
    
    log_time(f"Create {len(skus)} SKU objects", t)
    t = time.perf_counter()
    
    # Step 4: Process each SKU
    results = []
    images_found = []
    images_not_found = []
    
    logger.info("Processing wines through pipeline (reuse image for same wine, different vintage)...")

    import shutil

    sku_results: dict[str, object] = {}
    sku_has_image: dict[str, bool] = {}
    base_wine_images: dict[str, Path] = {}

    for sku in skus:
        safe_id = _safe_sku_filename(sku.id)
        for folder in (img_cache, Path(output_dir)):
            src_path = folder / f"{safe_id}.jpg"
            if src_path.is_file() and src_path.stat().st_size > 0:
                sku_has_image[sku.id] = True
                if sku.id not in images_found:
                    images_found.append(sku.id)
                base_key = _base_wine_key(sku.full_name or "")
                if base_key:
                    base_wine_images[base_key] = src_path
                break

    already_done = len(images_found)
    if already_done:
        logger.info("Skipping %d SKUs that already have images", already_done)

    out_path = Path(output_dir)
    for sku in skus:
        if not sku_has_image.get(sku.id):
            continue
        safe_id = _safe_sku_filename(sku.id)
        for folder in (img_cache, out_path):
            src = folder / f"{safe_id}.jpg"
            if src.is_file() and src.stat().st_size > 0:
                _publish_image(sku.id, src, out_path)
                break

    semaphore = asyncio.Semaphore(1)

    async def _run_sku(idx: int, sku: SKU) -> None:
        async with semaphore:
            safe_id = _safe_sku_filename(sku.id)
            src_path = img_cache / f"{safe_id}.jpg"
            base_key = _base_wine_key(sku.full_name or "")

            if sku_has_image.get(sku.id):
                return

            logger.info(
                f"[{idx+1}/{len(skus)}] {sku.id}: {sku.full_name}"
            )

            if base_key and base_key in base_wine_images:
                donor = base_wine_images[base_key]
                shutil.copy(donor, src_path)
                sku_has_image[sku.id] = True
                images_found.append(sku.id)
                _publish_image(sku.id, src_path, out_path)
                logger.info("  ↻ Reused image from same wine (different vintage)")
                return

            try:
                result = await asyncio.wait_for(
                    pipeline.process_sku(
                        sku, bypass_cache=True, max_candidates=max_candidates
                    ),
                    timeout=sku_timeout_sec,
                )
                sku_results[sku.id] = result

                has_image = src_path.exists() and src_path.stat().st_size > 0
                sku_has_image[sku.id] = has_image
                if has_image:
                    if base_key:
                        base_wine_images[base_key] = src_path
                    images_found.append(sku.id)
                    _publish_image(sku.id, src_path, out_path)
                    logger.info(
                        "  ✓ %s confidence=%.0f%% producer=%.0f%% vintage=%.0f%%",
                        sku.id,
                        result.overall_confidence * 100,
                        result.producer_match * 100,
                        result.vintage_match * 100,
                    )
                else:
                    images_not_found.append(sku.id)
                    logger.warning(
                        "No accurate image for %s (verdict=%s, conf=%.0f%%)",
                        sku.id,
                        result.verdict.value,
                        result.overall_confidence * 100,
                    )
            except asyncio.TimeoutError:
                logger.error(
                    "Timeout (%ds) for %s — skipping",
                    sku_timeout_sec, sku.id,
                )
                images_not_found.append(sku.id)
                sku_has_image[sku.id] = False
            except Exception as e:
                logger.error(f"Error processing {sku.id}: {e}")
                images_not_found.append(sku.id)
                sku_has_image[sku.id] = False

    await asyncio.gather(*[_run_sku(i, sku) for i, sku in enumerate(skus)])

    results = [sku_results.get(sku.id) for sku in skus]
    
    log_time(f"Process {len(skus)} wines through pipeline", t)
    t = time.perf_counter()
    
    # Step 5: Copy images into output dir named by SKU Code only (Column A)
    logger.info("Copying images with SKU Code filenames...")
    renamed_count = 0
    
    for sku in skus:
        safe_id = _safe_sku_filename(sku.id)
        src_path = Path("data/images") / f"{safe_id}.jpg"
        if sku_has_image.get(sku.id, False) and src_path.exists():
            dst_name = f"{safe_id}.jpg"
            dst_path = Path(output_dir) / dst_name
            
            # Copy file
            import shutil
            shutil.copy(src_path, dst_path)
            renamed_count += 1
    
    log_time(f"Rename {renamed_count} images", t)
    t = time.perf_counter()
    
    # Step 6: Create ZIP file
    logger.info(f"Creating ZIP file: {zip_name}")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img_file in Path(output_dir).glob("*.jpg"):
            zf.write(img_file, img_file.name)
    
    log_time(f"Create ZIP file with {renamed_count} images", t)
    
    # Summary
    total_time = time.perf_counter() - total_start
    log_time("TOTAL PROCESSING TIME", total_start)
    
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total wines: {len(skus)}")
    logger.info(f"Images found: {len(images_found)}")
    logger.info(f"Images not found: {len(images_not_found)}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    logger.info(f"Average time per wine: {total_time/len(skus):.2f}s")
    logger.info(f"ZIP file created: {zip_name}")
    
    if images_found:
        logger.info(f"Wines with images: {images_found}")
    if images_not_found:
        logger.info(f"Wines without images: {images_not_found}")
    
    # Write timing log
    timing_file = "pricelist_timing.json"
    with open(timing_file, "w") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "total_wines": len(skus),
            "images_found": len(images_found),
            "images_not_found": images_not_found,
            "timing": TIMING_LOG
        }, f, indent=2)
    logger.info(f"Timing log saved to: {timing_file}")
    
    return results


if __name__ == "__main__":
    # Usage:
    #   python scripts/process_pricelist.py "<excel>" [output_dir] [zip_name] [limit] [--from SKU]
    excel = sys.argv[1] if len(sys.argv) >= 2 else "./use_this.xlsx"
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "pricelist_images"
    zip_name = sys.argv[3] if len(sys.argv) >= 4 else "vinobuzz_images.zip"
    limit = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4].isdigit() else None
    resume = "--fresh" not in sys.argv
    start_from = None
    if "--from" in sys.argv:
        start_from = sys.argv[sys.argv.index("--from") + 1]
    asyncio.run(
        process_pricelist(
            excel,
            output_dir=out_dir,
            zip_name=zip_name,
            limit=limit,
            resume=resume,
            start_from_sku=start_from,
        )
    )
