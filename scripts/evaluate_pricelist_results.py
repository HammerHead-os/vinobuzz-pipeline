#!/usr/bin/env python3
"""Evaluate saved pricelist images (no re-search).

Reads an Excel pricelist + an output image folder, scores each image with the
pipeline's verifier + quality filter, and writes a review report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from wine_pipeline.label_extractor import LabelExtractor
from wine_pipeline.models import OCRResult, SKU, Verdict
from wine_pipeline.quality_filter import QualityFilter
from wine_pipeline.scoring import ConfidenceScorer
from wine_pipeline.verifier import FingerprintVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_pricelist")


def _safe_code(code: str) -> str:
    return str(code).replace("/", "_")


def _row_to_sku(row: pd.Series, df: pd.DataFrame) -> SKU:
    code = str(row["Code"])
    name = str(row["Name"])
    vintage_str = str(row["Vintage"]) if pd.notna(row["Vintage"]) else None
    vintage = None
    if vintage_str and vintage_str.upper() != "NV":
        try:
            vintage = int(vintage_str)
        except ValueError:
            pass

    region_col = [c for c in df.columns if "Re" in c and "ion" in c][0]
    region = str(row[region_col]).strip() if pd.notna(row[region_col]) else ""
    grape = str(row["Grape"]) if pd.notna(row["Grape"]) else ""

    producer_parts = name.split()
    if len(producer_parts) >= 2 and producer_parts[1].lower() in (
        "peak", "vineyards", "estate", "winery",
    ):
        producer = f"{producer_parts[0]} {producer_parts[1]}"
    else:
        producer = producer_parts[0] if producer_parts else name

    return SKU(
        id=code,
        producer=producer,
        appellation=region,
        cru_vineyard=name,
        vintage=vintage,
        format=str(row["Volume"]) if pd.notna(row["Volume"]) else "750ml",
        region=region,
        full_name=name,
        wine_type=str(row["Type"]) if pd.notna(row["Type"]) else None,
        country=str(row["Country"]) if pd.notna(row["Country"]) else None,
        grapes=grape,
    )


async def evaluate_image(
    image_path: Path,
    sku: SKU,
    label_extractor: LabelExtractor,
    verifier: FingerprintVerifier,
    quality_filter: QualityFilter,
    scorer: ConfidenceScorer,
) -> dict:
    image_bytes = image_path.read_bytes()
    label_result = label_extractor.extract_label(image_bytes)
    verification = await verifier.verify(label_result.cropped_image, sku)
    quality = await quality_filter.evaluate(image_bytes)

    scored = scorer.score(
        field_scores=verification.field_scores,
        ocr_result=OCRResult(
            raw_text="",
            producer_confirmed=False,
            appellation_confirmed=False,
            cru_confirmed=False,
            vintage_confirmed=False,
            fields_disagreed=[],
        ),
        quality_result=quality,
        label_detected=label_result.label_detected,
        is_nv=verification.is_nv,
    )

    fp = verification.fingerprint
    return {
        "code": sku.id,
        "name": sku.full_name,
        "image_file": image_path.name,
        "verdict": scored.verdict.value,
        "confidence": round(scored.overall_confidence, 3),
        "producer_match": round(scored.producer_match, 3),
        "appellation_match": round(scored.appellation_match, 3),
        "cru_match": round(scored.cru_match, 3),
        "vintage_match": round(scored.vintage_match, 3),
        "image_quality": round(scored.image_quality, 3),
        "quality_passed": quality.passed,
        "quality_reasons": quality.rejection_reasons,
        "label_producer": fp.producer if fp else None,
        "label_appellation": fp.appellation if fp else None,
        "label_cru": fp.cru_vineyard if fp else None,
        "label_vintage": fp.vintage if fp else None,
        "is_wine_bottle": fp.is_wine_bottle if fp else None,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--excel",
        default="./Pricelist_Vinobuzz_260507 Jacky edited.xlsx",
    )
    parser.add_argument("--images-dir", default="pricelist_images_jacky")
    parser.add_argument("--report", default="data/pricelist_eval_jacky.json")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    excel_path = Path(args.excel)
    images_dir = Path(args.images_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(excel_path)
    skus = [_row_to_sku(row, df) for _, row in df.iterrows()]

    label_extractor = LabelExtractor()
    verifier = FingerprintVerifier()
    quality_filter = QualityFilter()
    scorer = ConfidenceScorer()

    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []

    async def _one(sku: SKU) -> None:
        img = images_dir / f"{_safe_code(sku.id)}.jpg"
        if not img.exists():
            rows.append({
                "code": sku.id,
                "name": sku.full_name,
                "image_file": None,
                "verdict": "NO_IMAGE",
                "confidence": 0.0,
                "quality_passed": False,
                "quality_reasons": ["No image file"],
            })
            return
        async with sem:
            try:
                row = await evaluate_image(
                    img, sku, label_extractor, verifier, quality_filter, scorer
                )
                rows.append(row)
                logger.info(
                    "%s → %s (%.0f%%) quality=%s",
                    sku.id, row["verdict"], row["confidence"] * 100, row["quality_passed"],
                )
            except Exception as exc:
                logger.error("Failed %s: %s", sku.id, exc)
                rows.append({
                    "code": sku.id,
                    "name": sku.full_name,
                    "image_file": img.name,
                    "verdict": "ERROR",
                    "confidence": 0.0,
                    "quality_passed": False,
                    "quality_reasons": [str(exc)],
                })

    await asyncio.gather(*[_one(sku) for sku in skus])

    rows.sort(key=lambda r: r["code"])
    report_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Summary
    total = len(skus)
    has_image = sum(1 for r in rows if r.get("image_file"))
    pass_n = sum(1 for r in rows if r.get("verdict") == "PASS")
    quar_n = sum(1 for r in rows if r.get("verdict") == "QUARANTINE")
    reject_n = sum(1 for r in rows if r.get("verdict") == "REJECT")
    no_img = sum(1 for r in rows if r.get("verdict") == "NO_IMAGE")
    qual_fail = sum(1 for r in rows if r.get("image_file") and not r.get("quality_passed"))

    print("\n=== PRICELIST EVALUATION SUMMARY ===")
    print(f"Excel: {excel_path}")
    print(f"Images dir: {images_dir}")
    print(f"Total SKUs: {total}")
    print(f"With image file: {has_image}")
    print(f"PASS: {pass_n}  QUARANTINE: {quar_n}  REJECT: {reject_n}  NO_IMAGE: {no_img}")
    print(f"Quality filter failed (among images): {qual_fail}")
    print(f"Report: {report_path}")

    # Low confidence / quality failures for manual review
    review = [
        r for r in rows
        if r.get("verdict") in ("REJECT", "QUARANTINE", "NO_IMAGE", "ERROR")
        or not r.get("quality_passed", True)
    ]
    review_path = report_path.with_name("pricelist_eval_jacky_review.json")
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    print(f"Needs review ({len(review)}): {review_path}")


if __name__ == "__main__":
    asyncio.run(main())
