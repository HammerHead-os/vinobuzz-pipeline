#!/usr/bin/env python3
"""Streamlit review UI for pricelist image sourcing results."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXCEL = REPO_ROOT / "08_06_2026.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "Goldgate_wine"


def _safe_sku_filename(sku_id: str) -> str:
    return sku_id.replace("/", "_").replace("\\", "_")


def _load_pricelist(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    rename = {}
    if "wine_id" in df.columns:
        rename["wine_id"] = "sku_code"
    elif "Code" in df.columns:
        rename["Code"] = "sku_code"
    if "full_wine_name" in df.columns:
        rename["full_wine_name"] = "wine_name"
    elif "Name" in df.columns:
        rename["Name"] = "wine_name"
    if "vendor_sku_id" in df.columns:
        rename["vendor_sku_id"] = "vendor_sku"
    if "vintage" in df.columns:
        rename["vintage"] = "vintage"
    df = df.rename(columns=rename)
    required = {"sku_code", "wine_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing columns: {sorted(missing)}")
    return df


def _resolve_image_path(sku_code: str, output_dir: Path) -> Path | None:
    safe = _safe_sku_filename(str(sku_code))
    path = output_dir / f"{safe}.jpg"
    if path.is_file() and path.stat().st_size > 0:
        return path
    return None


def main() -> None:
    st.set_page_config(
        page_title="VinoBuzz Pricelist Review",
        page_icon="🍷",
        layout="wide",
    )

    st.title("Pricelist image review")
    st.caption("Review sourced bottle images against the Excel pricelist.")

    with st.sidebar:
        st.header("Data sources")
        excel_path = Path(
            st.text_input("Excel file", value=str(DEFAULT_EXCEL))
        )
        output_dir = Path(
            st.text_input("Images folder", value=str(DEFAULT_OUTPUT_DIR))
        )
        if st.button("Refresh"):
            st.rerun()

    if not excel_path.is_file():
        st.error(f"Excel not found: {excel_path}")
        return

    try:
        df = _load_pricelist(excel_path)
    except Exception as exc:
        st.error(str(exc))
        return

    rows = []
    for _, row in df.iterrows():
        sku = str(row["sku_code"])
        img_path = _resolve_image_path(sku, output_dir)
        rows.append(
            {
                "sku_code": sku,
                "wine_name": str(row.get("wine_name", "")),
                "vendor_sku": str(row.get("vendor_sku", "")) if pd.notna(row.get("vendor_sku")) else "",
                "vintage": str(row.get("vintage", "")) if pd.notna(row.get("vintage")) else "",
                "has_image": img_path is not None,
                "image_path": str(img_path) if img_path else "",
            }
        )

    review = pd.DataFrame(rows)
    found = int(review["has_image"].sum())
    total = len(review)
    missing = total - found

    c1, c2, c3 = st.columns(3)
    c1.metric("Total SKUs", total)
    c2.metric("Images found", found)
    c3.metric("Missing", missing)

    st.progress(found / total if total else 0.0, text=f"{found}/{total} complete")

    filter_mode = st.radio(
        "Show",
        ["All", "Found only", "Missing only"],
        horizontal=True,
    )
    query = st.text_input("Search SKU or wine name").strip().lower()

    filtered = review.copy()
    if filter_mode == "Found only":
        filtered = filtered[filtered["has_image"]]
    elif filter_mode == "Missing only":
        filtered = filtered[~filtered["has_image"]]
    if query:
        mask = (
            filtered["sku_code"].str.lower().str.contains(re.escape(query), regex=True)
            | filtered["wine_name"].str.lower().str.contains(re.escape(query), regex=True)
            | filtered["vendor_sku"].str.lower().str.contains(re.escape(query), regex=True)
        )
        filtered = filtered[mask]

    st.subheader(f"Showing {len(filtered)} of {total}")

    cols_per_row = st.slider("Columns", min_value=2, max_value=5, value=3)
    for start in range(0, len(filtered), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (_, item) in zip(cols, filtered.iloc[start : start + cols_per_row].iterrows()):
            with col:
                status = "✅" if item["has_image"] else "❌"
                st.markdown(f"**{status} `{item['sku_code']}`**")
                if item["has_image"]:
                    st.image(item["image_path"], use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:280px;background:#1a1a1a;border-radius:8px;'
                        'display:flex;align-items:center;justify-content:center;color:#888;">'
                        "No image yet</div>",
                        unsafe_allow_html=True,
                    )
                st.caption(item["wine_name"])
                meta = []
                if item["vintage"]:
                    meta.append(f"Vintage: {item['vintage']}")
                if item["vendor_sku"]:
                    meta.append(f"Vendor: {item['vendor_sku']}")
                if meta:
                    st.caption(" · ".join(meta))


if __name__ == "__main__":
    main()
