#!/usr/bin/env python3
"""Build a minimal static HTML page listing all JPGs from a folder."""

from __future__ import annotations

import argparse
import html
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simple image gallery HTML.")
    parser.add_argument(
        "--images-dir",
        default="pricelist_images_jacky",
        help="Folder containing .jpg images (default: pricelist_images_jacky)",
    )
    parser.add_argument(
        "--out",
        default="image_gallery.html",
        help="Output HTML path relative to repo root (default: image_gallery.html)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    img_dir = (repo_root / args.images_dir).resolve()
    out_path = (repo_root / args.out).resolve()

    if not img_dir.is_dir():
        raise SystemExit(f"Not a directory: {img_dir}")

    jpgs = sorted(img_dir.glob("*.jpg"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_parent = out_path.parent.resolve()
    rel_urls = [
        jp.resolve().relative_to(out_parent).as_posix() for jp in jpgs
    ]
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        "<title>Wine images</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; background: #121212; color: #eee; margin: 16px; }",
        "h1 { font-size: 1.1rem; font-weight: 600; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }",
        ".card { background: #1e1e1e; border-radius: 8px; padding: 8px; }",
        ".card img { width: 100%; height: 380px; object-fit: contain; background: #0a0a0a; border-radius: 4px; }",
        ".caption { font-size: 12px; margin-top: 8px; word-break: break-all; color: #aaa; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(img_dir.name)} — {len(jpgs)} images</h1>",
        '<div class="grid">',
    ]

    for url, name in zip(rel_urls, [p.name for p in jpgs]):
        safe_alt = html.escape(name, quote=True)
        safe_src = html.escape(url, quote=True)
        parts.extend(
            [
                '<div class="card">',
                f'<img loading="lazy" src="{safe_src}" alt="{safe_alt}" />',
                f'<div class="caption">{safe_alt}</div>',
                "</div>",
            ]
        )

    parts.extend(["</div>", "</body>", "</html>"])
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {len(jpgs)} images → {out_path}")


if __name__ == "__main__":
    main()
