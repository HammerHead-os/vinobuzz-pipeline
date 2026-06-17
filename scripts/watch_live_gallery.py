#!/usr/bin/env python3
"""Rebuild a live HTML gallery from data/images every few seconds."""

from __future__ import annotations

import html
import subprocess
import sys
import time
from pathlib import Path

REFRESH_SEC = 12
IMAGES_DIR = "data/images"
OUT = "artifacts/live_gallery.html"


def build(repo_root: Path) -> int:
    img_dir = repo_root / IMAGES_DIR
    out_path = repo_root / OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        return 0

    jpgs = sorted(img_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    rel_urls = [Path("..") / IMAGES_DIR / jp.name for jp in jpgs]

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        f'<meta http-equiv="refresh" content="{REFRESH_SEC}" />',
        "<title>Live wine images</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; background: #121212; color: #eee; margin: 16px; }",
        "h1 { font-size: 1.1rem; font-weight: 600; }",
        ".meta { color: #888; font-size: 13px; margin-bottom: 16px; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }",
        ".card { background: #1e1e1e; border-radius: 8px; padding: 8px; }",
        ".card img { width: 100%; height: 380px; object-fit: contain; background: #fff; border-radius: 4px; }",
        ".caption { font-size: 12px; margin-top: 8px; word-break: break-all; color: #aaa; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Live run — {len(jpgs)} images saved</h1>",
        f'<p class="meta">Auto-refreshes every {REFRESH_SEC}s · folder: {html.escape(IMAGES_DIR)}</p>',
        '<div class="grid">',
    ]

    for url, name in zip(rel_urls, [p.name for p in jpgs]):
        safe_alt = html.escape(name, quote=True)
        safe_src = html.escape(url.as_posix(), quote=True)
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
    return len(jpgs)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Watching {repo_root / IMAGES_DIR} → {repo_root / OUT}")
    print(f"Open: http://127.0.0.1:8765/{OUT}")
    while True:
        n = build(repo_root)
        print(f"[{time.strftime('%H:%M:%S')}] {n} images", flush=True)
        time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
