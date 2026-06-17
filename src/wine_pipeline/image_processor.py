"""Post-process selected wine photos: remove.bg cutout on white canvas, full bottle preserved."""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

REMOVEBG_API_KEY = os.getenv("REMOVEBG_API_KEY", "") or os.getenv("REMOVE_BG_API_KEY", "")
REMOVEBG_URL = "https://api.remove.bg/v1.0/removebg"

CANVAS_W = 800
CANVAS_H = 1200
JPEG_QUALITY = 94
MIN_EDGE_MARGIN = 0.02  # cap and base must sit inside frame, not flush to edge


def is_vivino_product_bottle(url: str) -> bool:
    """True for Vivino full-bottle product shots (_pb_ URLs)."""
    lower = url.lower()
    return "images.vivino.com" in lower and "_pb_" in lower


class ProductImageProcessor:
    """remove.bg background removal, then center on white — never crop the bottle."""

    def prepare_for_delivery(self, image_bytes: bytes) -> Optional[bytes]:
        if not REMOVEBG_API_KEY.strip():
            logger.error("REMOVEBG_API_KEY is not set — add it to .env in the project root")
            return None

        if not self.has_full_body_margins(image_bytes):
            logger.warning("Skipping image: bottle is cropped (cap or base touches frame edge)")
            return None

        cutout = self._remove_background(image_bytes)
        if cutout is None:
            return None

        img = self._decode(cutout)
        if img is None:
            return None

        if img.shape[0] < 80 or img.shape[1] < 40:
            return None

        result = self._center_on_white_canvas(img)
        if result and not self.has_full_body_margins(result):
            logger.warning("remove.bg output lost full-body margins")
            return None
        return result

    @classmethod
    def has_full_body_margins(
        cls, image_bytes: bytes, min_margin: float = MIN_EDGE_MARGIN
    ) -> bool:
        """True when foreground bottle has clear space above cap and below base."""
        img = cls._decode(image_bytes)
        if img is None:
            return False
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            bgr = img[:, :, :3].astype(np.float32)
            alpha = img[:, :, 3].astype(np.float32) / 255.0
            white = np.full_like(bgr, 255.0)
            a = alpha[:, :, np.newaxis]
            img = (bgr * a + white * (1.0 - a)).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.where(gray < 245, 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return False

        _x, y, bw, bh = cv2.boundingRect(coords)
        h, w = img.shape[:2]
        if bh < h * 0.45:
            return False
        if bw <= 0 or bh / bw < 1.4:
            return False

        top_margin = y / h
        bottom_margin = (h - y - bh) / h
        return top_margin >= min_margin and bottom_margin >= min_margin

    def _remove_background(self, image_bytes: bytes) -> Optional[bytes]:
        try:
            with httpx.Client(timeout=90.0) as client:
                resp = client.post(
                    REMOVEBG_URL,
                    files={
                        "image_file": (
                            "image.jpg",
                            image_bytes,
                            "application/octet-stream",
                        )
                    },
                    data={
                        "size": "auto",
                        "type": "auto",
                        "crop": "false",
                        "scale": "100%",
                        "position": "original",
                        "bg_color": "FFFFFF",
                        "format": "png",
                    },
                    headers={"X-Api-Key": REMOVEBG_API_KEY.strip()},
                )
        except httpx.HTTPError as exc:
            logger.warning("remove.bg request failed: %s", exc)
            return None

        if resp.status_code != 200:
            detail = resp.text[:400].replace("\n", " ")
            logger.warning("remove.bg HTTP %s: %s", resp.status_code, detail)
            return None

        if not resp.content:
            logger.warning("remove.bg returned empty body")
            return None

        return resp.content

    @staticmethod
    def _decode(image_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def _center_on_white_canvas(img: np.ndarray) -> Optional[bytes]:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            bgr = img[:, :, :3].astype(np.float32)
            alpha = img[:, :, 3].astype(np.float32) / 255.0
            white = np.full_like(bgr, 255.0)
            a = alpha[:, :, np.newaxis]
            img = (bgr * a + white * (1.0 - a)).astype(np.uint8)
        elif img.shape[2] != 3:
            return None

        rh, rw = img.shape[:2]
        scale = min((CANVAS_W * 0.90) / rw, (CANVAS_H * 0.94) / rh)
        new_w = max(1, int(rw * scale))
        new_h = max(1, int(rh * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)
        x0 = (CANVAS_W - new_w) // 2
        y0 = (CANVAS_H - new_h) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return None
        return buf.tobytes()
