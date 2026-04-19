"""Label region extraction from wine bottle images using OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from .models import LabelExtractionResult

# Minimum contour area as a fraction of total image area
_MIN_AREA_RATIO = 0.02
# Maximum contour area as a fraction of total image area
_MAX_AREA_RATIO = 0.90
# Aspect ratio bounds for label-like rectangles (width / height)
_MIN_ASPECT_RATIO = 0.3
_MAX_ASPECT_RATIO = 5.0


class LabelExtractor:
    """Detects and crops the label region from a wine bottle image."""

    def extract_label(self, image: bytes) -> LabelExtractionResult:
        """Detect and crop label region using OpenCV contour detection.

        Algorithm:
        1. Decode image bytes into a cv2 image
        2. Convert to grayscale
        3. Apply Gaussian blur and adaptive thresholding
        4. Find contours, filter by area and aspect ratio
        5. Crop the largest qualifying contour region
        6. If no qualifying contour, return full image with label_detected=False

        Args:
            image: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            LabelExtractionResult with cropped label or full image fallback.
        """
        img_array = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            # Undecodable image — return raw bytes as fallback
            return LabelExtractionResult(cropped_image=image, label_detected=False)

        total_area = img.shape[0] * img.shape[1]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < _MIN_AREA_RATIO * total_area:
                continue
            if area > _MAX_AREA_RATIO * total_area:
                continue

            aspect = w / h if h > 0 else 0
            if aspect < _MIN_ASPECT_RATIO or aspect > _MAX_ASPECT_RATIO:
                continue

            if area > best_area:
                best_area = area
                best_contour = (x, y, w, h)

        if best_contour is None:
            # No qualifying contour — return full image
            _, encoded = cv2.imencode(".png", img)
            return LabelExtractionResult(
                cropped_image=encoded.tobytes(), label_detected=False
            )

        x, y, w, h = best_contour
        cropped = img[y : y + h, x : x + w]
        _, encoded = cv2.imencode(".png", cropped)

        return LabelExtractionResult(cropped_image=encoded.tobytes(), label_detected=True)
