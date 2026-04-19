"""Unit tests for the Label Extractor."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from wine_pipeline.label_extractor import LabelExtractor


def _make_image_bytes(img: np.ndarray) -> bytes:
    """Encode a cv2 image to PNG bytes."""
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_detects_and_crops_rectangle():
    """A synthetic image with a white rectangle on black background should be detected."""
    # 400x600 black image with a 200x100 white rectangle (label-like)
    img = np.zeros((600, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 200), (300, 400), (255, 255, 255), -1)

    extractor = LabelExtractor()
    result = extractor.extract_label(_make_image_bytes(img))

    assert result.label_detected is True
    # Decode the cropped image and verify it's smaller than the original
    cropped = cv2.imdecode(np.frombuffer(result.cropped_image, np.uint8), cv2.IMREAD_COLOR)
    assert cropped is not None
    assert cropped.shape[0] < img.shape[0] or cropped.shape[1] < img.shape[1]


def test_uniform_image_returns_full_with_no_label():
    """A uniform color image has no contours — should return full image with label_detected=False."""
    img = np.full((400, 300, 3), 128, dtype=np.uint8)

    extractor = LabelExtractor()
    result = extractor.extract_label(_make_image_bytes(img))

    assert result.label_detected is False
    # The returned image should be decodable and same dimensions as input
    decoded = cv2.imdecode(np.frombuffer(result.cropped_image, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == img.shape[:2]
