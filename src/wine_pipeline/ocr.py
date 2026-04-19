"""OCR Cross-Checker for the Wine Photo Pipeline.

Uses Google Cloud Vision OCR to independently extract text from label images
and cross-reference against GPT-4o fingerprint results.
"""

from __future__ import annotations

from google.cloud import vision

from rapidfuzz.fuzz import partial_ratio

from wine_pipeline.models import Fingerprint, OCRResult


# Minimum fuzzy match ratio to consider a token confirmed
_MATCH_THRESHOLD = 75.0


class OCRCrossChecker:
    """Sends label images to Google Cloud Vision OCR and cross-references
    extracted text against a GPT-4o Fingerprint."""

    def __init__(self, client: vision.ImageAnnotatorClient | None = None):
        self._client = client or vision.ImageAnnotatorClient(
            client_options={"api_endpoint": "eu-vision.googleapis.com"}
        )

    async def check(self, label_image: bytes, fingerprint: Fingerprint) -> OCRResult:
        """Send image to Google Cloud Vision, extract text, cross-reference."""
        raw_text = self._extract_text(label_image)
        return self._cross_reference(raw_text, fingerprint)

    def _extract_text(self, label_image: bytes) -> str:
        """Call Google Cloud Vision OCR and return raw extracted text."""
        image = vision.Image(content=label_image)
        response = self._client.text_detection(image=image)
        annotations = response.text_annotations
        if annotations:
            return annotations[0].description
        return ""

    @staticmethod
    def _cross_reference(raw_text: str, fingerprint: Fingerprint) -> OCRResult:
        """Parse raw OCR text and cross-reference against fingerprint fields."""
        fields_disagreed: list[str] = []

        producer_confirmed = _token_found(raw_text, fingerprint.producer)
        appellation_confirmed = _token_found(raw_text, fingerprint.appellation)
        cru_confirmed = _token_found(raw_text, fingerprint.cru_vineyard)
        vintage_confirmed = _token_found(raw_text, fingerprint.vintage)

        # A field is "disagreed" when the fingerprint has a value but OCR
        # does not confirm it.
        if fingerprint.producer and not producer_confirmed:
            fields_disagreed.append("producer")
        if fingerprint.appellation and not appellation_confirmed:
            fields_disagreed.append("appellation")
        if fingerprint.cru_vineyard and not cru_confirmed:
            fields_disagreed.append("cru")
        if fingerprint.vintage and not vintage_confirmed:
            fields_disagreed.append("vintage")

        return OCRResult(
            raw_text=raw_text,
            producer_confirmed=producer_confirmed,
            appellation_confirmed=appellation_confirmed,
            cru_confirmed=cru_confirmed,
            vintage_confirmed=vintage_confirmed,
            fields_disagreed=fields_disagreed,
        )


def _token_found(raw_text: str, token: str | None) -> bool:
    """Check whether *token* appears in *raw_text* using fuzzy matching.

    Returns False when the token is None or empty.
    Uses rapidfuzz partial_ratio for tolerance against OCR noise.
    """
    if not token or not raw_text:
        return False
    score = partial_ratio(token.lower(), raw_text.lower())
    return score >= _MATCH_THRESHOLD
