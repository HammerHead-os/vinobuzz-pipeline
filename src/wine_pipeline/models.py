"""Core data models for the Wine Photo Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Verdict(Enum):
    PASS = "PASS"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


@dataclass
class SKU:
    id: str
    producer: str
    appellation: str
    cru_vineyard: Optional[str]
    vintage: Optional[int]  # None for NV wines
    format: str  # e.g., "750ml"
    region: str
    # Extended fields from production data
    full_name: Optional[str] = None
    wine_type: Optional[str] = None  # Red, White, Sparkling, etc.
    country: Optional[str] = None
    grapes: Optional[str] = None
    reference_image_url: Optional[str] = None  # Existing image URL if available
    rating: Optional[float] = None


@dataclass
class CandidateImage:
    url: str
    source: str  # "serper", "brave", "google", "bing", "duckduckgo", etc.
    raw_image: Optional[bytes] = None


@dataclass
class LabelExtractionResult:
    cropped_image: bytes
    label_detected: bool  # False if full image was returned as fallback


@dataclass
class Fingerprint:
    producer: Optional[str]
    appellation: Optional[str]
    cru_vineyard: Optional[str]
    vintage: Optional[str]
    is_wine_bottle: bool = True  # True if image shows a wine bottle


@dataclass
class FieldScores:
    producer_match: float  # 0.0 to 1.0
    appellation_match: float
    cru_match: float
    vintage_match: float  # 0.0 for NV wines (weight redistributed)


@dataclass
class OCRResult:
    raw_text: str
    producer_confirmed: bool
    appellation_confirmed: bool
    cru_confirmed: bool
    vintage_confirmed: bool
    fields_disagreed: list = field(default_factory=list)  # field names where OCR contradicts GPT-4o


@dataclass
class QualityResult:
    passed: bool
    image_quality_score: float  # 0.0 to 1.0
    rejection_reasons: list = field(default_factory=list)  # empty if passed


@dataclass
class ScoredResult:
    sku_id: str
    image_url: Optional[str]  # None if "No Image"
    producer_match: float
    appellation_match: float
    cru_match: float
    vintage_match: float
    image_quality: float
    overall_confidence: float
    verdict: Verdict
    fingerprint: Optional[Fingerprint]
    rejection_reasons: list = field(default_factory=list)

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "sku_id": self.sku_id,
            "image_url": self.image_url,
            "producer_match": self.producer_match,
            "appellation_match": self.appellation_match,
            "cru_match": self.cru_match,
            "vintage_match": self.vintage_match,
            "image_quality": self.image_quality,
            "overall_confidence": self.overall_confidence,
            "verdict": self.verdict.value,
            "fingerprint": {
                "producer": self.fingerprint.producer,
                "appellation": self.fingerprint.appellation,
                "cru_vineyard": self.fingerprint.cru_vineyard,
                "vintage": self.fingerprint.vintage,
            } if self.fingerprint is not None else None,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_json(cls, data: dict) -> "ScoredResult":
        """Deserialize from JSON-compatible dict."""
        fp_data = data.get("fingerprint")
        fingerprint = Fingerprint(
            producer=fp_data["producer"],
            appellation=fp_data["appellation"],
            cru_vineyard=fp_data["cru_vineyard"],
            vintage=fp_data["vintage"],
        ) if fp_data is not None else None

        return cls(
            sku_id=data["sku_id"],
            image_url=data.get("image_url"),
            producer_match=data["producer_match"],
            appellation_match=data["appellation_match"],
            cru_match=data["cru_match"],
            vintage_match=data["vintage_match"],
            image_quality=data["image_quality"],
            overall_confidence=data["overall_confidence"],
            verdict=Verdict(data["verdict"]),
            fingerprint=fingerprint,
            rejection_reasons=list(data.get("rejection_reasons", [])),
        )
