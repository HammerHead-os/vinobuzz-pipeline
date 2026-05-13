"""Fingerprint Verifier for the Wine Photo Pipeline.

Uses Google Gemini vision to extract structured label data and compares
field-by-field against SKU metadata.  Gemini is available in Hong Kong
(unlike OpenAI), making this suitable for HK deployment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from wine_pipeline.models import FieldScores, Fingerprint, SKU
from wine_pipeline.scoring import compare_field


EXTRACTION_PROMPT = (
    "You are a wine label expert. Examine this wine bottle label image and "
    "extract the following fields as a JSON object:\n"
    '{"producer": "...", "appellation": "...", "cru_vineyard": "...", "vintage": "...", "is_wine_bottle": true/false}\n'
    "Rules:\n"
    "- producer: the winery, domaine, or château name (include 'Domaine', 'Château' prefix if present)\n"
    "- appellation: the AOC/DOC/AVA or regional designation (e.g., 'Clos de Vougeot', 'Saint-Émilion')\n"
    "- cru_vineyard: the cru classification (e.g., 'Grand Cru', 'Premier Cru') or specific vineyard name, or null if none\n"
    "- vintage: the year on the label as a string, or null if non-vintage\n"
    "- is_wine_bottle: true if this image shows a wine bottle with a readable label, false otherwise\n"
    "Important:\n"
    "- Extract exactly what is written on the label, preserving accents (é, è, â, etc.)\n"
    "- If the label shows both a region and a cru, put the region in appellation and cru in cru_vineyard\n"
    "- Return ONLY valid JSON, no markdown fences or extra text."
)


@dataclass
class VerificationResult:
    fingerprint: Fingerprint
    field_scores: FieldScores
    is_nv: bool  # True when SKU is non-vintage


class FingerprintVerifier:
    """Extracts a structured fingerprint from a label image via Google Gemini
    and compares it field-by-field against SKU metadata."""

    def __init__(
        self,
        client: genai.Client | None = None,
        model: str = "gemini-2.5-flash",
    ):
        if client is not None:
            self._client = client
        else:
            import os
            import json
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            project_id = None
            if creds_path:
                try:
                    with open(creds_path) as f:
                        project_id = json.load(f).get("project_id")
                except Exception:
                    pass
            self._client = genai.Client(
                vertexai=True,
                project=project_id or "pragmatic-cat-419908",
                location="us-central1",
            )
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(self, label_image: bytes, sku: SKU) -> VerificationResult:
        """Orchestrate fingerprint extraction and field comparison."""
        fingerprint = await self._extract_fingerprint(label_image)
        field_scores = self._compare_fields(fingerprint, sku)
        is_nv = sku.vintage is None
        return VerificationResult(
            fingerprint=fingerprint,
            field_scores=field_scores,
            is_nv=is_nv,
        )

    # ------------------------------------------------------------------
    # Fingerprint extraction
    # ------------------------------------------------------------------

    async def _extract_fingerprint(self, label_image: bytes) -> Fingerprint:
        """Call Gemini vision API to extract structured label fields."""
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=label_image, mime_type="image/jpeg"),
                EXTRACTION_PROMPT,
            ],
        )

        raw_text = response.text or ""
        return self._parse_fingerprint(raw_text)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fingerprint(raw_text: str) -> Fingerprint:
        """Parse a Gemini JSON response into a Fingerprint, defaulting
        missing or unparseable fields to None."""
        # Strip markdown fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return Fingerprint(
                producer=None, appellation=None,
                cru_vineyard=None, vintage=None,
                is_wine_bottle=False,
            )

        if not isinstance(data, dict):
            return Fingerprint(
                producer=None, appellation=None,
                cru_vineyard=None, vintage=None,
                is_wine_bottle=False,
            )

        def _str_or_none(val: object) -> Optional[str]:
            if val is None:
                return None
            s = str(val).strip()
            return s if s else None

        return Fingerprint(
            producer=_str_or_none(data.get("producer")),
            appellation=_str_or_none(data.get("appellation")),
            cru_vineyard=_str_or_none(data.get("cru_vineyard")),
            vintage=_str_or_none(data.get("vintage")),
            is_wine_bottle=bool(data.get("is_wine_bottle", True)),
        )

    # ------------------------------------------------------------------
    # Field comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_fields(fingerprint: Fingerprint, sku: SKU) -> FieldScores:
        """Compare each fingerprint field against SKU metadata.

        Uses cross-field matching: if a SKU field doesn't match its direct
        counterpart in the fingerprint, check if it appears in any other
        fingerprint field (common with wine labels where cuvée names end up
        in the appellation field or vice versa).
        """
        is_nv = sku.vintage is None

        # Combine all extracted text for cross-field matching
        all_extracted = " ".join(
            f for f in [
                fingerprint.producer,
                fingerprint.appellation,
                fingerprint.cru_vineyard,
                fingerprint.vintage,
            ] if f
        )

        producer_score = compare_field(fingerprint.producer, sku.producer)

        # Appellation: try direct match, then cross-field
        appellation_score = compare_field(fingerprint.appellation, sku.appellation)
        appellation_cross = compare_field(all_extracted, sku.appellation)
        appellation_score = max(appellation_score, appellation_cross)

        # Cru: try direct match, then cross-field, then check if in appellation
        cru_score = compare_field(fingerprint.cru_vineyard, sku.cru_vineyard)
        cru_cross = compare_field(all_extracted, sku.cru_vineyard)
        # Also check if cru appears in the extracted appellation field
        if sku.cru_vineyard and fingerprint.appellation:
            cru_in_appellation = compare_field(fingerprint.appellation, sku.cru_vineyard)
            cru_score = max(cru_score, cru_in_appellation)
        cru_score = max(cru_score, cru_cross)

        if is_nv:
            vintage_score = 0.0
        else:
            vintage_score = compare_field(
                fingerprint.vintage, str(sku.vintage) if sku.vintage else None
            )

        return FieldScores(
            producer_match=producer_score,
            appellation_match=appellation_score,
            cru_match=cru_score,
            vintage_match=vintage_score,
        )
