"""Quality Filter for the Wine Photo Pipeline.

Uses Google Gemini vision to evaluate candidate wine images against
VinoBuzz photo standards: no watermarks, no blur/glare, single
upright bottle, white/grey background, no lifestyle props, not
AI-generated.

Gemini is available in Hong Kong (unlike OpenAI).
"""

from __future__ import annotations

import json

from google import genai
from google.genai import types

from wine_pipeline.models import QualityResult


QUALITY_PROMPT = (
    "You are a wine product photo quality inspector. Examine this image and "
    "evaluate it against the following criteria. For each criterion, answer "
    '"pass" or "fail".\n\n'
    "1. watermarks: The image must NOT contain visible watermarks.\n"
    "2. blur_glare: The image must NOT exhibit blur or glare that obscures label text.\n"
    "3. single_upright_bottle: The image must show a single upright bottle.\n"
    "4. background: The background must be white or neutral grey.\n"
    "5. no_lifestyle_props: The image must NOT contain lifestyle props, table settings, "
    "or non-product elements.\n"
    "6. not_ai_generated: The image must NOT appear to be AI-generated.\n\n"
    "Return ONLY valid JSON with this structure:\n"
    "{\n"
    '  "watermarks": "pass" or "fail",\n'
    '  "blur_glare": "pass" or "fail",\n'
    '  "single_upright_bottle": "pass" or "fail",\n'
    '  "background": "pass" or "fail",\n'
    '  "no_lifestyle_props": "pass" or "fail",\n'
    '  "not_ai_generated": "pass" or "fail"\n'
    "}\n"
    "No markdown fences or extra text."
)

# Human-readable descriptions for each criterion failure.
_REJECTION_DESCRIPTIONS: dict[str, str] = {
    "watermarks": "Image contains visible watermarks",
    "blur_glare": "Image exhibits blur or glare that obscures label text",
    "single_upright_bottle": "Image does not show a single upright bottle",
    "background": "Background is not white or neutral grey",
    "no_lifestyle_props": "Image contains lifestyle props or non-product elements",
    "not_ai_generated": "Image appears to be AI-generated",
}

_ALL_CRITERIA = list(_REJECTION_DESCRIPTIONS.keys())


class QualityFilter:
    """Evaluates candidate wine images against VinoBuzz photo standards
    using Google Gemini vision."""

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

    async def evaluate(self, image: bytes) -> QualityResult:
        """Send image to Gemini and parse quality evaluation."""
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image, mime_type="image/jpeg"),
                QUALITY_PROMPT,
            ],
        )

        raw_text = response.text or ""
        return self._parse_response(raw_text)

    @staticmethod
    def _parse_response(raw_text: str) -> QualityResult:
        """Parse Gemini JSON response into a QualityResult."""
        # Strip markdown fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return QualityResult(
                passed=False,
                image_quality_score=0.0,
                rejection_reasons=["Unable to parse quality evaluation response"],
            )

        if not isinstance(data, dict):
            return QualityResult(
                passed=False,
                image_quality_score=0.0,
                rejection_reasons=["Unable to parse quality evaluation response"],
            )

        rejection_reasons: list[str] = []
        passed_count = 0

        for criterion in _ALL_CRITERIA:
            value = str(data.get(criterion, "fail")).strip().lower()
            if value == "pass":
                passed_count += 1
            else:
                rejection_reasons.append(_REJECTION_DESCRIPTIONS[criterion])

        total = len(_ALL_CRITERIA)
        image_quality_score = passed_count / total if total > 0 else 0.0
        passed = len(rejection_reasons) == 0

        return QualityResult(
            passed=passed,
            image_quality_score=round(image_quality_score, 4),
            rejection_reasons=rejection_reasons,
        )
