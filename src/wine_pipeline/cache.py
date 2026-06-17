"""SQLite-based result caching for the Wine Photo Pipeline.

Stores and retrieves ScoredResult objects to avoid redundant API calls.
Requirements: 7.1, 7.2, 7.3
"""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

from wine_pipeline.models import Fingerprint, ScoredResult, Verdict


class ResultCache:
    """SQLite cache for pipeline results."""

    def __init__(self, db_path: str = "pipeline_cache.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_results (
                sku_id TEXT PRIMARY KEY,
                image_url TEXT,
                producer_match REAL,
                appellation_match REAL,
                cru_match REAL,
                vintage_match REAL,
                image_quality REAL,
                overall_confidence REAL,
                verdict TEXT,
                fingerprint_json TEXT,
                rejection_reasons_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def get(self, sku_id: str) -> Optional[ScoredResult]:
        """Retrieve a cached result by SKU ID. Returns None on cache miss."""
        row = self._conn.execute(
            "SELECT * FROM pipeline_results WHERE sku_id = ?", (sku_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_result(row)

    def put(self, sku_id: str, result: ScoredResult) -> None:
        """Store or update a result in the cache."""
        fp_json = self._serialize_fingerprint(result.fingerprint)
        rr_json = json.dumps(result.rejection_reasons)

        self._conn.execute(
            """
            INSERT INTO pipeline_results
                (sku_id, image_url, producer_match, appellation_match,
                 cru_match, vintage_match, image_quality, overall_confidence,
                 verdict, fingerprint_json, rejection_reasons_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(sku_id) DO UPDATE SET
                image_url = excluded.image_url,
                producer_match = excluded.producer_match,
                appellation_match = excluded.appellation_match,
                cru_match = excluded.cru_match,
                vintage_match = excluded.vintage_match,
                image_quality = excluded.image_quality,
                overall_confidence = excluded.overall_confidence,
                verdict = excluded.verdict,
                fingerprint_json = excluded.fingerprint_json,
                rejection_reasons_json = excluded.rejection_reasons_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                sku_id,
                result.image_url,
                result.producer_match,
                result.appellation_match,
                result.cru_match,
                result.vintage_match,
                result.image_quality,
                result.overall_confidence,
                result.verdict.value,
                fp_json,
                rr_json,
            ),
        )
        self._conn.commit()

    def invalidate(self, sku_id: str) -> None:
        """Remove a cached result for the given SKU ID."""
        self._conn.execute(
            "DELETE FROM pipeline_results WHERE sku_id = ?", (sku_id,)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_fingerprint(fp: Optional[Fingerprint]) -> Optional[str]:
        if fp is None:
            return None
        return json.dumps({
            "producer": fp.producer,
            "appellation": fp.appellation,
            "cru_vineyard": fp.cru_vineyard,
            "vintage": fp.vintage,
        })

    @staticmethod
    def _deserialize_fingerprint(fp_json: Optional[str]) -> Optional[Fingerprint]:
        if fp_json is None:
            return None
        data = json.loads(fp_json)
        return Fingerprint(
            producer=data.get("producer"),
            appellation=data.get("appellation"),
            cru_vineyard=data.get("cru_vineyard"),
            vintage=data.get("vintage"),
        )

    def _row_to_result(self, row: sqlite3.Row) -> ScoredResult:
        rr_json = row["rejection_reasons_json"]
        rejection_reasons = json.loads(rr_json) if rr_json else []

        return ScoredResult(
            sku_id=row["sku_id"],
            image_url=row["image_url"],
            producer_match=row["producer_match"],
            appellation_match=row["appellation_match"],
            cru_match=row["cru_match"],
            vintage_match=row["vintage_match"],
            image_quality=row["image_quality"],
            overall_confidence=row["overall_confidence"],
            verdict=Verdict(row["verdict"]),
            fingerprint=self._deserialize_fingerprint(row["fingerprint_json"]),
            rejection_reasons=rejection_reasons,
        )
