"""Unit tests for the SQLite cache layer.

Tests cover:
- Result serialization and storage (Requirement 9.1)
- Cache retrieval and miss handling (Requirement 9.2)
- Cache invalidation (Requirements 9.1, 9.2, 9.3, 9.4)
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wine_pipeline.cache import ResultCache
from wine_pipeline.models import Fingerprint, ScoredResult, Verdict


class TestSQLiteStorage:
    """Tests for result serialization and database storage (Requirement 9.1)."""

    def test_store_result_creates_database_entry(self):
        """Test that storing a result creates a row in the database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                result = ScoredResult(
                    sku_id="test-sku-001",
                    image_url="https://example.com/wine.jpg",
                    producer_match=0.95,
                    appellation_match=0.88,
                    cru_match=0.92,
                    vintage_match=1.0,
                    image_quality=0.85,
                    overall_confidence=0.90,
                    verdict=Verdict.PASS,
                    fingerprint=Fingerprint(
                        producer="Château Margaux",
                        appellation="Margaux",
                        cru_vineyard="Premier Grand Cru",
                        vintage="2015"
                    ),
                    rejection_reasons=[]
                )
                
                cache.put(result.sku_id, result)
                
                # Verify database row exists
                row = cache._conn.execute(
                    "SELECT * FROM pipeline_results WHERE sku_id = ?",
                    (result.sku_id,)
                ).fetchone()
                
                assert row is not None
                assert row["sku_id"] == result.sku_id
                assert row["image_url"] == result.image_url
                assert row["producer_match"] == result.producer_match
                assert row["verdict"] == "PASS"
            finally:
                cache.close()

    def test_store_result_serializes_fingerprint(self):
        """Test that fingerprint is correctly serialized to JSON."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                fingerprint = Fingerprint(
                    producer="Test Producer",
                    appellation="Test Appellation",
                    cru_vineyard="Test Cru",
                    vintage="2020"
                )
                result = ScoredResult(
                    sku_id="test-sku-002",
                    image_url="https://example.com/wine2.jpg",
                    producer_match=0.8,
                    appellation_match=0.7,
                    cru_match=0.9,
                    vintage_match=1.0,
                    image_quality=0.75,
                    overall_confidence=0.82,
                    verdict=Verdict.QUARANTINE,
                    fingerprint=fingerprint,
                    rejection_reasons=["Low quality"]
                )
                
                cache.put(result.sku_id, result)
                
                row = cache._conn.execute(
                    "SELECT fingerprint_json FROM pipeline_results WHERE sku_id = ?",
                    (result.sku_id,)
                ).fetchone()
                
                assert row is not None
                fp_data = json.loads(row["fingerprint_json"])
                assert fp_data["producer"] == "Test Producer"
                assert fp_data["appellation"] == "Test Appellation"
                assert fp_data["cru_vineyard"] == "Test Cru"
                assert fp_data["vintage"] == "2020"
            finally:
                cache.close()

    def test_store_result_serializes_rejection_reasons(self):
        """Test that rejection reasons are serialized to JSON."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                result = ScoredResult(
                    sku_id="test-sku-003",
                    image_url=None,
                    producer_match=0.0,
                    appellation_match=0.0,
                    cru_match=0.0,
                    vintage_match=0.0,
                    image_quality=0.0,
                    overall_confidence=0.0,
                    verdict=Verdict.REJECT,
                    fingerprint=None,
                    rejection_reasons=["No Image", "Low Quality", "Watermark Detected"]
                )
                
                cache.put(result.sku_id, result)
                
                row = cache._conn.execute(
                    "SELECT rejection_reasons_json FROM pipeline_results WHERE sku_id = ?",
                    (result.sku_id,)
                ).fetchone()
                
                assert row is not None
                reasons = json.loads(row["rejection_reasons_json"])
                assert reasons == ["No Image", "Low Quality", "Watermark Detected"]
            finally:
                cache.close()

    def test_store_result_with_null_fingerprint(self):
        """Test storing result with None fingerprint."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                result = ScoredResult(
                    sku_id="test-sku-004",
                    image_url=None,
                    producer_match=0.0,
                    appellation_match=0.0,
                    cru_match=0.0,
                    vintage_match=0.0,
                    image_quality=0.0,
                    overall_confidence=0.0,
                    verdict=Verdict.REJECT,
                    fingerprint=None,
                    rejection_reasons=["No Image"]
                )
                
                cache.put(result.sku_id, result)
                
                row = cache._conn.execute(
                    "SELECT fingerprint_json FROM pipeline_results WHERE sku_id = ?",
                    (result.sku_id,)
                ).fetchone()
                
                assert row is not None
                assert row["fingerprint_json"] is None
            finally:
                cache.close()

    def test_update_existing_result(self):
        """Test that storing an existing SKU ID updates the result."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                # Store initial result
                result1 = ScoredResult(
                    sku_id="test-sku-005",
                    image_url="https://example.com/wine1.jpg",
                    producer_match=0.5,
                    appellation_match=0.5,
                    cru_match=0.5,
                    vintage_match=0.5,
                    image_quality=0.5,
                    overall_confidence=0.5,
                    verdict=Verdict.QUARANTINE,
                    fingerprint=None,
                    rejection_reasons=[]
                )
                cache.put(result1.sku_id, result1)
                
                # Store updated result
                result2 = ScoredResult(
                    sku_id="test-sku-005",
                    image_url="https://example.com/wine2.jpg",
                    producer_match=0.95,
                    appellation_match=0.95,
                    cru_match=0.95,
                    vintage_match=0.95,
                    image_quality=0.95,
                    overall_confidence=0.95,
                    verdict=Verdict.PASS,
                    fingerprint=Fingerprint(
                        producer="Updated Producer",
                        appellation="Updated Appellation",
                        cru_vineyard="Updated Cru",
                        vintage="2021"
                    ),
                    rejection_reasons=[]
                )
                cache.put(result2.sku_id, result2)
                
                # Verify only one row exists and it has updated values
                row = cache._conn.execute(
                    "SELECT COUNT(*) as count FROM pipeline_results WHERE sku_id = ?",
                    (result1.sku_id,)
                ).fetchone()
                assert row["count"] == 1
                
                retrieved = cache.get(result1.sku_id)
                assert retrieved.image_url == "https://example.com/wine2.jpg"
                assert retrieved.overall_confidence == 0.95
                assert retrieved.verdict == Verdict.PASS
            finally:
                cache.close()


class TestCacheRetrieval:
    """Tests for cache retrieval and miss handling (Requirement 9.2)."""

    def test_retrieve_existing_result(self):
        """Test retrieving a cached result by SKU ID."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                result = ScoredResult(
                    sku_id="test-sku-010",
                    image_url="https://example.com/wine.jpg",
                    producer_match=0.88,
                    appellation_match=0.92,
                    cru_match=0.85,
                    vintage_match=0.95,
                    image_quality=0.90,
                    overall_confidence=0.88,
                    verdict=Verdict.PASS,
                    fingerprint=Fingerprint(
                        producer="Château Lafite",
                        appellation="Pauillac",
                        cru_vineyard="Premier Cru",
                        vintage="2018"
                    ),
                    rejection_reasons=[]
                )
                
                cache.put(result.sku_id, result)
                retrieved = cache.get(result.sku_id)
                
                assert retrieved is not None
                assert retrieved.sku_id == result.sku_id
                assert retrieved.image_url == result.image_url
                assert retrieved.producer_match == result.producer_match
                assert retrieved.appellation_match == result.appellation_match
                assert retrieved.cru_match == result.cru_match
                assert retrieved.vintage_match == result.vintage_match
                assert retrieved.image_quality == result.image_quality
                assert retrieved.overall_confidence == result.overall_confidence
                assert retrieved.verdict == result.verdict
                assert retrieved.fingerprint.producer == "Château Lafite"
                assert retrieved.rejection_reasons == []
            finally:
                cache.close()

    def test_cache_miss_returns_none(self):
        """Test that retrieving non-existent SKU ID returns None."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                retrieved = cache.get("non-existent-sku-id")
                assert retrieved is None
            finally:
                cache.close()

    def test_retrieve_result_with_null_fields(self):
        """Test retrieving result with None/null fields."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                result = ScoredResult(
                    sku_id="test-sku-011",
                    image_url=None,
                    producer_match=0.0,
                    appellation_match=0.0,
                    cru_match=0.0,
                    vintage_match=0.0,
                    image_quality=0.0,
                    overall_confidence=0.0,
                    verdict=Verdict.REJECT,
                    fingerprint=None,
                    rejection_reasons=["No Image"]
                )
                
                cache.put(result.sku_id, result)
                retrieved = cache.get(result.sku_id)
                
                assert retrieved is not None
                assert retrieved.image_url is None
                assert retrieved.fingerprint is None
                assert retrieved.rejection_reasons == ["No Image"]
            finally:
                cache.close()


class TestCacheInvalidation:
    """Tests for cache invalidation (Requirements 9.1, 9.2, 9.3, 9.4)."""

    def test_invalidate_removes_cached_result(self):
        """Test that invalidation removes a cached result."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                result = ScoredResult(
                    sku_id="test-sku-020",
                    image_url="https://example.com/wine.jpg",
                    producer_match=0.9,
                    appellation_match=0.9,
                    cru_match=0.9,
                    vintage_match=0.9,
                    image_quality=0.9,
                    overall_confidence=0.9,
                    verdict=Verdict.PASS,
                    fingerprint=None,
                    rejection_reasons=[]
                )
                
                cache.put(result.sku_id, result)
                assert cache.get(result.sku_id) is not None
                
                cache.invalidate(result.sku_id)
                assert cache.get(result.sku_id) is None
            finally:
                cache.close()

    def test_invalidate_non_existent_is_safe(self):
        """Test that invalidating non-existent SKU ID doesn't raise error."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                # Should not raise
                cache.invalidate("non-existent-sku-id")
                
                # Verify no rows affected
                row = cache._conn.execute(
                    "SELECT COUNT(*) as count FROM pipeline_results"
                ).fetchone()
                assert row["count"] == 0
            finally:
                cache.close()


class TestSerializationRoundTrip:
    """Tests for fingerprint serialization/deserialization round trip."""

    def test_fingerprint_round_trip_full(self):
        """Test fingerprint with all fields serializes correctly."""
        fp = Fingerprint(
            producer="Château Margaux",
            appellation="Margaux",
            cru_vineyard="Premier Grand Cru Classé",
            vintage="2015"
        )
        
        serialized = ResultCache._serialize_fingerprint(fp)
        deserialized = ResultCache._deserialize_fingerprint(serialized)
        
        assert deserialized.producer == fp.producer
        assert deserialized.appellation == fp.appellation
        assert deserialized.cru_vineyard == fp.cru_vineyard
        assert deserialized.vintage == fp.vintage

    def test_fingerprint_round_trip_partial(self):
        """Test fingerprint with some None fields serializes correctly."""
        fp = Fingerprint(
            producer="Test Winery",
            appellation=None,
            cru_vineyard=None,
            vintage="NV"
        )
        
        serialized = ResultCache._serialize_fingerprint(fp)
        deserialized = ResultCache._deserialize_fingerprint(serialized)
        
        assert deserialized.producer == "Test Winery"
        assert deserialized.appellation is None
        assert deserialized.cru_vineyard is None
        assert deserialized.vintage == "NV"

    def test_fingerprint_round_trip_none(self):
        """Test None fingerprint serializes to None."""
        serialized = ResultCache._serialize_fingerprint(None)
        deserialized = ResultCache._deserialize_fingerprint(serialized)
        
        assert serialized is None
        assert deserialized is None


class TestCacheBypass:
    """Tests for cache bypass functionality (Requirement 9.3).
    
    Note: The bypass_cache parameter is implemented in the Pipeline class,
    not in ResultCache directly. These tests verify the cache-level behavior
    that enables bypass (invalidate + reprocess pattern).
    """

    def test_bypass_via_invalidation_and_reprocess(self):
        """Test that invalidating a cached result allows reprocessing.
        
        Simulates the bypass_cache=True pattern:
        1. Store initial result
        2. Invalidate (what bypass does internally before reprocessing)
        3. Store new result
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                # Initial result
                result1 = ScoredResult(
                    sku_id="test-sku-bypass",
                    image_url="https://example.com/wine1.jpg",
                    producer_match=0.5,
                    appellation_match=0.5,
                    cru_match=0.5,
                    vintage_match=0.5,
                    image_quality=0.5,
                    overall_confidence=0.5,
                    verdict=Verdict.QUARANTINE,
                    fingerprint=None,
                    rejection_reasons=["Initial processing"]
                )
                cache.put(result1.sku_id, result1)
                
                # Simulate bypass: invalidate
                cache.invalidate(result1.sku_id)
                
                # Simulate reprocessing: store new result
                result2 = ScoredResult(
                    sku_id="test-sku-bypass",
                    image_url="https://example.com/wine2.jpg",
                    producer_match=0.95,
                    appellation_match=0.95,
                    cru_match=0.95,
                    vintage_match=0.95,
                    image_quality=0.95,
                    overall_confidence=0.95,
                    verdict=Verdict.PASS,
                    fingerprint=Fingerprint(
                        producer="Reprocessed Producer",
                        appellation="Reprocessed Appellation",
                        cru_vineyard=None,
                        vintage="2022"
                    ),
                    rejection_reasons=[]
                )
                cache.put(result2.sku_id, result2)
                
                # Verify new result is retrieved
                retrieved = cache.get(result2.sku_id)
                assert retrieved.image_url == "https://example.com/wine2.jpg"
                assert retrieved.overall_confidence == 0.95
                assert retrieved.verdict == Verdict.PASS
                assert retrieved.fingerprint.producer == "Reprocessed Producer"
            finally:
                cache.close()

    def test_get_after_invalidate_returns_none(self):
        """Test that get returns None after invalidation, enabling reprocess check."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
            cache = ResultCache(db_path=tmp.name)
            try:
                result = ScoredResult(
                    sku_id="test-sku-bypass2",
                    image_url="https://example.com/wine.jpg",
                    producer_match=0.8,
                    appellation_match=0.8,
                    cru_match=0.8,
                    vintage_match=0.8,
                    image_quality=0.8,
                    overall_confidence=0.8,
                    verdict=Verdict.PASS,
                    fingerprint=None,
                    rejection_reasons=[]
                )
                
                # Store and verify
                cache.put(result.sku_id, result)
                assert cache.get(result.sku_id) is not None
                
                # Invalidate and verify None
                cache.invalidate(result.sku_id)
                assert cache.get(result.sku_id) is None
                
                # This None signals to pipeline that reprocessing is needed
            finally:
                cache.close()
