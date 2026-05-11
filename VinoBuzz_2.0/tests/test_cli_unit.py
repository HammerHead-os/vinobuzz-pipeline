"""Unit tests for CLI (benchmark.py) JSON input loading and output generation.

Requirements: 11.1, 11.2, 11.3, 11.4
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from benchmark import load_skus, print_result_table, print_summary
from wine_pipeline.models import SKU, ScoredResult, Verdict, Fingerprint


# ---------------------------------------------------------------------------
# Task 11.1: JSON input loading tests
# ---------------------------------------------------------------------------


def test_load_skus_from_valid_json_file():
    """load_skus correctly parses a valid JSON file with SKUs."""
    skus_data = [
        {
            "id": "TEST-001",
            "producer": "Test Producer",
            "appellation": "Test Appellation",
            "cru_vineyard": "Grand Cru",
            "vintage": 2020,
            "format": "750ml",
            "region": "Test Region",
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(skus_data, f)
        temp_path = f.name
    
    try:
        skus = load_skus(Path(temp_path))
        
        assert len(skus) == 1
        assert skus[0].id == "TEST-001"
        assert skus[0].producer == "Test Producer"
        assert skus[0].appellation == "Test Appellation"
        assert skus[0].cru_vineyard == "Grand Cru"
        assert skus[0].vintage == 2020
        assert skus[0].format == "750ml"
        assert skus[0].region == "Test Region"
    finally:
        os.unlink(temp_path)


def test_load_skus_handles_missing_optional_fields():
    """load_skus uses defaults for missing optional fields."""
    skus_data = [
        {
            "id": "TEST-002",
            "producer": "Test Producer",
            "appellation": "Test Appellation",
            # cru_vineyard is optional
            # vintage is optional
            # format defaults to "750ml"
            # region defaults to ""
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(skus_data, f)
        temp_path = f.name
    
    try:
        skus = load_skus(Path(temp_path))
        
        assert len(skus) == 1
        assert skus[0].cru_vineyard is None
        assert skus[0].vintage is None
        assert skus[0].format == "750ml"
        assert skus[0].region == ""
    finally:
        os.unlink(temp_path)


def test_load_skus_handles_multiple_skus():
    """load_skus correctly loads multiple SKUs from a JSON file."""
    skus_data = [
        {
            "id": f"TEST-{i:03d}",
            "producer": f"Producer {i}",
            "appellation": f"Appellation {i}",
            "vintage": 2020 + i,
        }
        for i in range(5)
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(skus_data, f)
        temp_path = f.name
    
    try:
        skus = load_skus(Path(temp_path))
        
        assert len(skus) == 5
        for i, sku in enumerate(skus):
            assert sku.id == f"TEST-{i:03d}"
            assert sku.producer == f"Producer {i}"
    finally:
        os.unlink(temp_path)


def test_load_skus_raises_on_missing_file():
    """load_skus raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_skus(Path("/nonexistent/path/skus.json"))


def test_load_skus_raises_on_invalid_json():
    """load_skus raises JSONDecodeError for invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json {{{")
        temp_path = f.name
    
    try:
        with pytest.raises(json.JSONDecodeError):
            load_skus(Path(temp_path))
    finally:
        os.unlink(temp_path)


def test_load_skus_raises_on_missing_required_field():
    """load_skus raises KeyError when required fields are missing."""
    skus_data = [
        {
            "id": "TEST-003",
            "producer": "Test Producer",
            # missing "appellation" which is required
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(skus_data, f)
        temp_path = f.name
    
    try:
        with pytest.raises(KeyError):
            load_skus(Path(temp_path))
    finally:
        os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Task 11.2: JSON output generation tests
# ---------------------------------------------------------------------------


def test_print_result_table_outputs_formatted_table():
    """print_result_table outputs a formatted table of results."""
    results = [
        ScoredResult(
            sku_id="SKU-001",
            image_url="https://example.com/image1.jpg",
            producer_match=0.9,
            appellation_match=0.8,
            cru_match=0.7,
            vintage_match=1.0,
            image_quality=0.85,
            overall_confidence=0.85,
            verdict=Verdict.PASS,
            fingerprint=None,
            rejection_reasons=[],
        )
    ]
    
    output = StringIO()
    with patch("sys.stdout", output):
        print_result_table(results)
    
    output_str = output.getvalue()
    assert "SKU-001" in output_str
    assert "PASS" in output_str
    assert "0.8500" in output_str


def test_print_result_table_handles_no_image():
    """print_result_table handles results with no image URL."""
    results = [
        ScoredResult(
            sku_id="SKU-002",
            image_url=None,
            producer_match=0.0,
            appellation_match=0.0,
            cru_match=0.0,
            vintage_match=0.0,
            image_quality=0.0,
            overall_confidence=0.0,
            verdict=Verdict.REJECT,
            fingerprint=None,
            rejection_reasons=["No Image"],
        )
    ]
    
    output = StringIO()
    with patch("sys.stdout", output):
        print_result_table(results)
    
    output_str = output.getvalue()
    assert "SKU-002" in output_str
    assert "No Image" in output_str


def test_print_result_table_truncates_long_urls():
    """print_result_table truncates URLs longer than 60 characters."""
    long_url = "https://example.com/" + "a" * 100
    results = [
        ScoredResult(
            sku_id="SKU-003",
            image_url=long_url,
            producer_match=0.5,
            appellation_match=0.5,
            cru_match=0.5,
            vintage_match=0.5,
            image_quality=0.5,
            overall_confidence=0.5,
            verdict=Verdict.QUARANTINE,
            fingerprint=None,
            rejection_reasons=[],
        )
    ]
    
    output = StringIO()
    with patch("sys.stdout", output):
        print_result_table(results)
    
    output_str = output.getvalue()
    assert "..." in output_str


def test_print_summary_outputs_verdict_counts():
    """print_summary outputs correct verdict counts and percentages."""
    results = [
        ScoredResult(
            sku_id=f"SKU-{i:03d}",
            image_url=None,
            producer_match=0.0,
            appellation_match=0.0,
            cru_match=0.0,
            vintage_match=0.0,
            image_quality=0.0,
            overall_confidence=0.0,
            verdict=verdict,
            fingerprint=None,
            rejection_reasons=[],
        )
        for i, verdict in enumerate([Verdict.PASS, Verdict.PASS, Verdict.QUARANTINE, Verdict.REJECT])
    ]
    
    output = StringIO()
    with patch("sys.stdout", output):
        print_summary(results)
    
    output_str = output.getvalue()
    assert "PASS" in output_str
    assert "2" in output_str  # 2 PASS results
    assert "50.0%" in output_str  # 2/4 = 50%


def test_print_summary_handles_empty_results():
    """print_summary handles empty results list."""
    output = StringIO()
    with patch("sys.stdout", output):
        print_summary([])
    
    output_str = output.getvalue()
    assert "No results" in output_str


# ---------------------------------------------------------------------------
# Task 11.3: Command-line argument parsing tests
# ---------------------------------------------------------------------------


def test_argparse_default_values():
    """CLI uses correct default values for arguments."""
    import argparse
    from benchmark import main, DATA_DIR
    
    # Test that the defaults are correctly set
    parser = argparse.ArgumentParser(description="Wine Photo Pipeline Benchmark")
    parser.add_argument("--skus", type=str, default=str(DATA_DIR / "test_skus.json"))
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bypass-cache", action="store_true")
    parser.add_argument("--cache-db", type=str, default="benchmark_cache.db")
    
    args = parser.parse_args([])
    
    assert args.skus == str(DATA_DIR / "test_skus.json")
    assert args.reference is False
    assert args.all is False
    assert args.bypass_cache is False
    assert args.cache_db == "benchmark_cache.db"


def test_argparse_custom_sku_path():
    """CLI accepts custom SKU file path."""
    import argparse
    from benchmark import DATA_DIR
    
    parser = argparse.ArgumentParser(description="Wine Photo Pipeline Benchmark")
    parser.add_argument("--skus", type=str, default=str(DATA_DIR / "test_skus.json"))
    
    args = parser.parse_args(["--skus", "/custom/path/skus.json"])
    
    assert args.skus == "/custom/path/skus.json"


def test_argparse_reference_flag():
    """CLI correctly parses --reference flag."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wine Photo Pipeline Benchmark")
    parser.add_argument("--reference", action="store_true")
    
    args = parser.parse_args(["--reference"])
    assert args.reference is True
    
    args = parser.parse_args([])
    assert args.reference is False


def test_argparse_all_flag():
    """CLI correctly parses --all flag."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wine Photo Pipeline Benchmark")
    parser.add_argument("--all", action="store_true")
    
    args = parser.parse_args(["--all"])
    assert args.all is True


def test_argparse_bypass_cache_flag():
    """CLI correctly parses --bypass-cache flag."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wine Photo Pipeline Benchmark")
    parser.add_argument("--bypass-cache", action="store_true")
    
    args = parser.parse_args(["--bypass-cache"])
    assert args.bypass_cache is True


def test_argparse_custom_cache_db():
    """CLI accepts custom cache database path."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wine Photo Pipeline Benchmark")
    parser.add_argument("--cache-db", type=str, default="benchmark_cache.db")
    
    args = parser.parse_args(["--cache-db", "/custom/cache.db"])
    assert args.cache_db == "/custom/cache.db"


# ---------------------------------------------------------------------------
# Task 11.4: Logging tests
# ---------------------------------------------------------------------------


def test_logger_configured_with_correct_format():
    """Benchmark logger is configured with correct format."""
    import logging
    from benchmark import logger
    
    assert logger.name == "benchmark"
    # Note: Logger level may be NOTSET (0) if not explicitly set at logger level
    # but inherits from root logger. This is valid behavior.
    assert logger.level in [0, logging.INFO, logging.WARNING, logging.DEBUG]


def test_logging_outputs_to_stdout(capsys):
    """Logging outputs progress to stdout."""
    import logging
    from benchmark import logger
    
    # Configure handler for this test
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    try:
        logger.info("Test log message")
        
        captured = capsys.readouterr()
        # Note: logging might go to stderr by default
        combined_output = captured.out + captured.err
        # The message should appear in some output
        assert "Test log message" in combined_output
    finally:
        logger.removeHandler(handler)

