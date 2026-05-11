"""Unit tests for Pipeline output generation functions.

Requirements: 10.1, 10.2, 10.3, 10.4
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from wine_pipeline.models import ScoredResult, Verdict
from wine_pipeline.pipeline import Pipeline, _no_image_result


# ---------------------------------------------------------------------------
# Task 10.2: _no_image_result tests
# ---------------------------------------------------------------------------


def test_no_image_result_returns_correct_structure():
    """_no_image_result returns a ScoredResult with all zero scores."""
    result = _no_image_result("SKU-TEST-001")
    
    assert isinstance(result, ScoredResult)
    assert result.sku_id == "SKU-TEST-001"
    assert result.image_url is None
    assert result.producer_match == 0.0
    assert result.appellation_match == 0.0
    assert result.cru_match == 0.0
    assert result.vintage_match == 0.0
    assert result.image_quality == 0.0
    assert result.overall_confidence == 0.0
    assert result.verdict == Verdict.REJECT
    assert result.fingerprint is None
    assert "No Image" in result.rejection_reasons


def test_no_image_result_with_different_sku_ids():
    """_no_image_result works with various SKU ID formats."""
    sku_ids = ["SKU-001", "wine-123", "test_sku", "ABC123!@#"]
    
    for sku_id in sku_ids:
        result = _no_image_result(sku_id)
        assert result.sku_id == sku_id
        assert result.image_url is None
        assert result.verdict == Verdict.REJECT


def test_no_image_result_json_round_trip():
    """_no_image_result can be serialized and deserialized correctly."""
    result = _no_image_result("SKU-JSON-TEST")
    
    json_data = result.to_json()
    restored = ScoredResult.from_json(json_data)
    
    assert restored.sku_id == result.sku_id
    assert restored.image_url is None
    assert restored.verdict == Verdict.REJECT
    assert restored.fingerprint is None
    assert "No Image" in restored.rejection_reasons


# ---------------------------------------------------------------------------
# Task 10.3: _save_image_locally tests
# ---------------------------------------------------------------------------


def test_save_image_locally_creates_directory():
    """_save_image_locally creates the data/images directory if it doesn't exist."""
    fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch the working directory to use temp dir
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            Pipeline._save_image_locally("SKU-SAVE-TEST", fake_image)
            
            # Check directory was created
            img_dir = os.path.join("data", "images")
            assert os.path.exists(img_dir)
            
            # Check file was created
            expected_path = os.path.join(img_dir, "SKU-SAVE-TEST.jpg")
            assert os.path.exists(expected_path)
            
            # Check file contents
            with open(expected_path, "rb") as f:
                saved_content = f.read()
            assert saved_content == fake_image
        finally:
            os.chdir(original_cwd)


def test_save_image_locally_overwrites_existing_file():
    """_save_image_locally overwrites existing image file for same SKU."""
    fake_image_1 = b"\x89PNG\r\n\x1a\n" + b"\x01" * 50
    fake_image_2 = b"\x89PNG\r\n\x1a\n" + b"\x02" * 50
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            # Save first image
            Pipeline._save_image_locally("SKU-OVERWRITE", fake_image_1)
            
            # Save second image with same SKU ID
            Pipeline._save_image_locally("SKU-OVERWRITE", fake_image_2)
            
            # Check file contains second image
            expected_path = os.path.join("data", "images", "SKU-OVERWRITE.jpg")
            with open(expected_path, "rb") as f:
                saved_content = f.read()
            assert saved_content == fake_image_2
        finally:
            os.chdir(original_cwd)


def test_save_image_locally_handles_special_characters_in_sku_id():
    """_save_image_locally handles SKU IDs with special characters."""
    fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            # Test SKU IDs that are valid for filenames
            test_ids = ["SKU-001", "SKU_TEST", "SKU.123", "SKU 456"]
            
            for sku_id in test_ids:
                Pipeline._save_image_locally(sku_id, fake_image)
                
                expected_path = os.path.join("data", "images", f"{sku_id}.jpg")
                assert os.path.exists(expected_path), f"File not created for {sku_id}"
        finally:
            os.chdir(original_cwd)


def test_save_image_locally_creates_unique_files():
    """_save_image_locally creates separate files for different SKU IDs."""
    fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            Pipeline._save_image_locally("SKU-UNIQUE-1", fake_image)
            Pipeline._save_image_locally("SKU-UNIQUE-2", fake_image)
            
            # Check both files exist
            path1 = os.path.join("data", "images", "SKU-UNIQUE-1.jpg")
            path2 = os.path.join("data", "images", "SKU-UNIQUE-2.jpg")
            
            assert os.path.exists(path1)
            assert os.path.exists(path2)
        finally:
            os.chdir(original_cwd)

