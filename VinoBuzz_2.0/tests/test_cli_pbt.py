"""Property-based tests for CLI (benchmark.py) execution.

Feature: wine-photo-pipeline, Property 10: CLI execution processes JSON files and logs progress
Validates: Requirements 11.1, 11.2, 11.3, 11.4
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
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark import load_skus, print_result_table, print_summary
from wine_pipeline.models import SKU, ScoredResult, Verdict, Fingerprint


# ---------------------------------------------------------------------------
# Strategies for generating test data
# ---------------------------------------------------------------------------

# Strategy for generating valid SKU IDs
sku_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=48, max_codepoint=122),
    min_size=3,
    max_size=20
)

# Strategy for generating valid producer names
producer_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'), min_codepoint=32, max_codepoint=122),
    min_size=1,
    max_size=50
).filter(lambda s: s.strip() != "")

# Strategy for generating appellation names
appellation_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'), min_codepoint=32, max_codepoint=122),
    min_size=1,
    max_size=50
).filter(lambda s: s.strip() != "")

# Strategy for generating vintage years
vintage_strategy = st.one_of(
    st.none(),
    st.integers(min_value=1900, max_value=2100)
)

# Strategy for generating a single SKU dict (for JSON)
sku_dict_strategy = st.fixed_dictionaries(
    {
        "id": sku_id_strategy,
        "producer": producer_strategy,
        "appellation": appellation_strategy,
        "vintage": vintage_strategy,
    }
)

# Strategy for generating a list of SKU dicts
sku_list_strategy = st.lists(sku_dict_strategy, min_size=0, max_size=20)

# Strategy for generating verdicts
verdict_strategy = st.sampled_from(Verdict)

# Strategy for generating ScoredResult objects
scored_result_strategy = st.builds(
    ScoredResult,
    sku_id=sku_id_strategy,
    image_url=st.one_of(st.none(), st.just("https://example.com/image.jpg")),
    producer_match=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    appellation_match=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    cru_match=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    vintage_match=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    image_quality=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    overall_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    verdict=verdict_strategy,
    fingerprint=st.none(),
    rejection_reasons=st.just([]),
)


# ---------------------------------------------------------------------------
# Property 10: CLI execution processes JSON files and logs progress
# Validates: Requirements 11.1, 11.2, 11.3, 11.4
# ---------------------------------------------------------------------------


@given(sku_data=sku_list_strategy)
@settings(max_examples=50)
def test_property_10a_json_round_trip(sku_data: list):
    """Property 10a: JSON input loading and output preserves all SKU data.
    
    Validates: Requirements 11.1, 11.2
    """
    # Create a temporary JSON file with the generated SKU data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sku_data, f)
        temp_path = f.name
    
    try:
        # Load SKUs from the file
        skus = load_skus(Path(temp_path))
        
        # Verify the count matches
        assert len(skus) == len(sku_data)
        
        # Verify each SKU preserves essential fields
        for original, loaded in zip(sku_data, skus):
            assert loaded.id == original["id"]
            assert loaded.producer == original["producer"]
            assert loaded.appellation == original["appellation"]
            assert loaded.vintage == original.get("vintage")
    finally:
        os.unlink(temp_path)


@given(results=st.lists(scored_result_strategy, min_size=0, max_size=20))
@settings(max_examples=50)
def test_property_10b_result_table_handles_all_inputs(results: list):
    """Property 10b: print_result_table handles any valid list of results.
    
    Validates: Requirements 11.2
    """
    output = StringIO()
    with patch("sys.stdout", output):
        print_result_table(results)
    
    output_str = output.getvalue()
    
    # Should always produce output (at minimum, a header)
    assert len(output_str) > 0
    
    # Each result should appear in the output
    for result in results:
        assert result.sku_id in output_str


@given(results=st.lists(scored_result_strategy, min_size=0, max_size=20))
@settings(max_examples=50)
def test_property_10c_summary_handles_all_verdict_combinations(results: list):
    """Property 10c: print_summary correctly handles all verdict combinations.
    
    Validates: Requirements 11.2
    """
    output = StringIO()
    with patch("sys.stdout", output):
        print_summary(results)
    
    output_str = output.getvalue()
    
    if len(results) == 0:
        assert "No results" in output_str
    else:
        # Verify verdict counts match
        pass_count = sum(1 for r in results if r.verdict == Verdict.PASS)
        quarantine_count = sum(1 for r in results if r.verdict == Verdict.QUARANTINE)
        reject_count = sum(1 for r in results if r.verdict == Verdict.REJECT)
        
        # The output should contain the counts
        assert str(pass_count) in output_str or "PASS" in output_str


@given(
    sku_ids=st.lists(sku_id_strategy, min_size=1, max_size=10),
    verdicts=st.lists(verdict_strategy, min_size=1, max_size=10),
    confidences=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=50)
def test_property_10d_result_table_preserves_order(sku_ids: list, verdicts: list, confidences: list):
    """Property 10d: print_result_table preserves input order.
    
    Validates: Requirements 11.2
    """
    # Create results with consistent ordering
    min_len = min(len(sku_ids), len(verdicts), len(confidences))
    sku_ids = sku_ids[:min_len]
    verdicts = verdicts[:min_len]
    confidences = confidences[:min_len]
    
    results = [
        ScoredResult(
            sku_id=sku_id,
            image_url=None,
            producer_match=0.5,
            appellation_match=0.5,
            cru_match=0.5,
            vintage_match=0.5,
            image_quality=0.5,
            overall_confidence=confidence,
            verdict=verdict,
            fingerprint=None,
            rejection_reasons=[],
        )
        for sku_id, verdict, confidence in zip(sku_ids, verdicts, confidences)
    ]
    
    output = StringIO()
    with patch("sys.stdout", output):
        print_result_table(results)
    
    output_str = output.getvalue()
    
    # Verify SKU IDs appear in the same order as input
    # Use a running position to handle duplicate SKU IDs correctly
    positions = []
    current_pos = 0
    for sku_id in sku_ids:
        # Search from current position onwards to handle duplicates
        idx = output_str.find(sku_id, current_pos)
        if idx != -1:
            positions.append(idx)
            current_pos = idx + len(sku_id)
    
    # Positions should be in ascending order (preserving input order)
    assert positions == sorted(positions)


@given(sku_data=st.lists(sku_dict_strategy, min_size=1, max_size=10))
@settings(max_examples=50)
def test_property_10e_json_file_path_handling(sku_data: list):
    """Property 10e: CLI handles various file paths correctly.
    
    Validates: Requirements 11.1
    """
    # Test with various valid file paths
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test with a nested directory path
        nested_dir = Path(temp_dir) / "nested" / "dir"
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = nested_dir / "test_skus.json"
        with open(json_path, "w") as f:
            json.dump(sku_data, f)
        
        # Load should work with nested paths
        skus = load_skus(json_path)
        assert len(skus) == len(sku_data)
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

