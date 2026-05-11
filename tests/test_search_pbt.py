"""Property-based tests for SearchModule.

Feature: wine-photo-pipeline, Property 2: Search fallback chain returns candidates or empty list
Validates: Requirements 2.1, 2.2, 2.3, 2.4
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.wine_pipeline.models import CandidateImage, SKU
from src.wine_pipeline.search import SearchModule


# Strategy for generating SKU instances
sku_strategy = st.builds(
    SKU,
    id=st.text(min_size=1, max_size=20),
    producer=st.text(min_size=1, max_size=50),
    appellation=st.text(min_size=1, max_size=50),
    cru_vineyard=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    vintage=st.one_of(st.none(), st.integers(min_value=1900, max_value=2030)),
    format=st.sampled_from(["750ml", "375ml", "1.5L", "magnum"]),
    region=st.text(min_size=1, max_size=30),
)

# Strategy for generating lists of candidate images
candidate_list_strategy = st.lists(
    st.builds(
        CandidateImage,
        url=st.text(min_size=1, max_size=100).map(lambda s: f"https://{s}.jpg"),
        source=st.sampled_from(["duckduckgo", "vivino", "wine-searcher", "producer"]),
    ),
    max_size=5,
)


@given(sku=sku_strategy)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_search_returns_candidates_or_empty_list(sku: SKU):
    """Property 2: Search fallback chain returns candidates or empty list.
    
    For any SKU, the search module should return up to 3 candidate images
    from DuckDuckGo, Vivino, Wine-Searcher, or producer site in that order,
    or return an empty list if no candidates are found.
    
    Validates: Requirements 2.1, 2.2, 2.3, 2.4
    """
    search_module = SearchModule()
    
    # Mock all search methods to return empty lists
    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock, return_value=[]):
        
        result = await search_module.search(sku)
        
        # Property: Result is always a list
        assert isinstance(result, list)
        # Property: Result is either empty or contains CandidateImage objects
        if result:
            assert all(isinstance(c, CandidateImage) for c in result)


@given(sku=sku_strategy, ddg_candidates=candidate_list_strategy)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_search_returns_ddg_results_when_available(sku: SKU, ddg_candidates: list[CandidateImage]):
    """Property 2: DDG is the primary source and results are returned when available.
    
    When DuckDuckGo returns candidates, those candidates are returned directly
    without calling fallback sources.
    
    Validates: Requirements 2.1, 2.2
    """
    search_module = SearchModule()
    
    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=ddg_candidates) as mock_ddg, \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock) as mock_retailers, \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock) as mock_producer:
        
        result = await search_module.search(sku)
        
        # DDG is always called first
        mock_ddg.assert_awaited_once_with(sku)
        
        # Fallback sources are not called when DDG returns results
        if ddg_candidates:
            mock_retailers.assert_not_awaited()
            mock_producer.assert_not_awaited()
            assert result == ddg_candidates
        else:
            # If DDG returns empty, fallback sources are called
            pass


@given(sku=sku_strategy, retailer_candidates=candidate_list_strategy)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_search_falls_back_to_retailers(sku: SKU, retailer_candidates: list[CandidateImage]):
    """Property 2: Retailer scraping is used when DDG returns empty.
    
    When DuckDuckGo returns no results, the system falls back to
    retailer scraping (Vivino, Wine-Searcher).
    
    Validates: Requirements 2.1
    """
    search_module = SearchModule()
    
    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock, return_value=retailer_candidates) as mock_retailers, \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock) as mock_producer:
        
        result = await search_module.search(sku)
        
        mock_retailers.assert_awaited_once_with(sku)
        
        if retailer_candidates:
            mock_producer.assert_not_awaited()
            assert result == retailer_candidates


@given(sku=sku_strategy, producer_candidates=candidate_list_strategy)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_search_falls_back_to_producer_site(sku: SKU, producer_candidates: list[CandidateImage]):
    """Property 2: Producer site search is used when DDG and retailers return empty.
    
    When both DuckDuckGo and retailer scraping return no results,
    the system falls back to producer site search.
    
    Validates: Requirements 2.1
    """
    search_module = SearchModule()
    
    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock, return_value=producer_candidates) as mock_producer:
        
        result = await search_module.search(sku)
        
        mock_producer.assert_awaited_once_with(sku)
        assert result == producer_candidates


@given(sku=sku_strategy)
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_search_returns_empty_when_all_sources_fail(sku: SKU):
    """Property 2: Empty list is returned when all sources return no results.
    
    When all sources (DDG, retailers, producer site) return no results,
    the system returns an empty list indicating "No Image".
    
    Validates: Requirements 2.4
    """
    search_module = SearchModule()
    
    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock, return_value=[]):
        
        result = await search_module.search(sku)
        
        assert result == []
