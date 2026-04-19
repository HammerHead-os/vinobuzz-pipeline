"""Unit tests for SearchModule fallback chain."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.wine_pipeline.models import CandidateImage, SKU
from src.wine_pipeline.search import SearchModule


@pytest.fixture
def sample_sku() -> SKU:
    return SKU(
        id="SKU001",
        producer="Domaine Leflaive",
        appellation="Puligny-Montrachet",
        cru_vineyard="Les Pucelles",
        vintage=2020,
        format="750ml",
        region="Burgundy",
    )


@pytest.fixture
def search_module() -> SearchModule:
    return SearchModule()


@pytest.mark.asyncio
async def test_serpapi_called_first(search_module: SearchModule, sample_sku: SKU):
    """DDG is always the first source queried (Req 1.1)."""
    ddg_candidates = [CandidateImage(url="https://img.example.com/wine1.jpg", source="duckduckgo")]

    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=ddg_candidates) as mock_ddg, \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock) as mock_retail, \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock) as mock_prod:

        result = await search_module.search(sample_sku)

        mock_ddg.assert_awaited_once_with(sample_sku)
        mock_retail.assert_not_awaited()
        mock_prod.assert_not_awaited()
        assert len(result) == 1
        assert result[0].source == "duckduckgo"


@pytest.mark.asyncio
async def test_retailers_called_when_serpapi_empty(search_module: SearchModule, sample_sku: SKU):
    """When DDG returns empty, retailers are called next (Req 1.2)."""
    retailer_candidates = [CandidateImage(url="https://vivino.com/wine.jpg", source="vivino")]

    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=[]) as mock_ddg, \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock, return_value=retailer_candidates) as mock_retail, \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock) as mock_prod:

        result = await search_module.search(sample_sku)

        mock_ddg.assert_awaited_once()
        mock_retail.assert_awaited_once_with(sample_sku)
        mock_prod.assert_not_awaited()
        assert len(result) == 1
        assert result[0].source == "vivino"


@pytest.mark.asyncio
async def test_producer_called_when_retailers_empty(search_module: SearchModule, sample_sku: SKU):
    """When retailers also return empty, producer site is tried (Req 1.3)."""
    producer_candidates = [CandidateImage(url="https://producer.com/bottle.jpg", source="producer")]

    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock, return_value=producer_candidates):

        result = await search_module.search(sample_sku)

        assert len(result) == 1
        assert result[0].source == "producer"


@pytest.mark.asyncio
async def test_no_image_when_all_sources_empty(search_module: SearchModule, sample_sku: SKU):
    """When all sources return empty, result is empty list — 'No Image' (Req 1.4)."""
    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock, return_value=[]):

        result = await search_module.search(sample_sku)

        assert result == []
