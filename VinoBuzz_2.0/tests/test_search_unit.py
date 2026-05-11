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


# ------------------------------------------------------------------
# Query Construction Tests (Req 2.1, 2.2)
# ------------------------------------------------------------------

def test_build_query_with_all_fields(search_module: SearchModule, sample_sku: SKU):
    """Query includes producer, appellation, cru, vintage, region, and 'wine bottle'."""
    query = SearchModule._build_query(sample_sku)
    assert "Domaine Leflaive" in query
    assert "Puligny-Montrachet" in query
    assert "Les Pucelles" in query
    assert "2020" in query
    assert "Burgundy" in query
    assert "wine bottle" in query


def test_build_query_without_cru(search_module: SearchModule):
    """Query omits cru when not provided."""
    sku = SKU(
        id="SKU002",
        producer="Château Margaux",
        appellation="Margaux",
        cru_vineyard=None,
        vintage=2018,
        format="750ml",
        region="Bordeaux",
    )
    query = SearchModule._build_query(sku)
    assert "Château Margaux" in query
    assert "Margaux" in query
    assert "2018" in query
    assert "Bordeaux" in query
    assert "wine bottle" in query


def test_build_query_without_vintage_nv_wine(search_module: SearchModule):
    """Query omits vintage for NV wines."""
    sku = SKU(
        id="SKU003",
        producer="Veuve Clicquot",
        appellation="Champagne",
        cru_vineyard=None,
        vintage=None,  # NV wine
        format="750ml",
        region="Champagne",
    )
    query = SearchModule._build_query(sku)
    assert "Veuve Clicquot" in query
    assert "Champagne" in query
    assert "NV" not in query  # Should not include vintage
    assert "wine bottle" in query


def test_build_query_avoids_duplicate_cru_in_appellation(search_module: SearchModule):
    """Cru is not duplicated if already in appellation."""
    sku = SKU(
        id="SKU004",
        producer="Domaine Test",
        appellation="Les Pucelles Montrachet",
        cru_vineyard="Les Pucelles",
        vintage=2019,
        format="750ml",
        region="Burgundy",
    )
    query = SearchModule._build_query(sku)
    # cru_vineyard should not be added since it's already in appellation
    assert query.count("Les Pucelles") == 1


# ------------------------------------------------------------------
# Image URL Filtering Tests (Req 2.1, 2.2)
# ------------------------------------------------------------------

def test_is_product_image_accepts_valid_jpg(search_module: SearchModule):
    """Valid JPG URLs are accepted."""
    assert SearchModule._is_product_image("https://example.com/wine.jpg")
    assert SearchModule._is_product_image("https://example.com/wine.JPEG")


def test_is_product_image_accepts_valid_png(search_module: SearchModule):
    """Valid PNG URLs are accepted."""
    assert SearchModule._is_product_image("https://example.com/wine.png")


def test_is_product_image_accepts_valid_webp(search_module: SearchModule):
    """Valid WebP URLs are accepted."""
    assert SearchModule._is_product_image("https://example.com/wine.webp")


def test_is_product_image_accepts_image_in_url(search_module: SearchModule):
    """URLs containing 'image' are accepted even without extension."""
    assert SearchModule._is_product_image("https://example.com/image/wine123")


def test_is_product_image_rejects_logo(search_module: SearchModule):
    """Logo images are rejected."""
    assert not SearchModule._is_product_image("https://example.com/logo.png")


def test_is_product_image_rejects_icon(search_module: SearchModule):
    """Icon images are rejected."""
    assert not SearchModule._is_product_image("https://example.com/icon.jpg")


def test_is_product_image_rejects_sprite(search_module: SearchModule):
    """Sprite images are rejected."""
    assert not SearchModule._is_product_image("https://example.com/sprite.png")


def test_is_product_image_rejects_tracking_pixels(search_module: SearchModule):
    """Tracking pixels are rejected."""
    assert not SearchModule._is_product_image("https://example.com/pixel.gif")
    assert not SearchModule._is_product_image("https://example.com/1x1.png")


def test_is_product_image_rejects_badge(search_module: SearchModule):
    """Badge images are rejected."""
    assert not SearchModule._is_product_image("https://example.com/badge.png")


def test_is_product_image_handles_query_params(search_module: SearchModule):
    """Query parameters don't break extension detection."""
    assert SearchModule._is_product_image("https://example.com/wine.jpg?width=200&height=300")


# ------------------------------------------------------------------
# Fallback Chain Tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ddg_called_first(search_module: SearchModule, sample_sku: SKU):
    """DDG is always the first source queried (Req 2.1)."""
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
async def test_retailers_called_when_ddg_empty(search_module: SearchModule, sample_sku: SKU):
    """When DDG returns empty, retailers are called next (Req 2.1)."""
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
    """When retailers also return empty, producer site is tried (Req 2.1)."""
    producer_candidates = [CandidateImage(url="https://producer.com/bottle.jpg", source="producer")]

    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock, return_value=producer_candidates):

        result = await search_module.search(sample_sku)

        assert len(result) == 1
        assert result[0].source == "producer"


@pytest.mark.asyncio
async def test_no_image_when_all_sources_empty(search_module: SearchModule, sample_sku: SKU):
    """When all sources return empty, result is empty list — 'No Image' (Req 2.4)."""
    with patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_retailers", new_callable=AsyncMock, return_value=[]), \
         patch.object(search_module, "_search_producer_site", new_callable=AsyncMock, return_value=[]):

        result = await search_module.search(sample_sku)

        assert result == []


# ------------------------------------------------------------------
# Retailer Scraping Tests (Req 2.1)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_retailers_constructs_correct_urls(search_module: SearchModule, sample_sku: SKU):
    """Retailer search constructs Vivino and Wine-Searcher URLs correctly."""
    with patch.object(search_module, "_scrape_retailer", new_callable=AsyncMock, return_value=[]) as mock_scrape:
        await search_module._search_retailers(sample_sku)
        
        # Check that both retailers were called with correct URL patterns
        assert mock_scrape.call_count == 2
        calls = mock_scrape.call_args_list
        
        # First call should be Vivino
        vivino_call = calls[0]
        assert "vivino.com/search/wines" in vivino_call[0][0]
        assert "Domaine+Leflaive" in vivino_call[0][0]
        assert vivino_call[0][2] == "vivino"
        
        # Second call should be Wine-Searcher
        ws_call = calls[1]
        assert "wine-searcher.com/find" in ws_call[0][0]
        assert "Domaine+Leflaive" in ws_call[0][0]
        assert ws_call[0][2] == "wine-searcher"


@pytest.mark.asyncio
async def test_search_retailers_combines_results(search_module: SearchModule, sample_sku: SKU):
    """Retailer search combines results from both Vivino and Wine-Searcher."""
    vivino_candidates = [CandidateImage(url="https://vivino.com/img1.jpg", source="vivino")]
    ws_candidates = [CandidateImage(url="https://ws.com/img2.jpg", source="wine-searcher")]
    
    async def mock_scrape(url, sku, source):
        if "vivino" in url:
            return vivino_candidates
        return ws_candidates
    
    with patch.object(search_module, "_scrape_retailer", new_callable=AsyncMock, side_effect=mock_scrape):
        result = await search_module._search_retailers(sample_sku)
        
        assert len(result) == 2
        assert result[0].source == "vivino"
        assert result[1].source == "wine-searcher"


@pytest.mark.asyncio
async def test_scrape_retailer_handles_playwright_not_installed(search_module: SearchModule, sample_sku: SKU):
    """Scrape retailer gracefully handles Playwright not being installed."""
    with patch.dict("sys.modules", {"playwright.async_api": None}):
        # The import will fail, but we test the exception handling
        result = await search_module._scrape_retailer(
            "https://www.vivino.com/search/wines?q=test",
            sample_sku,
            "vivino"
        )
        # Should return empty list when Playwright is not available
        assert result == []


@pytest.mark.asyncio
async def test_scrape_retailer_returns_empty_on_error(search_module: SearchModule, sample_sku: SKU):
    """Scrape retailer returns empty list on errors."""
    with patch("playwright.async_api.async_playwright") as mock_pw:
        # Create an async mock that raises an exception
        async def raise_error():
            raise Exception("Browser failed to launch")
        
        mock_pw.return_value = raise_error()
        
        result = await search_module._scrape_retailer(
            "https://www.vivino.com/search/wines?q=test",
            sample_sku,
            "vivino"
        )
        
        assert result == []


# ------------------------------------------------------------------
# Producer Site Search Tests (Req 2.1)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_producer_site_constructs_site_query(search_module: SearchModule):
    """Producer site search constructs site-restricted DuckDuckGo query."""
    sku = SKU(
        id="SKU005",
        producer="Château Margaux",
        appellation="Margaux",
        cru_vineyard=None,
        vintage=2018,
        format="750ml",
        region="Bordeaux",
    )
    
    with patch.object(search_module._ddgs, "images", return_value=[]) as mock_images:
        await search_module._search_producer_site(sku)
        
        # Check the query contains site: restriction
        call_args = mock_images.call_args[0][0]
        assert "site:" in call_args
        assert "châteaumargaux.com" in call_args.lower()  # Accents are preserved
        assert "Margaux" in call_args
        assert "2018" in call_args
        assert "wine bottle" in call_args


@pytest.mark.asyncio
async def test_search_producer_site_includes_cru(search_module: SearchModule, sample_sku: SKU):
    """Producer site search includes cru in query when present."""
    with patch.object(search_module._ddgs, "images", return_value=[]) as mock_images:
        await search_module._search_producer_site(sample_sku)
        
        call_args = mock_images.call_args[0][0]
        assert "Les Pucelles" in call_args


@pytest.mark.asyncio
async def test_search_producer_site_returns_candidates(search_module: SearchModule, sample_sku: SKU):
    """Producer site search returns candidates from DDG results."""
    mock_results = [
        {"image": "https://producer.com/wine1.jpg"},
        {"image": "https://producer.com/wine2.jpg"},
    ]
    
    with patch.object(search_module._ddgs, "images", return_value=mock_results):
        result = await search_module._search_producer_site(sample_sku)
        
        assert len(result) == 2
        assert all(c.source == "producer" for c in result)


@pytest.mark.asyncio
async def test_search_producer_site_handles_ddg_error(search_module: SearchModule, sample_sku: SKU):
    """Producer site search returns empty list on DDG error."""
    with patch.object(search_module._ddgs, "images", side_effect=Exception("DDG failed")):
        result = await search_module._search_producer_site(sample_sku)
        
        assert result == []
