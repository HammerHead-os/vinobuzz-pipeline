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

def test_build_simple_query_matches_manual_google(search_module: SearchModule, sample_sku: SKU):
    """Simple query is wine name + vintage only (no extra keywords)."""
    sample_sku.full_name = "Domaine de Bila-Haut Chrysopee Blanc, Collioure AOP"
    query = SearchModule._build_simple_query(sample_sku)
    assert query == "Domaine de Bila-Haut Chrysopee Blanc, Collioure AOP 2020"


def test_build_query_with_all_fields(search_module: SearchModule, sample_sku: SKU):
    """Query uses wine name + vintage + wine bottle (human-style Google search)."""
    sample_sku.full_name = "Domaine Leflaive Puligny-Montrachet Les Pucelles"
    query = SearchModule._build_query(sample_sku)
    assert "Domaine Leflaive Puligny-Montrachet Les Pucelles" in query
    assert "2020" in query
    assert "wine bottle" in query


def test_build_query_without_cru(search_module: SearchModule):
    """Query uses full_name when provided."""
    sku = SKU(
        id="SKU002",
        producer="Château Margaux",
        appellation="Margaux",
        cru_vineyard=None,
        vintage=2018,
        format="750ml",
        region="Bordeaux",
        full_name="Château Margaux Margaux",
    )
    query = SearchModule._build_query(sku)
    assert "Château Margaux Margaux" in query
    assert "2018" in query
    assert "wine bottle" in query


def test_build_query_without_vintage_nv_wine(search_module: SearchModule):
    """Query omits vintage for NV wines."""
    sku = SKU(
        id="SKU003",
        producer="Veuve Clicquot",
        appellation="Champagne",
        cru_vineyard=None,
        vintage=None,
        format="750ml",
        region="Champagne",
        full_name="Veuve Clicquot Champagne Brut",
    )
    query = SearchModule._build_query(sku)
    assert "Veuve Clicquot" in query
    assert "NV" not in query
    assert "wine bottle" in query


def test_build_query_falls_back_to_cru(search_module: SearchModule, sample_sku: SKU):
    """Without full_name, cru_vineyard is used."""
    query = SearchModule._build_query(sample_sku)
    assert "Les Pucelles" in query
    assert "2020" in query


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
# Multi-source search tests
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_sources_queried_every_search(search_module: SearchModule, sample_sku: SKU):
    """Brave, Google CSE, Google Images, Bing, and DDG are all queried."""
    brave = [CandidateImage(url="https://brave.example.com/1.jpg", source="brave")]
    google = [CandidateImage(url="https://google.example.com/1.jpg", source="google")]
    serper = [CandidateImage(url="https://serper.example.com/1.jpg", source="serper")]
    bing = [CandidateImage(url="https://bing.example.com/1.jpg", source="bing")]
    ddg = [CandidateImage(url="https://ddg.example.com/1.jpg", source="duckduckgo")]

    with patch.object(search_module, "_search_brave", new_callable=AsyncMock, return_value=brave) as mock_brave, \
         patch.object(search_module, "_search_google", new_callable=AsyncMock, return_value=google) as mock_google, \
         patch.object(search_module, "_search_serper", new_callable=AsyncMock, return_value=serper) as mock_serper, \
         patch.object(search_module, "_search_bing", new_callable=AsyncMock, return_value=bing) as mock_bing, \
         patch.object(search_module, "_search_ddg", new_callable=AsyncMock, return_value=ddg) as mock_ddg:

        result = await search_module.search(sample_sku, limit=20)

        assert mock_brave.await_count >= 1
        assert mock_google.await_count >= 1
        assert mock_serper.await_count >= 1
        assert mock_bing.await_count >= 1
        assert mock_ddg.await_count >= 1
        assert len(result) == 5
        assert {c.source for c in result} == {
            "serper", "google", "brave", "bing", "duckduckgo"
        }


@pytest.mark.asyncio
async def test_search_merges_multiple_query_variants(search_module: SearchModule, sample_sku: SKU):
    """Each query variant triggers a full five-source fetch."""
    sample_sku.full_name = "Domaine Leflaive Puligny-Montrachet Les Pucelles"

    with patch.object(
        search_module,
        "_fetch_all_sources",
        new_callable=AsyncMock,
        return_value=(
            [CandidateImage(url="https://a.example.com/1.jpg", source="brave")],
            [],
            [],
            [],
            [],
        ),
    ) as mock_fetch:
        await search_module.search(sample_sku, limit=20)

        assert mock_fetch.await_count == len(SearchModule._build_query_variants(sample_sku))


@pytest.mark.asyncio
async def test_no_image_when_all_sources_empty(search_module: SearchModule, sample_sku: SKU):
    """When all sources return empty, result is empty list — 'No Image'."""
    with patch.object(search_module, "_fetch_all_sources", new_callable=AsyncMock, return_value=([], [], [], [], [])):
        result = await search_module.search(sample_sku)

        assert result == []


# ------------------------------------------------------------------
# Legacy retailer / producer helpers (still used elsewhere)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_retailers_constructs_correct_urls(search_module: SearchModule, sample_sku: SKU):
    """Retailer search constructs Vivino and Wine-Searcher URLs correctly."""
    with patch.object(search_module, "_scrape_retailer", new_callable=AsyncMock, return_value=[]) as mock_scrape:
        await search_module._search_retailers(sample_sku, limit=20)
        
        # Check that both retailers were called with correct URL patterns
        assert mock_scrape.call_count == 2
        calls = mock_scrape.call_args_list
        
        # First call should be Vivino
        vivino_call = calls[0]
        assert "vivino.com/search/wines" in vivino_call[0][0]
        assert "Les+Pucelles" in vivino_call[0][0] or "2020" in vivino_call[0][0]
        assert vivino_call[0][2] == "vivino"

        ws_call = calls[1]
        assert "wine-searcher.com/find" in ws_call[0][0]
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
        result = await search_module._search_retailers(sample_sku, limit=20)
        
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
        await search_module._search_producer_site(sku, limit=20)
        
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
        await search_module._search_producer_site(sample_sku, limit=20)
        
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
        result = await search_module._search_producer_site(sample_sku, limit=20)
        
        assert len(result) == 2
        assert all(c.source == "producer" for c in result)


@pytest.mark.asyncio
async def test_search_producer_site_handles_ddg_error(search_module: SearchModule, sample_sku: SKU):
    """Producer site search returns empty list on DDG error."""
    with patch.object(search_module._ddgs, "images", side_effect=Exception("DDG failed")):
        result = await search_module._search_producer_site(sample_sku, limit=20)
        
        assert result == []
