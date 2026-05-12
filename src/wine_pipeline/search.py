"""Search Module — multi-source image search with fallback chain.

Uses Google Custom Search API as the primary source for high-quality wine images.
Falls back to DuckDuckGo if Google quota exceeded, then retailer scraping.
"""

from __future__ import annotations

import logging
import os

import httpx
from ddgs import DDGS

from .models import CandidateImage, SKU

logger = logging.getLogger(__name__)

# Google Custom Search API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = os.getenv("GOOGLE_CX", "d1e5e7b8f4a6c4d8e")  # Default CX for image search


class SearchModule:
    """Finds candidate wine images from multiple web sources using a fallback chain.

    Fallback order: Google Custom Search → DuckDuckGo Images → Retailers → Producer site.
    If all sources return zero candidates, returns an empty list (indicating "No Image").
    """

    def __init__(self) -> None:
        self._ddgs = DDGS()
        self._http_client = httpx.AsyncClient(timeout=15.0)

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(sku: SKU) -> str:
        """Build a search query string from SKU metadata."""
        parts = [sku.producer, sku.appellation]
        # Only add cru if it's not already part of the appellation
        if sku.cru_vineyard and sku.cru_vineyard.lower() not in sku.appellation.lower():
            parts.append(sku.cru_vineyard)
        if sku.vintage is not None:
            parts.append(str(sku.vintage))
        parts.append(sku.region)
        parts.append("wine bottle")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Google Custom Search — first source (high quality results)
    # ------------------------------------------------------------------

    async def _search_google(self, sku: SKU) -> list[CandidateImage]:
        """Query Google Custom Search API for wine bottle images."""
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set — skipping Google search")
            return []

        query = self._build_query(sku)
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": query,
            "searchType": "image",
            "num": 10,
            "imgSize": "large",
            "imgType": "photo",
        }

        try:
            response = await self._http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Google API error: %s", exc)
            return []
        except Exception as exc:
            logger.error("Google search failed: %s", exc)
            return []

        candidates: list[CandidateImage] = []
        for item in data.get("items", []):
            img_url = item.get("link")
            if img_url:
                candidates.append(CandidateImage(url=img_url, source="google"))

        logger.info("Google returned %d candidates for SKU %s", len(candidates), sku.id)
        return candidates

    # ------------------------------------------------------------------
    # DuckDuckGo Images — fallback source (no API key needed)
    # ------------------------------------------------------------------

    async def _search_ddg(self, sku: SKU) -> list[CandidateImage]:
        """Query DuckDuckGo Images and return candidate image URLs."""
        query = self._build_query(sku)

        try:
            results = self._ddgs.images(query, max_results=10)
        except Exception as exc:
            logger.error("DuckDuckGo image search failed: %s", exc)
            return []

        candidates: list[CandidateImage] = []
        for item in results:
            url = item.get("image")
            if url:
                candidates.append(CandidateImage(url=url, source="duckduckgo"))

        logger.info("DuckDuckGo returned %d candidates for SKU %s", len(candidates), sku.id)
        return candidates

    # ------------------------------------------------------------------
    # Retailer scraping (Vivino, Wine-Searcher) — second source
    # ------------------------------------------------------------------

    async def _scrape_retailer(
        self, url: str, sku: SKU, source_name: str
    ) -> list[CandidateImage]:
        """Scrape a single retailer page for product images using Playwright."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning("Playwright not installed — skipping %s", source_name)
            return []

        candidates: list[CandidateImage] = []
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=20_000)

                images = await page.query_selector_all("img")
                for img in images:
                    src = await img.get_attribute("src")
                    if src and self._is_product_image(src):
                        full_url = src if src.startswith("http") else f"https:{src}"
                        candidates.append(
                            CandidateImage(url=full_url, source=source_name)
                        )

                await browser.close()
        except Exception as exc:
            logger.error("Playwright scraping failed for %s: %s", source_name, exc)

        logger.info(
            "%s returned %d candidates for SKU %s",
            source_name, len(candidates), sku.id,
        )
        return candidates

    @staticmethod
    def _is_product_image(url: str) -> bool:
        """Heuristic filter: keep URLs that look like product images."""
        url_lower = url.lower()
        skip_patterns = ("logo", "icon", "sprite", "pixel", "tracking", "1x1", "badge")
        if any(p in url_lower for p in skip_patterns):
            return False
        image_exts = (".jpg", ".jpeg", ".png", ".webp")
        return any(url_lower.split("?")[0].endswith(ext) for ext in image_exts) or "image" in url_lower

    async def _search_retailers(self, sku: SKU) -> list[CandidateImage]:
        """Scrape Vivino and Wine-Searcher for candidate images."""
        query = self._build_query(sku).replace(" ", "+")
        vivino_url = f"https://www.vivino.com/search/wines?q={query}"
        ws_url = f"https://www.wine-searcher.com/find/{query}"

        vivino_results = await self._scrape_retailer(vivino_url, sku, "vivino")
        ws_results = await self._scrape_retailer(ws_url, sku, "wine-searcher")

        return vivino_results + ws_results

    # ------------------------------------------------------------------
    # Producer website — third source
    # ------------------------------------------------------------------

    async def _search_producer_site(self, sku: SKU) -> list[CandidateImage]:
        """Search the producer's own website for product images via DDG site-restricted query."""
        producer_domain = sku.producer.lower().replace(" ", "").replace("'", "")
        query = f"site:{producer_domain}.com {sku.appellation}"
        if sku.cru_vineyard:
            query += f" {sku.cru_vineyard}"
        if sku.vintage is not None:
            query += f" {sku.vintage}"
        query += " wine bottle"

        try:
            results = self._ddgs.images(query, max_results=5)
        except Exception as exc:
            logger.error("Producer site search failed: %s", exc)
            return []

        candidates: list[CandidateImage] = []
        for item in results:
            url = item.get("image")
            if url:
                candidates.append(CandidateImage(url=url, source="producer"))

        logger.info(
            "Producer site returned %d candidates for SKU %s",
            len(candidates), sku.id,
        )
        return candidates

    # ------------------------------------------------------------------
    # Fallback chain orchestration
    # ------------------------------------------------------------------

    async def search(self, sku: SKU) -> list[CandidateImage]:
        """Execute the full fallback chain: Google → DDG → Retailers → Producer site."""
        # Try Google first (highest quality)
        candidates = await self._search_google(sku)
        if candidates:
            return candidates

        # Fall back to DuckDuckGo
        candidates = await self._search_ddg(sku)
        if candidates:
            return candidates

        candidates = await self._search_retailers(sku)
        if candidates:
            return candidates

        candidates = await self._search_producer_site(sku)
        if candidates:
            return candidates

        logger.warning("All sources exhausted for SKU %s — No Image", sku.id)
        return []
