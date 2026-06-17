"""Search Module — multi-source image search.

Queries Vivino first, then Brave, Google CSE, Serper, Bing, and DuckDuckGo in
parallel, merges and de-duplicates results, then returns the combined pool for
downstream quality scoring (best candidate wins).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx
from ddgs import DDGS

from .models import CandidateImage, SKU

logger = logging.getLogger(__name__)

# Serper.dev (Google Images) credentials
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "") or os.getenv("SERP_API_KEY", "")

# Google Custom Search API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = os.getenv("GOOGLE_CX", "d1e5e7b8f4a6c4d8e")  # Default CX for image search

# Brave Search API (image search)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "") or os.getenv("BRAVE_SEARCH_API_KEY", "")

_BLOCKED_IMAGE_HOSTS = (
    "instagram.com",
    "lookaside.instagram.com",
    "facebook.com",
    "lookaside.fbsbx.com",
    "pinterest.",
    "tiktok.com",
    "x.com",
    "twitter.com",
)

# Obvious non-wine results from Bing/DDG when queries go wrong.
_JUNK_URL_PARTS = (
    "motorcycle",
    "apache-rtr",
    "apache_rtr",
    "cars24",
    "bikedekho",
    "motonews",
    "somosmoto",
    "qurban",
    "pixabay.com/photo/2013",
    "walmartimages",
    "letter-146007",
    "fireboldweb.com",
)


class SearchModule:
    """Finds candidate wine images from Brave, Google, Serper, Bing, and DDG.

    All configured sources are queried in parallel; results are interleaved and
    de-duplicated so the pipeline can pick the best passing image per SKU.
    """

    def __init__(self) -> None:
        self._ddgs = DDGS()
        self._http_client = httpx.AsyncClient(timeout=15.0)
        self._serper_lock = asyncio.Lock()
        self._last_serper_at = 0.0
        self._serper_min_interval = 1.0

    async def _throttle_serper(self) -> None:
        """Space Serper calls to avoid 429 rate limits."""
        async with self._serper_lock:
            elapsed = time.monotonic() - self._last_serper_at
            if elapsed < self._serper_min_interval:
                await asyncio.sleep(self._serper_min_interval - elapsed)
            self._last_serper_at = time.monotonic()

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    @staticmethod
    def _wine_display_name(sku: SKU) -> str:
        """Primary search text — mirrors how a human would Google the wine."""
        if sku.full_name:
            return sku.full_name.strip()
        if sku.cru_vineyard:
            return sku.cru_vineyard.strip()
        return " ".join(p for p in (sku.producer, sku.appellation) if p).strip()

    @staticmethod
    def _build_simple_query(sku: SKU) -> str:
        """Plain Google-style query: wine name + vintage (no extra keywords)."""
        name = SearchModule._wine_display_name(sku)
        parts = [name] if name else []
        if sku.vintage is not None:
            parts.append(str(sku.vintage))
        return " ".join(parts).strip()

    @staticmethod
    def _build_query(sku: SKU) -> str:
        """Extended query: full wine name + vintage + wine bottle."""
        simple = SearchModule._build_simple_query(sku)
        if not simple:
            return "wine bottle"
        return f"{simple} wine bottle"

    @staticmethod
    def _build_query_variants(sku: SKU) -> list[str]:
        """Ordered query variants — simple name first (matches manual Google search)."""
        variants: list[str] = []
        simple = SearchModule._build_simple_query(sku)
        if simple:
            variants.append(simple)

        full = SearchModule._build_query(sku)
        if full:
            variants.append(full)

        if sku.producer:
            producer_q = f"{sku.producer} {sku.vintage or ''} wine bottle".strip()
            variants.append(producer_q)

        # De-dupe while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for q in variants:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
        return out

    @staticmethod
    def _is_blocked_url(url: str) -> bool:
        lower = url.lower()
        return any(bad in lower for bad in _BLOCKED_IMAGE_HOSTS)

    @staticmethod
    def _filter_junk_urls(candidates: list[CandidateImage]) -> list[CandidateImage]:
        """Drop URLs that are obviously not wine product photos."""
        kept: list[CandidateImage] = []
        for candidate in candidates:
            lower = candidate.url.lower()
            if any(junk in lower for junk in _JUNK_URL_PARTS):
                continue
            if SearchModule._is_blocked_url(candidate.url):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _merge_candidates(*groups: list[CandidateImage]) -> list[CandidateImage]:
        """Interleave and de-duplicate candidates from multiple sources."""
        merged: list[CandidateImage] = []
        seen: set[str] = set()
        max_len = max((len(g) for g in groups), default=0)
        for i in range(max_len):
            for group in groups:
                if i >= len(group):
                    continue
                candidate = group[i]
                if candidate.url in seen:
                    continue
                seen.add(candidate.url)
                merged.append(candidate)
        return merged

    # ------------------------------------------------------------------
    # Brave Image Search
    # ------------------------------------------------------------------

    async def _search_brave(
        self, sku: SKU, limit: int, query: str | None = None
    ) -> list[CandidateImage]:
        """Query Brave Image Search API."""
        if not BRAVE_API_KEY:
            return []

        query = query or self._build_query(sku)
        url = "https://api.search.brave.com/res/v1/images/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_API_KEY,
        }
        params = {
            "q": query,
            "count": max(1, min(50, int(limit))),
            "search_lang": "en",
            "country": "ALL",
            "safesearch": "strict",
            "spellcheck": "true",
        }

        try:
            response = await self._http_client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Brave Image Search HTTP error: %s", exc)
            return []
        except Exception as exc:
            logger.error("Brave Image Search failed: %s", exc)
            return []

        candidates: list[CandidateImage] = []
        seen: set[str] = set()
        for item in data.get("results") or []:
            props = item.get("properties") or {}
            thumb = item.get("thumbnail") or {}
            img_url = (
                props.get("url")
                or props.get("src")
                or thumb.get("src")
                or item.get("url")
            )
            if not img_url:
                continue
            img_url = str(img_url)
            if img_url in seen:
                continue
            seen.add(img_url)
            if self._is_blocked_url(img_url):
                continue
            candidates.append(CandidateImage(url=img_url, source="brave"))

        filtered = self._filter_junk_urls(candidates)
        if filtered:
            logger.info(
                "Brave returned %d candidates for SKU %s (query=%r)",
                len(filtered), sku.id, query[:80],
            )
        return filtered[:limit]

    # ------------------------------------------------------------------
    # Serper.dev (Google Images) — best-effort high quality
    # ------------------------------------------------------------------

    async def _search_serper(
        self, sku: SKU, limit: int, query: str | None = None
    ) -> list[CandidateImage]:
        """Query Serper.dev Google Images for wine bottle images."""
        if not SERPER_API_KEY:
            return []

        query = query or self._build_query(sku)
        url = "https://google.serper.dev/images"
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": max(1, min(100, int(limit))),
        }

        await self._throttle_serper()
        try:
            response = await self._http_client.post(url, headers=headers, json=payload)
            if response.status_code == 429:
                logger.warning("Serper rate-limited for SKU %s", sku.id)
                return []
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("Serper rate-limited for SKU %s", sku.id)
            else:
                logger.error("Serper HTTP error: %s", exc)
            return []
        except Exception as exc:
            logger.error("Serper search failed: %s", exc)
            return []

        candidates: list[CandidateImage] = []
        seen: set[str] = set()
        for item in data.get("images") or []:
            img_url = (
                item.get("imageUrl")
                or item.get("link")
                or item.get("thumbnailUrl")
            )
            if not img_url:
                continue
            img_url = str(img_url)
            if self._is_blocked_url(img_url):
                continue
            if img_url in seen:
                continue
            seen.add(img_url)
            candidates.append(CandidateImage(url=img_url, source="serper"))

        filtered = self._filter_junk_urls(candidates)
        if filtered:
            logger.info(
                "Serper (Google Images) returned %d candidates for SKU %s (query=%r)",
                len(filtered), sku.id, query[:80],
            )
        return filtered[:limit]

    # ------------------------------------------------------------------
    # Vivino — primary source (best bottle photography)
    # ------------------------------------------------------------------

    @staticmethod
    def _vivino_sort_key(url: str) -> tuple[int, int]:
        """Prefer full product-bottle shots (_pb_) at highest resolution."""
        lower = url.lower()
        if "_pb_" in lower:
            if "x960" in lower or "960" in lower:
                return (0, 0)
            if "600" in lower:
                return (0, 1)
            return (0, 2)
        if "/labels/" in lower:
            return (2, 0)
        return (1, 0)

    async def _search_vivino(
        self, sku: SKU, limit: int, query: str | None = None
    ) -> list[CandidateImage]:
        """Find Vivino product-bottle images via Serper (query + 'vivino')."""
        try:
            return await asyncio.wait_for(
                self._search_vivino_impl(sku, limit, query),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Vivino search timed out for SKU %s", sku.id)
            return []

    def _vivino_candidates_from_urls(
        self, urls: list[str], limit: int
    ) -> list[CandidateImage]:
        candidates: list[CandidateImage] = []
        seen: set[str] = set()
        for img_url in urls:
            if "vivino.com" not in img_url.lower():
                continue
            if img_url in seen:
                continue
            seen.add(img_url)
            candidates.append(CandidateImage(url=img_url, source="vivino"))
        candidates.sort(key=lambda c: self._vivino_sort_key(c.url))
        return self._filter_junk_urls(candidates)[:limit]

    async def _search_vivino_via_brave(
        self, sku: SKU, limit: int, vivino_query: str
    ) -> list[CandidateImage]:
        """Fallback Vivino discovery when Serper credits are unavailable."""
        brave = await self._search_brave(sku, limit, vivino_query)
        urls = [c.url for c in brave]
        filtered = self._vivino_candidates_from_urls(urls, limit)
        if filtered:
            logger.info(
                "Vivino (Brave fallback) returned %d candidates for SKU %s (query=%r)",
                len(filtered),
                sku.id,
                vivino_query[:80],
            )
        return filtered

    async def _search_vivino_impl(
        self, sku: SKU, limit: int, query: str | None = None
    ) -> list[CandidateImage]:
        query = query or self._build_simple_query(sku)
        vivino_query = f"{query} vivino"

        if SERPER_API_KEY:
            url = "https://google.serper.dev/images"
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            }
            payload = {
                "q": vivino_query,
                "num": max(10, min(100, int(limit))),
            }

            await self._throttle_serper()
            try:
                response = await self._http_client.post(
                    url, headers=headers, json=payload
                )
                if response.status_code == 429:
                    logger.warning(
                        "Serper rate-limited for Vivino search SKU %s", sku.id
                    )
                elif response.status_code == 400:
                    logger.warning(
                        "Serper unavailable for Vivino search SKU %s — using Brave",
                        sku.id,
                    )
                else:
                    response.raise_for_status()
                    data = response.json()
                    urls = [
                        str(
                            item.get("imageUrl")
                            or item.get("link")
                            or item.get("thumbnailUrl")
                        )
                        for item in data.get("images") or []
                    ]
                    filtered = self._vivino_candidates_from_urls(urls, limit)
                    if filtered:
                        logger.info(
                            "Vivino returned %d candidates for SKU %s (query=%r)",
                            len(filtered),
                            sku.id,
                            vivino_query[:80],
                        )
                        return filtered
            except Exception as exc:
                logger.error(
                    "Vivino (Serper) search failed for SKU %s: %s", sku.id, exc
                )

        return await self._search_vivino_via_brave(sku, limit, vivino_query)

    # ------------------------------------------------------------------
    # Google Custom Search — first source (high quality results)
    # ------------------------------------------------------------------

    async def _search_google(
        self, sku: SKU, limit: int, query: str | None = None
    ) -> list[CandidateImage]:
        """Query Google Custom Search API for wine bottle images."""
        if not GOOGLE_API_KEY:
            return []

        query = query or self._build_query(sku)
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": query,
            "searchType": "image",
            "num": max(1, min(10, int(limit))),
            "imgSize": "large",
            "imgType": "photo",
        }

        try:
            response = await self._http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                err = exc.response.json().get("error", {})
                detail = err.get("message", "") or ""
            except Exception:
                pass
            if "does not have the access to Custom Search JSON API" in detail:
                logger.warning(
                    "Custom Search JSON API returned 403: this GCP project likely has no JSON API "
                    "entitlement (Google closed JSON API access for new projects; DDG/Bing fallback will "
                    "be used). See https://developers.google.com/custom-search/v1/overview "
                    "(Vertex AI Search or a third-party SERP API are alternatives)."
                )
            else:
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

        filtered = self._filter_junk_urls(candidates)
        if filtered:
            logger.info(
                "Google CSE returned %d candidates for SKU %s", len(filtered), sku.id
            )
        return filtered[:limit]

    def _search_ddg_sync(
        self, sku: SKU, limit: int, query: str | None = None
    ) -> list[CandidateImage]:
        """Query DuckDuckGo Images backend."""
        query = query or self._build_query(sku)

        results = []
        last_exc: Exception | None = None
        for backend in ("duckduckgo", "auto"):
            try:
                kwargs: dict = {"max_results": max(1, int(limit))}
                if backend != "auto":
                    kwargs["backend"] = backend
                results = self._ddgs.images(query, **kwargs)
                if results:
                    break
            except Exception as exc:
                last_exc = exc
                continue
        if not results and last_exc is not None:
            logger.error("DuckDuckGo image search failed: %s", last_exc)
            return []

        candidates: list[CandidateImage] = []
        for item in results:
            url = item.get("image")
            if url:
                candidates.append(CandidateImage(url=url, source="duckduckgo"))

        filtered = self._filter_junk_urls(candidates)
        if filtered:
            logger.info(
                "DuckDuckGo returned %d candidates for SKU %s", len(filtered), sku.id
            )
        return filtered[:limit]

    def _search_bing_sync(
        self, sku: SKU, limit: int, query: str | None = None
    ) -> list[CandidateImage]:
        """Query Bing Images via ddgs backend."""
        query = query or self._build_query(sku)

        try:
            results = self._ddgs.images(
                query, max_results=max(1, int(limit)), backend="bing"
            )
        except Exception as exc:
            logger.error("Bing image search failed: %s", exc)
            return []

        candidates: list[CandidateImage] = []
        for item in results:
            url = item.get("image")
            if url:
                candidates.append(CandidateImage(url=url, source="bing"))

        filtered = self._filter_junk_urls(candidates)
        if filtered:
            logger.info("Bing returned %d candidates for SKU %s", len(filtered), sku.id)
        return filtered[:limit]

    async def _search_ddg(
        self, sku: SKU, limit: int, query: str | None = None
    ) -> list[CandidateImage]:
        """Async wrapper for DuckDuckGo Images."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._search_ddg_sync,
                    sku,
                    limit,
                    query or self._build_query(sku),
                ),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            logger.warning("DuckDuckGo timed out for SKU %s", sku.id)
            return []

    async def _search_bing(self, sku: SKU, limit: int, query: str | None = None) -> list[CandidateImage]:
        """Async wrapper for Bing Images."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._search_bing_sync,
                    sku,
                    limit,
                    query or self._build_query(sku),
                ),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Bing timed out for SKU %s", sku.id)
            return []

    # ------------------------------------------------------------------
    # Retailer scraping (Vivino, Wine-Searcher)
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

    async def _search_retailers(self, sku: SKU, limit: int) -> list[CandidateImage]:
        """Scrape Vivino and Wine-Searcher for candidate images."""
        query = self._build_query(sku).replace(" ", "+")
        vivino_url = f"https://www.vivino.com/search/wines?q={query}"
        ws_url = f"https://www.wine-searcher.com/find/{query}"

        vivino_results = await self._scrape_retailer(vivino_url, sku, "vivino")
        ws_results = await self._scrape_retailer(ws_url, sku, "wine-searcher")

        return (vivino_results + ws_results)[:limit]

    # ------------------------------------------------------------------
    # Producer website — third source
    # ------------------------------------------------------------------

    async def _search_producer_site(self, sku: SKU, limit: int) -> list[CandidateImage]:
        """Search the producer's own website for product images via DDG site-restricted query."""
        producer_domain = sku.producer.lower().replace(" ", "").replace("'", "")
        query = f"site:{producer_domain}.com {sku.appellation}"
        if sku.cru_vineyard:
            query += f" {sku.cru_vineyard}"
        if sku.vintage is not None:
            query += f" {sku.vintage}"
        query += " wine bottle"

        try:
            results = self._ddgs.images(query, max_results=max(1, int(limit)))
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
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Multi-source search orchestration
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_results(result: list[CandidateImage] | BaseException) -> list[CandidateImage]:
        if isinstance(result, BaseException):
            logger.error("Search source raised: %s", result)
            return []
        return result

    async def _fetch_all_sources(
        self, sku: SKU, per_source: int, query: str
    ) -> tuple[list[CandidateImage], ...]:
        """Query Brave, Google CSE, Google Images (Serper), Bing, and DDG in parallel."""

        async def _serper_with_timeout() -> list[CandidateImage]:
            try:
                return await asyncio.wait_for(
                    self._search_serper(sku, per_source, query),
                    timeout=20.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Serper timed out for SKU %s", sku.id)
                return []

        brave_r, google_r, serper_r, bing_r, ddg_r = await asyncio.gather(
            self._search_brave(sku, per_source, query),
            self._search_google(sku, per_source, query),
            _serper_with_timeout(),
            self._search_bing(sku, per_source, query),
            self._search_ddg(sku, per_source, query),
            return_exceptions=True,
        )

        brave = self._coerce_results(brave_r)
        google = self._coerce_results(google_r)
        serper = self._coerce_results(serper_r)
        bing = self._coerce_results(bing_r)
        ddg = self._coerce_results(ddg_r)

        logger.info(
            "SKU %s query=%r — brave=%d google_cse=%d google_images=%d bing=%d ddg=%d",
            sku.id,
            query[:70],
            len(brave),
            len(google),
            len(serper),
            len(bing),
            len(ddg),
        )
        return brave, google, serper, bing, ddg

    async def search(self, sku: SKU, limit: int = 20) -> list[CandidateImage]:
        """Query all five sources for every query variant; merge and de-duplicate."""
        limit = max(1, int(limit))
        per_source = max(15, limit)

        if getattr(self, "_vivino_first_only", False):
            vivino_only: list[CandidateImage] = []
            seen_v: set[str] = set()
            for query in self._build_query_variants(sku):
                for candidate in await self._search_vivino(sku, per_source, query):
                    if candidate.url in seen_v:
                        continue
                    seen_v.add(candidate.url)
                    vivino_only.append(candidate)
            if len(vivino_only) >= min(5, limit):
                logger.info(
                    "Vivino-only pass returned %d candidates for SKU %s",
                    len(vivino_only),
                    sku.id,
                )
                return vivino_only[:limit]

        merged_all: list[CandidateImage] = []
        seen_urls: set[str] = set()

        for query in self._build_query_variants(sku):
            vivino = await self._search_vivino(sku, per_source, query)
            brave, google, serper, bing, ddg = await self._fetch_all_sources(
                sku, per_source, query
            )
            # Vivino first, then Google Images, then the rest.
            batch = self._merge_candidates(vivino, serper, google, brave, bing, ddg)
            for candidate in batch:
                if candidate.url in seen_urls:
                    continue
                seen_urls.add(candidate.url)
                merged_all.append(candidate)

        if merged_all:
            logger.info(
                "Merged %d total candidates for SKU %s (all sources, %d queries)",
                len(merged_all),
                sku.id,
                len(self._build_query_variants(sku)),
            )
            return merged_all[:limit]

        logger.warning(
            "All sources returned zero candidates for SKU %s after querying "
            "brave, google_cse, google_images, bing, and ddg",
            sku.id,
        )
        return []
