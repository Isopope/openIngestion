"""Web Fetcher \u2014 crawls public websites to produce documents.

Requires:
    pip install playwright
    playwright install chromium
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from loguru import logger

from openingestion.document import FetchedDocument
from openingestion.fetcher.base import BaseFetcher

try:
    from playwright.sync_api import sync_playwright, BrowserContext, TimeoutError as PlaywrightTimeout
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


class WebFetcher(BaseFetcher):
    """Fetches public websites by rendering them to PDFs using Playwright.
    
    This preserves the visual layout, tables, and images of the web page, 
    making it perfectly compatible with layout-aware Chefs (MinerU, Docling).
    """

    def __init__(
        self, 
        output_dir: str | os.PathLike, 
        mode: Literal["pdf", "html"] = "pdf",
        headless: bool = True
    ) -> None:
        if not _PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "playwright is required for WebFetcher.\n"
                "Install with: pip install playwright && playwright install chromium"
            )
            
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.headless = headless
        
        self.user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )

    def fetch(
        self,
        path: str | os.PathLike | None = None,
        dir: str | os.PathLike | None = None,
        **kwargs,
    ) -> list[FetchedDocument]:
        """Fetch documents from a URL or list of URLs."""
        urls = kwargs.get("urls", [])
        if path and isinstance(path, str) and path.startswith("http"):
            urls.append(path)
            
        if not urls:
            raise ValueError("WebFetcher requires at least one URL (pass via `path` or `urls`).")

        wait_for = kwargs.get("wait_for", "networkidle")
        fetched_docs = []

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process",
                ]
            )
            
            context = browser.new_context(
                user_agent=self.user_agent,
                viewport={"width": 1440, "height": 900},
                ignore_https_errors=True,
            )

            for url in urls:
                try:
                    doc = self._scrape_url(context, url, wait_for=wait_for)
                    docs = [doc] if doc else []
                    fetched_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Failed to fetch {url}: {e}")

            context.close()
            browser.close()

        return fetched_docs

    def _scrape_url(self, context: BrowserContext, url: str, wait_for: str) -> FetchedDocument | None:
        page = context.new_page()
        try:
            logger.info("WebFetcher: Navigating to {}", url)
            response = page.goto(url, wait_until="commit", timeout=30000)
            page.wait_for_timeout(2000)
            
            if wait_for == "networkidle":
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except PlaywrightTimeout:
                    pass
            
            if response and response.status >= 400:
                logger.warning("WebFetcher: HTTP {} when fetching {}", response.status, url)
                
            parsed_url = urlparse(url)
            safe_name = parsed_url.netloc + parsed_url.path.replace("/", "_")
            if not safe_name.endswith(f".{self.mode}"):
                safe_name += f".{self.mode}"
                
            out_path = self.output_dir / safe_name
            
            if self.mode == "pdf":
                # Print to PDF to feed standard Chefs like MinerU
                page.emulate_media(media="screen")
                page.pdf(path=out_path, format="A4", print_background=True)
                mime = "application/pdf"
            else:
                content = page.content()
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(content)
                mime = "text/html"
                
            logger.info("WebFetcher: Saved {} to {}", url, out_path)
            return FetchedDocument(source=url, path=out_path, mime_type=mime)
        finally:
            page.close()
