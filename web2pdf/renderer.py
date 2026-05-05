"""PDF renderer using Playwright (Chromium headless)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from web2pdf.config import Config


# JS snippet to auto-scroll and trigger lazy-load
_SCROLL_SCRIPT = """
async () => {
    const delay = ms => new Promise(r => setTimeout(r, ms));
    const step = Math.max(document.documentElement.clientHeight, 400);
    let y = 0;
    const maxY = document.body.scrollHeight;
    while (y < maxY) {
        y += step;
        window.scrollTo(0, y);
        await delay(150);
    }
    window.scrollTo(0, 0);
    await delay(300);
}
"""


class PdfRenderer:
    """Manages a Playwright browser context for rendering pages to PDF."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._pw = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> "PdfRenderer":
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = await self._browser.new_context(
            user_agent=self._config.user_agent,
            viewport={"width": 1280, "height": 900},
            ignore_https_errors=True,
        )
        # Mask navigator.webdriver and other headless signals
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = {runtime: {}};
        """)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    async def new_page(self) -> Page:
        """Create a new page in the shared context."""
        assert self._context is not None
        page = await self._context.new_page()
        page.set_default_navigation_timeout(self._config.nav_timeout_ms)
        return page

    async def navigate(self, page: Page, url: str) -> bool:
        """Navigate to URL with retry and fallback wait strategy. Returns True on success."""
        strategies = [self._config.wait_until]
        if self._config.wait_until == "networkidle":
            strategies.append("domcontentloaded")

        for strategy in strategies:
            try:
                await page.goto(url, wait_until=strategy)
                return True
            except Exception:
                if strategy == strategies[-1]:
                    raise
                continue
        return False

    async def render_pdf(self, page: Page, url: str, out_path: Path) -> int:
        """Render the current page to PDF. Returns file size in bytes."""
        # Auto-scroll to trigger lazy-load
        try:
            await page.evaluate(_SCROLL_SCRIPT)
        except Exception:
            pass  # Non-critical

        # Inject custom print CSS if configured
        if self._config.print_css:
            await page.add_style_tag(content=f"@media print {{ {self._config.print_css} }}")

        # Emulate print media for better PDF output
        await page.emulate_media(media="print")

        await page.pdf(
            path=str(out_path),
            format="A4",
            print_background=True,
            margin={"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"},
        )
        return out_path.stat().st_size

    async def get_html(self, page: Page) -> str:
        """Get the current page HTML content."""
        return await page.content()

    async def get_title(self, page: Page) -> str:
        """Get the page title."""
        try:
            return await page.title() or ""
        except Exception:
            return ""
