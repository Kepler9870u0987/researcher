"""web2pdf — Crawl websites and save each page as PDF via Playwright/Chromium."""

from web2pdf.config import Config
from web2pdf.crawler import Crawler

__all__ = ["Config", "Crawler"]
