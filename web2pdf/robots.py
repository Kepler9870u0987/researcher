"""Robots.txt parser with per-host caching."""
from __future__ import annotations

import urllib.robotparser
from urllib.parse import urlparse

_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def _robots_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def _get_parser(url: str, user_agent: str) -> urllib.robotparser.RobotFileParser:
    """Get or create a cached RobotFileParser for the URL's host."""
    robots_url = _robots_url(url)
    if robots_url not in _cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception:
            # If we can't fetch robots.txt, allow everything
            rp.allow_all = True
        _cache[robots_url] = rp
    return _cache[robots_url]


def is_allowed(url: str, user_agent: str = "Web2PdfBot") -> bool:
    """Check if the URL is allowed by robots.txt for the given user agent."""
    try:
        rp = _get_parser(url, user_agent)
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # Fail open


def get_crawl_delay(url: str, user_agent: str = "Web2PdfBot") -> float | None:
    """Get Crawl-delay from robots.txt (seconds), or None if unspecified."""
    try:
        rp = _get_parser(url, user_agent)
        delay = rp.crawl_delay(user_agent)
        return float(delay) if delay is not None else None
    except Exception:
        return None


def clear_cache() -> None:
    """Clear the robots.txt parser cache."""
    _cache.clear()
