"""URL normalization, scope filtering, and link utilities."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

try:
    import tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False


SKIP_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".bz2", ".xz",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac", ".ogg", ".webm",
    ".css", ".js", ".mjs", ".json", ".xml", ".rss", ".atom",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".exe", ".msi", ".dmg", ".deb", ".rpm",
    ".pdf",
}

_DEFAULT_PORTS = {"http": "80", "https": "443"}


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication: lowercase host, strip fragment,
    remove default port, strip trailing slash on non-root paths."""
    url, _ = urldefrag(url)
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.hostname.lower() if parsed.hostname else ""
    port = parsed.port
    if port and str(port) == _DEFAULT_PORTS.get(scheme):
        port = None
    if port:
        netloc = f"{netloc}:{port}"

    path = parsed.path or "/"
    # NB: do NOT strip trailing slash — many sites differentiate
    # /foo/ (directory index) vs /foo (resource).

    return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))


def canonical_key(url: str) -> str:
    """Return a deduplicated key for a URL (host + path + sorted query)."""
    norm = normalize_url(url)
    parsed = urlparse(norm)
    # Sort query params for canonical form
    if parsed.query:
        pairs = sorted(parsed.query.split("&"))
        query = "&".join(pairs)
    else:
        query = ""
    return f"{parsed.netloc}{parsed.path}?{query}" if query else f"{parsed.netloc}{parsed.path}"


def same_domain(url: str, root_netloc: str, include_subdomains: bool = False) -> bool:
    """Check whether url belongs to the same domain scope as root_netloc."""
    try:
        url_host = urlparse(url).hostname
        root_host = root_netloc.split(":")[0] if ":" in root_netloc else root_netloc
        if not url_host or not root_host:
            return False
        url_host = url_host.lower()
        root_host = root_host.lower()

        if not include_subdomains:
            return url_host == root_host

        # Subdomain matching via tldextract if available
        if _HAS_TLDEXTRACT:
            u = tldextract.extract(url_host)
            r = tldextract.extract(root_host)
            return u.domain == r.domain and u.suffix == r.suffix
        else:
            # Fallback: check registrable domain by suffix
            return url_host == root_host or url_host.endswith("." + root_host)
    except Exception:
        return False


def is_http(url: str) -> bool:
    return url.lower().startswith(("http://", "https://"))


def has_skip_extension(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in SKIP_EXTENSIONS)


def is_allowed_url(
    url: str,
    root_netloc: str,
    same_domain_only: bool = True,
    include_subdomains: bool = False,
    allow_patterns: list[str] | None = None,
    deny_patterns: list[str] | None = None,
) -> bool:
    """Apply full filter chain: http, scope, deny, allow, extensions."""
    if not is_http(url):
        return False
    if has_skip_extension(url):
        return False
    if same_domain_only and not same_domain(url, root_netloc, include_subdomains):
        return False
    # Deny patterns (reject if any match)
    if deny_patterns:
        for pat in deny_patterns:
            if re.search(pat, url):
                return False
    # Allow patterns (if specified, at least one must match)
    if allow_patterns:
        if not any(re.search(pat, url) for pat in allow_patterns):
            return False
    return True


def url_to_filename(url: str, max_len: int = 80) -> str:
    """Generate a filesystem-safe filename with short hash for collision avoidance."""
    parsed = urlparse(url)
    raw = (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_") or "index"
    slug = slug[:max_len]
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{h}.pdf"


def url_to_mirror_path(url: str) -> str:
    """Convert URL path to a mirrored filesystem path."""
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    if parsed.query:
        h = hashlib.sha1(parsed.query.encode()).hexdigest()[:6]
        path = f"{path}_q{h}"
    if not path.endswith(".pdf"):
        path += ".pdf"
    # Sanitize for filesystem
    path = re.sub(r"[<>:\"|?*]", "_", path)
    return path


def extract_links(
    html: str,
    base_url: str,
    max_links: int = 200,
    skip_elements: list[str] | None = None,
) -> list[str]:
    """Extract and resolve links from HTML content.

    Args:
        skip_elements: HTML tag names whose subtrees are excluded from link
                       extraction (e.g. ['header', 'footer', 'nav']).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Resolve <base href> if present
    base_tag = soup.find("base", href=True)
    if base_tag:
        base_url = urljoin(base_url, base_tag["href"])

    # Remove excluded subtrees in-place before scanning links
    if skip_elements:
        for tag_name in skip_elements:
            for el in soup.find_all(tag_name):
                el.decompose()

    out: list[str] = []
    for tag in soup.find_all(["a", "area"], href=True):
        if len(out) >= max_links:
            break
        href = tag["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        absolute = urljoin(base_url, href)
        if is_http(absolute):
            out.append(normalize_url(absolute))

    return out
