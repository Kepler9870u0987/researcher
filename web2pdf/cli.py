"""CLI entry point for web2pdf: crawl, replay, merge sub-commands."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from web2pdf.config import Config
from web2pdf.crawler import Crawler


def _add_crawl_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("url", help="Target URL to start crawling from")
    p.add_argument("--depth", type=int, default=2, help="BFS depth (default: 2)")
    p.add_argument("--out", type=Path, default=Path("runs_pdf"), help="Output directory")
    p.add_argument("--concurrency", type=int, default=4, help="Parallel workers (default: 4)")
    p.add_argument("--rate", type=float, default=2.0, help="Max requests/sec (default: 2)")
    p.add_argument("--max-pages", type=int, default=200, help="Max pages to download")
    p.add_argument("--all-domains", action="store_true", help="Don't restrict to same domain")
    p.add_argument("--include-subdomains", action="store_true", help="Allow subdomains")
    p.add_argument("--allow", action="append", default=[], help="Regex: only crawl matching URLs")
    p.add_argument("--deny", action="append", default=[], help="Regex: skip matching URLs")
    p.add_argument(
        "--wait",
        choices=["load", "domcontentloaded", "networkidle", "commit"],
        default="networkidle",
        help="Playwright wait strategy",
    )
    p.add_argument("--timeout", type=int, default=30, help="Navigation timeout (seconds)")
    p.add_argument("--user-agent", default="Web2PdfBot/1.0", help="User-Agent header")
    p.add_argument("--no-robots", action="store_true", help="Ignore robots.txt")
    p.add_argument("--merge", action="store_true", help="Merge all PDFs into one file")
    p.add_argument("--mirror-paths", action="store_true", help="Mirror URL paths on filesystem")
    p.add_argument("--print-css", type=str, default=None, help="Extra CSS for @media print")
    p.add_argument("--dry-run", action="store_true", help="Discover URLs without saving PDFs")
    p.add_argument("--config", type=Path, default=None, help="Load config from YAML/JSON file")


def _build_config_from_args(args: argparse.Namespace) -> Config:
    """Build a Config object from CLI args, optionally merging a config file."""
    overrides: dict = {}
    if args.config and args.config.exists():
        text = args.config.read_text(encoding="utf-8")
        if args.config.suffix in (".yaml", ".yml"):
            try:
                import yaml
                overrides = yaml.safe_load(text) or {}
            except ImportError:
                print("Warning: PyYAML not installed, ignoring --config YAML file.", file=sys.stderr)
        else:
            overrides = json.loads(text)

    return Config(
        start_url=overrides.get("start_url", args.url),
        depth=overrides.get("depth", args.depth),
        max_pages=overrides.get("max_pages", args.max_pages),
        same_domain_only=overrides.get("same_domain_only", not args.all_domains),
        include_subdomains=overrides.get("include_subdomains", args.include_subdomains),
        allow_patterns=overrides.get("allow_patterns", args.allow),
        deny_patterns=overrides.get("deny_patterns", args.deny),
        wait_until=overrides.get("wait_until", args.wait),
        nav_timeout_ms=overrides.get("nav_timeout_ms", args.timeout * 1000),
        concurrency=overrides.get("concurrency", args.concurrency),
        rate_limit_rps=overrides.get("rate_limit_rps", args.rate),
        user_agent=overrides.get("user_agent", args.user_agent),
        output_dir=Path(overrides.get("output_dir", str(args.out))),
        merge_pdf=overrides.get("merge_pdf", args.merge),
        respect_robots=overrides.get("respect_robots", not args.no_robots),
        mirror_paths=overrides.get("mirror_paths", args.mirror_paths),
        print_css=overrides.get("print_css", args.print_css),
        dry_run=overrides.get("dry_run", args.dry_run),
    )


def cmd_crawl(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    crawler = Crawler(cfg)
    asyncio.run(crawler.run())
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Regenerate sitemap.html from an existing run's index.json."""
    run_dir = Path(args.run_dir)
    index_path = run_dir / "index.json"
    if not index_path.exists():
        print(f"Error: {index_path} not found.", file=sys.stderr)
        return 1

    index = json.loads(index_path.read_text(encoding="utf-8"))
    from urllib.parse import urlparse

    # Detect root netloc from first URL
    first_url = next(iter(index.keys()), "")
    root_netloc = urlparse(first_url).netloc if first_url else "unknown"

    rows = ""
    for url, info in sorted(index.items(), key=lambda x: x[1].get("depth", 0)):
        title = info.get("title") or url
        fname = info.get("file", "")
        depth = info.get("depth", "?")
        rows += f'  <tr><td>{depth}</td><td><a href="pdfs/{fname}">{title}</a></td><td><code>{url}</code></td></tr>\n'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Web2PDF Sitemap</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}}th{{background:#f5f5f5}}</style>
</head><body>
<h1>Web2PDF Crawl — {root_netloc}</h1>
<p>{len(index)} pages</p>
<table><tr><th>Depth</th><th>Title / PDF</th><th>URL</th></tr>
{rows}</table>
</body></html>"""
    out_path = run_dir / "sitemap.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Sitemap regenerated: {out_path}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge PDFs from an existing run directory."""
    run_dir = Path(args.run_dir)
    index_path = run_dir / "index.json"
    if not index_path.exists():
        print(f"Error: {index_path} not found.", file=sys.stderr)
        return 1

    try:
        from pypdf import PdfWriter
    except ImportError:
        print("Error: pypdf is required for merge. Install with: pip install pypdf", file=sys.stderr)
        return 1

    index = json.loads(index_path.read_text(encoding="utf-8"))
    pdf_dir = run_dir / "pdfs"
    writer = PdfWriter()

    for url, info in sorted(index.items(), key=lambda x: x[1].get("depth", 0)):
        pdf_path = pdf_dir / info["file"]
        if pdf_path.exists():
            title = info.get("title") or url
            writer.append(str(pdf_path), outline_item=title)

    merged_path = run_dir / "merged.pdf"
    with open(merged_path, "wb") as f:
        writer.write(f)
    print(f"Merged PDF: {merged_path} ({len(index)} pages)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="web2pdf",
        description="Crawl websites and save each page as PDF (Playwright/Chromium).",
    )
    sub = parser.add_subparsers(dest="command")

    # crawl (default)
    crawl_p = sub.add_parser("crawl", help="Crawl and save pages as PDF")
    _add_crawl_args(crawl_p)

    # replay
    replay_p = sub.add_parser("replay", help="Regenerate sitemap from existing run")
    replay_p.add_argument("run_dir", help="Path to existing run directory")

    # merge
    merge_p = sub.add_parser("merge", help="Merge PDFs from existing run into one file")
    merge_p.add_argument("run_dir", help="Path to existing run directory")

    args = parser.parse_args()

    # Default to crawl if URL provided without sub-command
    if args.command is None:
        # Re-parse with crawl as default
        _add_crawl_args(parser)
        args = parser.parse_args()
        if not hasattr(args, "url"):
            parser.print_help()
            return 1
        return cmd_crawl(args)

    dispatch = {"crawl": cmd_crawl, "replay": cmd_replay, "merge": cmd_merge}
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
