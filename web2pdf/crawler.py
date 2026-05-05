"""BFS crawler with async worker pool and rate limiting."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from web2pdf.config import Config
from web2pdf.renderer import PdfRenderer
from web2pdf.tracing import Tracer
from web2pdf.urls import (
    canonical_key,
    extract_links,
    is_allowed_url,
    normalize_url,
    url_to_filename,
    url_to_mirror_path,
)


class Crawler:
    """Async BFS crawler that saves each page as PDF."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.start_url = normalize_url(config.start_url)
        self.root_netloc = urlparse(self.start_url).netloc

        # Run directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        host_slug = self.root_netloc.replace(":", "_")
        self.run_dir = config.output_dir / f"{ts}__{host_slug}"
        self.pdf_dir = self.run_dir / "pdfs"
        self.meta_dir = self.run_dir / "meta"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.tracer = Tracer(self.run_dir / "trace.jsonl")
        self.index_path = self.run_dir / "index.json"

        # State
        self.index: dict[str, dict] = {}  # url -> {file, depth, title, status, bytes, parent}
        self.visited: set[str] = set()  # canonical keys
        self._errors: int = 0
        self._rate_delay: float = 1.0 / config.rate_limit_rps

    async def run(self) -> Path:
        """Execute the crawl. Returns the run directory path."""
        t0 = time.time()
        self.tracer.log(
            "start",
            url=self.start_url,
            depth=self.cfg.depth,
            concurrency=self.cfg.concurrency,
            same_domain_only=self.cfg.same_domain_only,
            out=str(self.run_dir),
        )

        if self.cfg.dry_run:
            await self._dry_run()
            return self.run_dir

        # Check robots.txt
        if self.cfg.respect_robots:
            try:
                from web2pdf.robots import is_allowed
                if not is_allowed(self.start_url, self.cfg.user_agent):
                    self.tracer.log("blocked_by_robots", url=self.start_url)
                    self.tracer.summary(0, 0, time.time() - t0, str(self.run_dir))
                    return self.run_dir
            except Exception:
                pass

        # BFS with worker pool
        queue: asyncio.Queue[tuple[str, int, str | None]] = asyncio.Queue()
        queue.put_nowait((self.start_url, 0, None))
        self.visited.add(canonical_key(self.start_url))

        self.tracer.start_progress(self.cfg.max_pages)

        async with PdfRenderer(self.cfg) as renderer:
            workers = [
                asyncio.create_task(self._worker(f"w{i}", queue, renderer))
                for i in range(self.cfg.concurrency)
            ]
            # Wait for queue to drain
            await queue.join()
            # Signal workers to stop
            for _ in workers:
                queue.put_nowait(("__STOP__", -1, None))
            await asyncio.gather(*workers, return_exceptions=True)

        self.tracer.stop_progress()
        elapsed = time.time() - t0

        # Write outputs
        self._write_index()
        self._write_sitemap()
        self.tracer.summary(len(self.index), self._errors, elapsed, str(self.run_dir))

        # Optional merge
        if self.cfg.merge_pdf:
            self._merge_pdfs()

        return self.run_dir

    async def _worker(
        self,
        name: str,
        queue: asyncio.Queue[tuple[str, int, str | None]],
        renderer: PdfRenderer,
    ) -> None:
        """Worker coroutine: dequeue URLs, render PDF, enqueue child links."""
        page = await renderer.new_page()
        try:
            while True:
                url, depth, parent = await queue.get()
                if url == "__STOP__":
                    queue.task_done()
                    break

                # Rate limit
                await asyncio.sleep(self._rate_delay)

                # Check limits
                if len(self.index) >= self.cfg.max_pages:
                    queue.task_done()
                    continue

                # Robots check per-URL
                if self.cfg.respect_robots:
                    try:
                        from web2pdf.robots import is_allowed
                        if not is_allowed(url, self.cfg.user_agent):
                            self.tracer.log("skipped_robots", url=url)
                            self.tracer.advance(error=False)
                            queue.task_done()
                            continue
                    except Exception:
                        pass

                # Navigate
                try:
                    await renderer.navigate(page, url)
                except Exception as e:
                    self.tracer.log("nav_error", url=url, depth=depth, error=str(e)[:200])
                    self.tracer.advance(error=True)
                    self._errors += 1
                    queue.task_done()
                    continue

                # Render PDF
                if self.cfg.mirror_paths:
                    rel_path = url_to_mirror_path(url)
                    pdf_path = self.pdf_dir / rel_path
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    fname = rel_path
                else:
                    fname = url_to_filename(url)
                    pdf_path = self.pdf_dir / fname

                try:
                    title = await renderer.get_title(page)
                    size = await renderer.render_pdf(page, url, pdf_path)
                    self.index[url] = {
                        "file": fname,
                        "depth": depth,
                        "title": title,
                        "status": "ok",
                        "bytes": size,
                        "parent_url": parent,
                        "fetched_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    self.tracer.log("pdf_saved", url=url, file=fname, depth=depth, bytes=size)
                    self.tracer.advance(error=False)
                except Exception as e:
                    self.tracer.log("pdf_error", url=url, error=str(e)[:200])
                    self.tracer.advance(error=True)
                    self._errors += 1
                    queue.task_done()
                    continue

                # Extract and enqueue links
                if depth < self.cfg.depth:
                    try:
                        html = await renderer.get_html(page)
                        links = extract_links(html, url)
                        enqueued = 0
                        for link in links:
                            key = canonical_key(link)
                            if key in self.visited:
                                continue
                            if not is_allowed_url(
                                link,
                                self.root_netloc,
                                self.cfg.same_domain_only,
                                self.cfg.include_subdomains,
                                self.cfg.allow_patterns or None,
                                self.cfg.deny_patterns or None,
                            ):
                                continue
                            self.visited.add(key)
                            queue.put_nowait((link, depth + 1, url))
                            enqueued += 1
                        if enqueued:
                            self.tracer.log("links_extracted", url=url, count=enqueued)
                            self.tracer.update_total(
                                min(len(self.visited), self.cfg.max_pages)
                            )
                    except Exception as e:
                        self.tracer.log("html_error", url=url, error=str(e)[:200])

                queue.task_done()
        finally:
            await page.close()

    async def _dry_run(self) -> None:
        """BFS without rendering — just discover and print URLs."""
        from web2pdf.urls import extract_links, is_allowed_url, canonical_key
        from playwright.async_api import async_playwright

        queue: list[tuple[str, int]] = [(self.start_url, 0)]
        self.visited.add(canonical_key(self.start_url))
        discovered: list[tuple[str, int]] = []

        async with PdfRenderer(self.cfg) as renderer:
            page = await renderer.new_page()
            while queue and len(discovered) < self.cfg.max_pages:
                url, depth = queue.pop(0)
                discovered.append((url, depth))
                print(f"[depth={depth}] {url}")

                if depth >= self.cfg.depth:
                    continue
                try:
                    await renderer.navigate(page, url)
                    html = await renderer.get_html(page)
                    for link in extract_links(html, url):
                        key = canonical_key(link)
                        if key in self.visited:
                            continue
                        if not is_allowed_url(
                            link, self.root_netloc,
                            self.cfg.same_domain_only, self.cfg.include_subdomains,
                            self.cfg.allow_patterns or None, self.cfg.deny_patterns or None,
                        ):
                            continue
                        self.visited.add(key)
                        queue.append((link, depth + 1))
                except Exception as e:
                    self.tracer.log("dry_nav_error", url=url, error=str(e)[:200])
            await page.close()

        self.tracer.log("dry_run_done", urls_found=len(discovered))
        print(f"\n--- Dry run: {len(discovered)} URLs found ---")

    def _write_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _write_sitemap(self) -> None:
        """Generate a navigable HTML index of all saved PDFs."""
        rows = ""
        for url, info in sorted(self.index.items(), key=lambda x: x[1]["depth"]):
            title = info.get("title") or url
            fname = info["file"]
            rows += f'  <tr><td>{info["depth"]}</td><td><a href="pdfs/{fname}">{title}</a></td><td><code>{url}</code></td></tr>\n'

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Web2PDF Sitemap</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}}th{{background:#f5f5f5}}</style>
</head><body>
<h1>Web2PDF Crawl — {self.root_netloc}</h1>
<p>{len(self.index)} pages saved</p>
<table><tr><th>Depth</th><th>Title / PDF</th><th>URL</th></tr>
{rows}</table>
</body></html>"""
        (self.run_dir / "sitemap.html").write_text(html, encoding="utf-8")

    def _merge_pdfs(self) -> None:
        """Merge all PDFs into a single file with bookmarks."""
        try:
            from pypdf import PdfWriter
        except ImportError:
            self.tracer.log("merge_skip", reason="pypdf not installed")
            return

        writer = PdfWriter()
        for url, info in sorted(self.index.items(), key=lambda x: x[1]["depth"]):
            pdf_path = self.pdf_dir / info["file"]
            if not pdf_path.exists():
                continue
            title = info.get("title") or url
            writer.append(str(pdf_path), outline_item=title)

        merged_path = self.run_dir / "merged.pdf"
        with open(merged_path, "wb") as f:
            writer.write(f)
        self.tracer.log("merged", file=str(merged_path), pages=len(self.index))
