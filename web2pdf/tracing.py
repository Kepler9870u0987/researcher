"""Tracing: structured JSONL logging and Rich console progress."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


class Tracer:
    """JSONL event logger with optional Rich console output."""

    def __init__(self, trace_path: Path, quiet: bool = False) -> None:
        self.trace_path = trace_path
        self.quiet = quiet
        self._console = Console() if _HAS_RICH and not quiet else None
        self._progress: Progress | None = None
        self._task_id: Any = None
        self._total: int = 0
        self._done: int = 0
        self._errors: int = 0

    def log(self, event: str, **payload: Any) -> None:
        """Write a structured event to the JSONL trace file and optionally print."""
        rec = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **payload,
        }
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if self._console and not self._progress:
            style = "red" if "error" in event else "green" if "saved" in event else "dim"
            msg = f"[{event}] " + " ".join(f"{k}={v}" for k, v in payload.items() if k != "ts")
            self._console.print(msg, style=style)

    def start_progress(self, total_estimate: int) -> None:
        """Start a Rich progress bar (no-op if Rich not available)."""
        if not _HAS_RICH or self.quiet:
            return
        self._total = total_estimate
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("errors={task.fields[errors]}"),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._progress.start()
        self._task_id = self._progress.add_task("Crawling", total=total_estimate, errors=0)

    def advance(self, error: bool = False) -> None:
        """Advance the progress bar by 1."""
        self._done += 1
        if error:
            self._errors += 1
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=1, errors=self._errors)

    def update_total(self, new_total: int) -> None:
        """Update the estimated total for the progress bar."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, total=new_total)

    def stop_progress(self) -> None:
        """Stop the progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def summary(self, pages: int, errors: int, elapsed_s: float, out_dir: str) -> None:
        """Print and log final summary."""
        self.log("done", pages=pages, errors=errors, elapsed_s=round(elapsed_s, 1), out=out_dir)
        if self._console:
            self._console.print(
                f"\n[bold green]Done![/] {pages} pages saved, {errors} errors, "
                f"{elapsed_s:.1f}s elapsed → {out_dir}",
            )
