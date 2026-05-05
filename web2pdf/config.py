"""Configuration model for web2pdf pipeline."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Config(BaseModel):
    """Validated configuration for a crawl run."""

    start_url: str
    depth: int = Field(default=2, ge=0, le=50)
    max_pages: int = Field(default=200, ge=1)
    same_domain_only: bool = True
    include_subdomains: bool = False
    allow_patterns: list[str] = Field(default_factory=list)
    deny_patterns: list[str] = Field(default_factory=list)
    wait_until: str = Field(default="networkidle")
    nav_timeout_ms: int = Field(default=30_000, ge=1000)
    concurrency: int = Field(default=4, ge=1, le=20)
    rate_limit_rps: float = Field(default=2.0, gt=0)
    user_agent: str = "Web2PdfBot/1.0"
    output_dir: Path = Field(default=Path("runs_pdf"))
    merge_pdf: bool = False
    respect_robots: bool = True
    mirror_paths: bool = False
    print_css: Optional[str] = None
    dry_run: bool = False

    @field_validator("start_url")
    @classmethod
    def _url_must_be_http(cls, v: str) -> str:
        if not v.lower().startswith(("http://", "https://")):
            raise ValueError("start_url must begin with http:// or https://")
        return v

    @field_validator("wait_until")
    @classmethod
    def _valid_wait(cls, v: str) -> str:
        allowed = {"load", "domcontentloaded", "networkidle", "commit"}
        if v not in allowed:
            raise ValueError(f"wait_until must be one of {allowed}")
        return v

    @field_validator("allow_patterns", "deny_patterns")
    @classmethod
    def _compile_regex(cls, patterns: list[str]) -> list[str]:
        for p in patterns:
            try:
                re.compile(p)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{p}': {e}")
        return patterns
