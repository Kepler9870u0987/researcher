"""
web2pdf_pipeline.py
===================

Legacy wrapper — delegates to the web2pdf package.
See web2pdf/ for the full implementation.

Usage:
    python web2pdf_pipeline.py https://example.com/docs --depth 2 --out runs_pdf

Or preferably:
    python -m web2pdf crawl https://example.com/docs --depth 2 --out runs_pdf
"""
import sys

from web2pdf.cli import main

sys.exit(main())

