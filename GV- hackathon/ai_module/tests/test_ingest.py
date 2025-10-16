from __future__ import annotations

"""Smoke test for ingest: generate a tiny one-page PDF programmatically and load it."""

import io
import os
from typing import List


def _make_sample_pdf(path: str) -> None:
    import fitz  # type: ignore

    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    # Write a line longer than the 20-char OCR threshold to avoid OCR/poppler requirement.
    page.insert_text((72, 100), "Hello AI Tutor – this page has sufficient raw text to skip OCR.", fontsize=12)
    doc.save(path)
    doc.close()


def test_load_pdf_smoke(tmp_path):
    from ai_module.core.ingest import load_pdf

    pdf_path = os.path.join(tmp_path, "sample.pdf")
    _make_sample_pdf(pdf_path)

    pages = load_pdf(pdf_path)
    assert isinstance(pages, list)
    assert len(pages) >= 1
    first = pages[0]
    assert "page_context" in first
    assert isinstance(first["page_context"], str)
    assert len(first["page_context"]) > 0
