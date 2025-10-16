"""OCR utilities built on EasyOCR with a lazy-loaded singleton reader.

This module isolates EasyOCR import and model loading to avoid heavy startup cost
when OCR isn't needed.
"""
from __future__ import annotations

from typing import Optional

from PIL import Image

_reader = None  # type: ignore[var-annotated]


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError("EasyOCR is not installed. Please install 'easyocr'.") from e
        # English by default; allow numbers/symbols
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text(pil_image: Image.Image) -> str:
    """Run OCR on the provided PIL image.

    Args:
        pil_image: A PIL Image.
    Returns:
        Extracted text string (may be empty).
    """
    reader = _get_reader()
    result = reader.readtext(pil_image, detail=0, paragraph=True)
    # result is a list[str]
    return "\n".join([t.strip() for t in result if t and isinstance(t, str)])
