"""PDF ingestion pipeline per page: text, OCR, image captions, merged context.

Relies on PyMuPDF (fitz) for text and image extraction, pdf2image to render full page
bitmaps for OCR fallback.
"""
from __future__ import annotations

from typing import List, Dict


def _word_tokens(s: str) -> int:
    return len([w for w in (s or "").split() if w])


def load_pdf(path: str) -> List[Dict]:
    """Load a PDF and produce page contexts.

    Returns list of dicts: {"page_id", "raw_text", "ocr_text", "captions", "page_context", "tokens"}
    """
    # Local imports to avoid heavy load if unused
    import fitz  # type: ignore
    from PIL import Image  # type: ignore

    from .cleaning import merge_fields, compact_whitespace
    from .ocr import extract_text as ocr_extract
    from .caption import caption_image

    doc = fitz.open(path)

    results: List[Dict] = []
    for i, page in enumerate(doc):
        page_id = i + 1
        raw_text = page.get_text("text") or ""

        # OCR fallback if raw text is missing or very short
        ocr_text = ""
        if len(raw_text.strip()) < 20:
            try:
                # Render this page only to avoid Poppler requirement if not needed
                try:
                    from pdf2image import convert_from_path  # type: ignore
                except Exception:
                    convert_from_path = None  # type: ignore

                if convert_from_path is not None:
                    page_images = convert_from_path(path, first_page=page_id, last_page=page_id)
                    full_img = page_images[0]
                else:
                    # Fallback: render via PyMuPDF directly
                    try:
                        zoom = 2.0
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat)
                        mode = "RGB" if pix.alpha == 0 else "RGBA"
                        pil_bytes_mode = mode
                        from PIL import Image  # type: ignore

                        full_img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                        ocr_text = ocr_extract(full_img)
                    except Exception:
                        full_img = None
                if full_img is not None and not ocr_text:
                    ocr_text = ocr_extract(full_img)
            except Exception:
                ocr_text = ""

        # Image captions for embedded images on the page
        captions: List[str] = []
        try:
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n < 5:  # GRAY or RGB
                        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    else:  # CMYK -> convert to RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    cap = caption_image(pil_img)
                    if cap:
                        captions.append(cap)
                except Exception:
                    continue
        except Exception:
            captions = []

        page_context = merge_fields(raw_text, ocr_text, captions)
        page_context = compact_whitespace(page_context)
        tokens = _word_tokens(page_context)

        results.append(
            {
                "page_id": page_id,
                "raw_text": raw_text,
                "ocr_text": ocr_text,
                "captions": captions,
                "page_context": page_context,
                "tokens": tokens,
            }
        )

    return results

######################################QWEN IMPLEMENTATION#####################################################
# from typing import List, Dict

# from typing import List, Dict

# def load_pdf(path: str) -> List[Dict]:
#     """Load a PDF and produce page contexts using Qwen VL for OCR and captions.

#     Returns list of dicts:
#     {
#         "page_id": int,
#         "raw_text": str,
#         "ocr_text": str,
#         "captions": List[str],
#         "page_context": str,
#         "tokens": List[str]
#     }
#     """
#     import fitz  # PyMuPDF
#     from PIL import Image  # for rasterization
#     from .cleaning import merge_fields, compact_whitespace
#     from .caption import vl_task  # <- uses Qwen internally (return_json=True)

#     doc = fitz.open(path)
#     results: List[Dict] = []

#     for i, page in enumerate(doc):
#         page_id = i + 1
#         raw_text = page.get_text("text") or ""

#         # --------- Rasterize PDF page to image for Qwen OCR ----------
#         ocr_text = ""
#         full_img = None
#         try:
#             # High-res rasterization via PyMuPDF
#             zoom = 2.0
#             mat = fitz.Matrix(zoom, zoom)
#             pix = page.get_pixmap(matrix=mat)
#             mode = "RGB" if pix.alpha == 0 else "RGBA"
#             full_img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
#             if mode == "RGBA":
#                 full_img = full_img.convert("RGB")
#         except Exception:
#             full_img = None

#         # Use Qwen model for OCR (and captioning)
#         if full_img is not None:
#             try:
#                 payload = vl_task([full_img], return_json=True)
#                 ocr_text = payload["outputs"].get("ocr_text", "") or ""
#             except Exception:
#                 ocr_text = ""

#         # --------- Embedded image captions via Qwen ----------
#         captions: List[str] = []
#         try:
#             image_list = page.get_images(full=True)
#             for img in image_list:
#                 xref = img[0]
#                 try:
#                     pix = fitz.Pixmap(doc, xref)
#                     if pix.n < 5:  # RGB or Gray
#                         pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                     else:  # CMYK → RGB
#                         pix = fitz.Pixmap(fitz.csRGB, pix)
#                         pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

#                     # Use Qwen caption from vl_task
#                     p = vl_task([pil_img], return_json=True)
#                     cap = p["outputs"].get("caption", "")
#                     if cap:
#                         captions.append(cap)
#                 except Exception:
#                     continue
#         except Exception:
#             captions = []

#         # --------- Combine page text, OCR, and captions ----------
#         page_context = merge_fields(raw_text, ocr_text, captions)
#         page_context = compact_whitespace(page_context)
#         tokens = _word_tokens(page_context)  # ← keep your existing tokenizer

#         results.append(
#             {
#                 "page_id": page_id,
#                 "raw_text": raw_text,
#                 "ocr_text": ocr_text,
#                 "captions": captions,
#                 "page_context": page_context,
#                 "tokens": tokens,
#             }
#         )

#     return results


