"""Image captioning using BLIP (Salesforce/blip-image-captioning-base).

Lazy-loads model and processor to keep import times low.
# """
from __future__ import annotations

from typing import Optional

from PIL import Image

_pipe = None  # type: ignore[var-annotated]


def _get_pipeline():
    global _pipe
    if _pipe is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise RuntimeError(
                "transformers is required for BLIP captions. Please install 'transformers'."
            ) from e
        # Lazy load heavy weights
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _pipe = (processor, model)
    return _pipe


def caption_image(pil_image: Image.Image) -> str:
    """Generate a caption for the given image.

    Returns an empty string if captioning fails for any reason.
    """
    try:
        processor, model = _get_pipeline()
        inputs = processor(images=pil_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30)
        # decode
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception:
        return ""




# from __future__ import annotations

# from typing import List, Dict, Any, Optional
# from PIL import Image
# import torch
# import re
# import time
# import json


# # -----------------------------
# # Extraction helpers (unchanged)
# # -----------------------------
# def _extract_numbers(*texts: str) -> List[str]:
#     s = " ".join([t for t in texts if t])
#     return re.findall(r"[-+]?\d*\.?\d+(?:[%])?", s)

# def _extract_units(*texts: str) -> List[Dict[str, str]]:
#     s = " ".join([t for t in texts if t])
#     # expand as needed
#     units = re.findall(r"\b(\d+(?:\.\d+)?)\s*(%|ms|s|Hz|kHz|MHz|GB|MB|cm|mm)\b", s)
#     return [{"value": v, "unit": u} for v, u in units]

# def _strip_chat_scaffold(s: str) -> str:
#     """
#     Remove any leading 'system/user/assistant' scaffolding that Qwen may echo.
#     Keeps only the assistant's content.
#     """
#     if not s:
#         return s
#     s = s.replace("\r\n", "\n").strip()

#     # Preferred: cut at the first explicit 'assistant' role marker
#     m = re.search(r"(?:^|\n)assistant\s*\n", s, flags=re.IGNORECASE)
#     if m:
#         return s[m.end():].lstrip(" .:\n\t")

#     # Fallback: delete any leading role-tagged blocks if no 'assistant' split
#     # (handles cases like ".\nuser\n...assistant\n..." or just "user\n...")
#     s = re.sub(r"(?im)^(system|user|assistant)\s*\n", "", s)
#     # If the model echoed "user\n...assistant\n..." inline, nuke the prefix up to assistant:
#     s = re.sub(r"(?is)^.*?\bassistant\s*\n", "", s).lstrip(" .:\n\t")

#     return s.strip()

# def flatten_json(data, parent_key='', sep='.'):
#     items = []
#     for k, v in data.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_json(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)

# # Tiny helper if you want just the two strings extracted from the JSON
# def vl_extract_caption_and_ocr(payload):
#     o = payload.get("outputs", {})
#     return o.get("caption", ""), o.get("ocr_text", "")



# # ---------------------------------
# # Common device / defaults (shared)
# # ---------------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ======================================================================
# #                           Q W E N    V L
# # ======================================================================
# class QwenVL:
#     """
#     Utility for OCR / captioning / VQA using Qwen image-text-to-text models.
#     Default model: 'Qwen/Qwen2.5-VL-3B-Instruct'
#     Fully lazy-loaded; preserves your original vl_task behavior & signature.
#     """

#     _processor = None
#     _model = None
#     _model_id = None
#     _dtype = None
#     _device = DEVICE

#     @classmethod
#     def _ensure_loaded(
#         cls,
#         model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
#         use_fast: bool = True,
#         low_cpu_mem_usage: bool = True,
#     ):
#         if cls._processor is not None and cls._model is not None and cls._model_id == model_id:
#             return

#         from transformers import AutoProcessor, AutoModelForImageTextToText  # type: ignore

#         cls._model_id = model_id
#         cls._dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#         # Fast processor handles tiling for high-res; legacy if use_fast=False.
#         cls._processor = AutoProcessor.from_pretrained(
#             model_id, trust_remote_code=True, use_fast=use_fast
#         )
#         cls._model = AutoModelForImageTextToText.from_pretrained(
#             model_id,
#             trust_remote_code=True,
#             dtype=cls._dtype,
#             low_cpu_mem_usage=low_cpu_mem_usage,
#         ).to(cls._device)

#     # -------- Core generation (keeps your prompt + apply_chat_template flow) --------
#     @classmethod
#     def run(
#         cls,
#         images: List[Image.Image],
#         text: str,
#         model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
#         max_new_tokens: int = 256,
#         num_beams: int = 2,
#         early_stopping: bool = True,
#         use_fast: bool = True,
#         low_cpu_mem_usage: bool = True,
#     ) -> str:
#         """
#         Multipurpose runner (drop-in replacement for your vl_task):
#           - images: list of PIL.Image
#           - text: instruction/question (str)

#         Returns decoded string.
#         """
#         cls._ensure_loaded(model_id=model_id, use_fast=use_fast, low_cpu_mem_usage=low_cpu_mem_usage)

#         # Build content blocks: one {"type":"image"} per image, followed by text
#         content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
#         messages = [{"role": "user", "content": content}]
#         prompt = cls._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#         enc = cls._processor(text=prompt, images=images, return_tensors="pt")
#         enc = {k: (v.to(cls._device) if hasattr(v, "to") else v) for k, v in enc.items()}

#         with torch.inference_mode():
#             out = cls._model.generate(
#                 **enc,
#                 max_new_tokens=max_new_tokens,
#                 num_beams=num_beams,
#                 early_stopping=early_stopping,
#             )
        
#         result = cls._processor.batch_decode(out, skip_special_tokens=True)[0].strip()

#         result = _strip_chat_scaffold(result)
        
#         return result
    
#     # ------------- Convenience wrappers (optional nice-to-haves) -------------
#     @classmethod
#     def caption(
#         cls,
#         image: Image.Image,
#         detail_prompt: str = "Give a detailed description of this image.",
#         **gen_kwargs,
#     ) -> str:
#         return cls.run([image], detail_prompt, **gen_kwargs)

#     @classmethod
#     def ocr(
#         cls,
#         image: Image.Image,
#         **gen_kwargs,
#     ) -> str:
#         return cls.run([image], "Transcribe all visible text in reading order.", **gen_kwargs)

#     @classmethod
#     def compare(
#         cls,
#         image_a: Image.Image,
#         image_b: Image.Image,
#         **gen_kwargs,
#     ) -> str:
#         return cls.run(
#             [image_a, image_b],
#             "Compare these images and list what changed.",
#             **gen_kwargs,
#         )


# # For strict backward-compat with your previous function name:
# # --- keep your existing imports/helpers/QwenVL.run(...) and _strip_chat_scaffold(...) ---

# def vl_task(
#     images,
#     text: Optional[str] = None,
#     *,
#     return_json: bool = False,
#     caption_prompt: str = "Give a detailed description of this image",
#     ocr_prompt: str = "Transcribe all visible text in reading order.",
#     model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
#     num_beams: int = 2,
#     max_new_tokens: int = 128,
#     use_fast: bool = True,
# ):
#     """
#     Dual-mode:
#       - Default (return_json=False and text provided): behaves like before, returns a STRING for the given `text`.
#       - JSON mode (return_json=True): runs *both* caption + OCR and returns a JSON-ready dict payload.
#     """
#     # ---- JSON MODE: do caption + OCR together ----
#     if return_json:
#         import time
#         t0 = time.time()
#         caption = QwenVL.run(
#             [images[0]] if images and len(images) == 1 else images,
#             caption_prompt,
#             model_id=model_id,
#             num_beams=num_beams,
#             max_new_tokens=max_new_tokens,
#             use_fast=use_fast,
#         )
#         ocr_text = QwenVL.run(
#             [images[0]] if images and len(images) == 1 else images,
#             ocr_prompt,
#             model_id=model_id,
#             num_beams=num_beams,
#             max_new_tokens=max_new_tokens,
#             use_fast=use_fast,
#         )
#         dt = time.time() - t0

#         payload = {
#             "model": model_id,
#             "device": DEVICE,
#             "duration_sec": dt,
#             "outputs": {
#                 "caption": caption,                 # already sanitized in QwenVL.run
#                 "ocr_text": ocr_text,               # already sanitized in QwenVL.run
#                 "numbers": _extract_numbers(caption, ocr_text),
#                 "units": _extract_units(caption, ocr_text),
#             },
#             "meta": {
#                 "generator": "image-text-to-text",
#                 "params": {
#                     "num_beams": num_beams,
#                     "max_new_tokens": max_new_tokens,
#                     "use_fast_processor": use_fast,
#                 },
#             },
#         }
#         print(json.dumps(payload, ensure_ascii=False, indent=2))
#         # cap, ocr = vl_extract_caption_and_ocr(payload)

#         return payload

#     # ---- CLASSIC MODE: return just the generated string for the given text ----
#     if text is None:
#         raise ValueError("In classic mode, you must provide `text`. Or set return_json=True to get caption+ocr.")
#     return QwenVL.run(
#         [images[0]] if images and len(images) == 1 else images,
#         text,
#         model_id=model_id,
#         num_beams=num_beams,
#         max_new_tokens=max_new_tokens,
#         use_fast=use_fast,
#     )


# # ======================================================================
# #                             B L I P
# # ======================================================================
# class ImageCaptionerBLIP:
#     """
#     Utility for generating captions using BLIP (kept intact).
#     Lazy-loads model + processor once.
#     """

#     _processor = None
#     _model = None

#     @classmethod
#     def _load_pipeline(cls):
#         if cls._processor is None or cls._model is None:
#             try:
#                 from transformers import BlipProcessor, BlipForConditionalGeneration
#             except Exception as e:
#                 raise RuntimeError(
#                     "transformers is required for BLIP captions. "
#                     "Please install it via `pip install transformers`."
#                 ) from e

#             cls._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#             cls._model = BlipForConditionalGeneration.from_pretrained(
#                 "Salesforce/blip-image-captioning-base"
#             ).to(DEVICE)

#     @classmethod
#     def caption(cls, image: Image.Image, max_new_tokens: int = 30) -> str:
#         try:
#             cls._load_pipeline()
#             inputs = cls._processor(images=image, return_tensors="pt")
#             inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
#             with torch.inference_mode():
#                 out = cls._model.generate(**inputs, max_new_tokens=max_new_tokens)
#             caption = cls._processor.decode(out[0], skip_special_tokens=True)
#             return caption.strip()
#         except Exception:
#             return ""


# # Backward-compat top-level function name (BLIP-based)
# def caption_image(pil_image: Image.Image) -> str:
#     """
#     Original BLIP captioning entry-point. Preserved.
#     """
#     return ImageCaptionerBLIP.caption(pil_image)

