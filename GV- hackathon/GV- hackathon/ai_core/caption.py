"""Image captioning using BLIP (Salesforce/blip-image-captioning-base).

Lazy-loads model and processor to keep import times low.
"""
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
