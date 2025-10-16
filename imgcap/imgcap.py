from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import re
import time, json

def _extract_numbers(*texts):
    s = " ".join([t for t in texts if t])
    return re.findall(r"[-+]?\d*\.?\d+(?:[%])?", s)

def _extract_units(*texts):
    s = " ".join([t for t in texts if t])
    # expand as needed
    units = re.findall(r"\b(\d+(?:\.\d+)?)\s*(%|ms|s|Hz|kHz|MHz|GB|MB|cm|mm)\b", s)
    return [{"value": v, "unit": u} for v, u in units]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"   # swap to: "OpenGVLab/InternVL2-2B"

# Fast processor handles tiling for high-res; set use_fast=False if you want legacy behavior.
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
).to(DEVICE)

def vl_task(images, text):
    """
    Multipurpose runner:
      - images: list of PIL.Image (one or more)
      - text: instruction/question (str)
    """
    # Build content blocks: one {"type":"image"} per image, followed by text
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    enc = processor(text=prompt, images=images, return_tensors="pt")
    # Move to same device
    for k, v in list(enc.items()):
        if hasattr(v, "to"):
            enc[k] = v.to(DEVICE)

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=256,
            num_beams=2,
            early_stopping=True,
        )
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

# === Examples ===
img = Image.open("data/ae.jpg").convert("RGB")
# print("\nCaption:\n", vl_task([img], "Give a detailed descripion and ocr of this image"))
# print("\nOCR:\n", vl_task([img], "Transcribe all visible text in reading order."))
# print("\nVQA:\n", vl_task([img], "What is the main takeaway from the chart? List any numbers you can read."))
# # Multi-image compare:
# img2 = Image.open("slide_prev.png").convert("RGB")
# print("\nCompare:\n", vl_task([img, img2], "Compare these: what changed between the two slides?"))
t0 = time.time()
cap = vl_task([img], "Give a detailed descripion  this image")
ocr = vl_task([img], "Transcribe all visible text in reading order.")
t1 = time.time()
dt=t1-t0
payload = {
    "model": MODEL_ID,
    "device": DEVICE,
    "duration_sec": dt,
    "outputs": {
        "caption": cap,
        "ocr_text": ocr,          # null if not requested
        # lightweight extracted facts (optional mini post-process examples):
        "numbers": _extract_numbers(cap, ocr),
        "units":   _extract_units(cap, ocr),
    },
    "meta": {
        "generator": "image-text-to-text",
        "params": {"num_beams": 2, "max_new_tokens": 128, "use_fast_processor": True}
    }
}

print(json.dumps(payload, ensure_ascii=False, indent=2))
