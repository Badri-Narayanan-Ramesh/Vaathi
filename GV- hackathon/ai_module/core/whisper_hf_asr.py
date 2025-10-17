# whisper_hf_asr.py
import io, os, json, time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# -------------------------
# Config / Model Load
# -------------------------
MODEL_ID = os.getenv("WHISPER_MODEL_ID", "openai/whisper-small")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if (DEVICE == "cuda" and os.getenv("WHISPER_FP16", "1") == "1") else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
model.eval()

# -------------------------
# Audio helpers
# -------------------------
def _to_mono_16k(x: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Return mono float32 at 16 kHz using fast NumPy linear resample."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr == target_sr:
        return x.astype(np.float32, copy=False), sr
    # Linear resample (good enough for ASR)
    t_old = np.linspace(0, len(x) / sr, num=len(x), endpoint=False)
    new_len = int(round(len(x) * target_sr / sr))
    t_new = np.linspace(0, len(x) / sr, num=new_len, endpoint=False)
    y = np.interp(t_new, t_old, x).astype(np.float32)
    return y, target_sr

def _read_audio_any(inp: Union[str, bytes, io.BytesIO]) -> Tuple[np.ndarray, int]:
    """Load audio into float32 numpy array (always_2d=True) and samplerate via soundfile."""
    if isinstance(inp, (bytes, bytearray)):
        data, sr = sf.read(io.BytesIO(inp), dtype="float32", always_2d=True)
    elif isinstance(inp, io.BytesIO):
        data, sr = sf.read(inp, dtype="float32", always_2d=True)
    else:
        data, sr = sf.read(str(inp), dtype="float32", always_2d=True)
    return data, sr

# -------------------------
# Chunking
# -------------------------
@dataclass
class ChunkerCfg:
    max_sec: float = 25.0   # keep chunks short for stable decoding
    stride_sec: float = 2.0 # overlap to reduce word cuts at boundaries

def _make_chunks(n_samples: int, sr: int, cfg: ChunkerCfg) -> List[Tuple[int, int]]:
    max_len = int(cfg.max_sec * sr)
    stride  = int(cfg.stride_sec * sr)
    chunks: List[Tuple[int, int]] = []
    start = 0
    while start < n_samples:
        end = min(start + max_len, n_samples)
        chunks.append((start, end))
        if end == n_samples:
            break
        start = max(0, end - stride)  # overlap next chunk
    return chunks

# -------------------------
# ASR core
# -------------------------
def transcribe(
    source: Union[str, bytes, io.BytesIO],
    language: Optional[str] = None,      # e.g., "en" (None => autodetect)
    task: str = "transcribe",            # "transcribe" or "translate"
    chunk_cfg: ChunkerCfg = ChunkerCfg(),
    return_segments: bool = True,
) -> Dict[str, Any]:
    """
    Transcribe audio of any rate/channels → text (with optional per-chunk timestamps).
    - Normalizes to 16 kHz mono
    - Chunks long audio with overlap
    - Uses safe max_length (no max_new_tokens overflow)
    """
    # 1) Load + normalize
    raw, sr = _read_audio_any(source)
    mono = raw.mean(axis=1)
    mono, sr = _to_mono_16k(mono, sr, 16000)

    # 2) Plan chunks
    chunks = _make_chunks(len(mono), sr, chunk_cfg)

    # 3) Decode each chunk
    texts: List[str] = []
    segs: List[Dict[str, Any]] = []
    t0 = time.time()

    # Safe total length for Whisper decoder (prevents your earlier error)
    max_target = getattr(model.config, "max_target_positions", 448)

    for (a, b) in chunks:
        audio_chunk = mono[a:b]

        inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt")
        # Move to device and dtype (match model precision)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        inputs["input_features"] = inputs["input_features"].to(DTYPE)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                language=language,     # None => autodetect
                task=task,             # "transcribe"/"translate"
                max_length=max_target, # total decoder length cap (SAFE)
                num_beams=1,
                do_sample=False,
            )

        txt = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        texts.append(txt)
        if return_segments:
            segs.append({
                "start": round(a / sr, 3),
                "end":   round(b / sr, 3),
                "text":  txt,
            })

    total_text = " ".join(t.strip() for t in texts).strip()
    out: Dict[str, Any] = {
        "text": total_text,
        "duration_sec": round(len(mono) / sr, 3),
        "processing_sec": round(time.time() - t0, 3),
        "dtype": str(DTYPE),
        "device": DEVICE,
        "num_chunks": len(chunks),
    }
    if return_segments:
        out["segments"] = segs
    return out

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Whisper (HF) ASR with safe settings")
    p.add_argument("path", help="Path to audio (wav/flac/ogg; mp3 support depends on your soundfile build)")
    p.add_argument("--language", default=None, help="e.g., en (omit for autodetect)")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--max-sec", type=float, default=25.0, help="Chunk length seconds")
    p.add_argument("--stride-sec", type=float, default=2.0, help="Chunk overlap seconds")
    p.add_argument("--segments", action="store_true", help="Include timestamped segments")
    args = p.parse_args()

    cfg = ChunkerCfg(max_sec=args.max_sec, stride_sec=args.stride_sec)
    result = transcribe(
        args.path,
        language=args.language,
        task=args.task,
        chunk_cfg=cfg,
        return_segments=args.segments,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
