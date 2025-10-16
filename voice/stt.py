# whisper_hf_asr.py
import json, time, math, io, os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = os.getenv("WHISPER_MODEL_ID", "openai/whisper-small")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if (DEVICE == "cuda" and os.getenv("WHISPER_FP16","1")=="1") else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
model.eval()

# ---- utils ----
def _to_mono_16k(x: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray,int]:
    """Return mono float32 at 16 kHz using numpy (fast, no torchaudio/librosa req)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr == target_sr:
        return x.astype(np.float32, copy=False), sr
    # linear resample (good enough for ASR)
    t_old = np.linspace(0, len(x)/sr, num=len(x), endpoint=False)
    new_len = int(round(len(x) * target_sr / sr))
    t_new = np.linspace(0, len(x)/sr, num=new_len, endpoint=False)
    y = np.interp(t_new, t_old, x).astype(np.float32)
    return y, target_sr

def _read_audio_any(inp: Union[str, bytes, np.ndarray, io.BytesIO]) -> Tuple[np.ndarray,int]:
    """Load audio to float32 numpy and sample rate using soundfile."""
    if isinstance(inp, (bytes, bytearray)):
        data, sr = sf.read(io.BytesIO(inp), dtype="float32", always_2d=True)
    elif isinstance(inp, io.BytesIO):
        data, sr = sf.read(inp, dtype="float32", always_2d=True)
    elif isinstance(inp, np.ndarray):
        # assume 16k mono if no sr; caller should provide sr separately (not handled here)
        raise ValueError("If passing ndarray, wrap into bytes or file; need a samplerate.")
    else:
        data, sr = sf.read(str(inp), dtype="float32", always_2d=True)
    return data, sr

# ---- chunking ----
@dataclass
class ChunkerCfg:
    max_sec: float = 25.0           # keep under ~30s context
    stride_sec: float = 2.0         # overlap to avoid word cuts

def _make_chunks(n_samples: int, sr: int, cfg: ChunkerCfg) -> List[Tuple[int,int]]:
    max_len = int(cfg.max_sec * sr)
    stride   = int(cfg.stride_sec * sr)
    chunks = []
    start = 0
    while start < n_samples:
        end = min(start + max_len, n_samples)
        chunks.append((start, end))
        if end == n_samples:
            break
        start = max(0, end - stride)
    return chunks

# ---- ASR core ----
def transcribe(
    source: Union[str, bytes, io.BytesIO],
    language: Optional[str] = None,
    task: str = "transcribe",
    chunk_cfg: ChunkerCfg = ChunkerCfg(),
    return_segments: bool = True,
) -> Dict[str, Any]:
    """
    Transcribe audio (any SR/mono/stereo) into text with optional segments.
    - Normalizes to 16 kHz mono
    - Splits long audio into overlapping chunks
    - Returns full text and per-chunk timestamps
    """
    # 1) load + normalize
    raw, sr = _read_audio_any(source)
    mono = raw.mean(axis=1)
    mono, sr = _to_mono_16k(mono, sr, 16000)

    # 2) chunk plan
    chunks = _make_chunks(len(mono), sr, chunk_cfg)

    # 3) run whisper on each chunk
    texts: List[str] = []
    segs: List[Dict[str, Any]] = []
    t0 = time.time()

    for (a, b) in chunks:
        audio_chunk = mono[a:b]
        inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        inputs["input_features"] = inputs["input_features"].to(DTYPE)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                language=language,            # None => auto
                task=task,                    # "transcribe" or "translate"
                max_new_tokens=448,
                num_beams=1,
                do_sample=False,
                # temperature=0.0,            # add if you want fallback sampling later
            )

        txt = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        texts.append(txt)
        if return_segments:
            segs.append({
                "start": round(a / sr, 3),
                "end":   round(b / sr, 3),
                "text":  txt
            })

    total = " ".join(t.strip() for t in texts).strip()
    out: Dict[str, Any] = {
        "text": total,
        "duration_sec": round(len(mono)/sr, 3),
        "processing_sec": round(time.time() - t0, 3),
        "dtype": str(DTYPE),
        "device": DEVICE,
        "num_chunks": len(chunks),
    }
    if return_segments:
        out["segments"] = segs
    return out

# ---- tiny CLI ----
if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to audio (wav/mp3/flac/...)")
    p.add_argument("--language", default=None, help="ISO code, e.g. en; omit for autodetect")
    p.add_argument("--task", default="transcribe", choices=["transcribe","translate"])
    p.add_argument("--max-sec", type=float, default=25.0)
    p.add_argument("--stride-sec", type=float, default=2.0)
    p.add_argument("--segments", action="store_true", help="Include segment list with timestamps")
    args = p.parse_args()

    cfg = ChunkerCfg(max_sec=args.max_sec, stride_sec=args.stride_sec)
    result = transcribe(args.path, language=args.language, task=args.task,
                        chunk_cfg=cfg, return_segments=args.segments)
    print(json.dumps(result, ensure_ascii=False, indent=2))
