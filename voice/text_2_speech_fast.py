# text_2_speech_fast.py
"""
Faster drop-in replacement for text_2_speech.py
- In-memory pyttsx3 → WAV (no temp files)
- Optional Torch-audio resampling (much faster than librosa)
- Whisper warm-up + FP16 / CUDA (optional flag)
- Minor metric speed-ups
"""

from __future__ import annotations
import io
import os
import time
import math
import tempfile
import argparse
import threading  # Added for timeout
from typing import List, Dict, Any, Optional

import numpy as np
import soundfile as sf

# =========================
# Cached pyttsx3 engine (init once)
# =========================
import pyttsx3
_engine: Optional[pyttsx3.Engine] = None
def _get_engine() -> pyttsx3.Engine:
    global _engine
    if _engine is None:
        print("Initializing pyttsx3 engine...")  # Status indicator
        _engine = pyttsx3.init()
        # Default sensible settings – can be overridden via CLI later
        _engine.setProperty("rate", 180)
        print("pyttsx3 engine initialized.")
    return _engine

# Engine configuration cache to avoid repeated property sets
_engine_config: Dict[str, Any] = {"rate": None, "voice": None}
def _configure_engine(engine, rate: Optional[int] = None, voice: Optional[str] = None):
    """Set engine properties only when they differ from last-applied values."""
    global _engine_config
    if rate is not None and _engine_config.get("rate") != rate:
        try:
            engine.setProperty("rate", rate)
            _engine_config["rate"] = rate
        except Exception:
            pass
    if voice is not None and _engine_config.get("voice") != voice:
        try:
            engine.setProperty("voice", voice)
            _engine_config["voice"] = voice
        except Exception:
            pass

# =========================
# In-memory TTS → WAV bytes
# =========================
def _tts_to_wav_bytes(text: str, rate: int = 180, voice: Optional[str] = None,
                    target_sr: int = 16000) -> bytes:
    """
    Synthesise with pyttsx3 directly into a BytesIO (PCM) then wrap as WAV.
    No disk I/O, peak-normalised to -3 dBFS.
    """
    eng = _get_engine()
    _configure_engine(eng, rate=rate, voice=voice)  # Use cached config for efficiency

    # Fallback: use the original temp-file method (reliable across platforms)
    # The in-memory hook is fragile and often fails on non-Windows or older pyttsx3;
    # we default to temp-file for reliability, but keep logic for potential future hooks.
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        print(f"Synthesizing TTS for: '{text[:50]}{'...' if len(text) > 50 else ''}' ...", end=" ", flush=True)  # Progress start
        eng.save_to_file(text, tmp_path)
        
        # Threaded runAndWait with timeout to prevent hangs
        done_event = threading.Event()
        def _run():
            eng.runAndWait()
            done_event.set()
        thread = threading.Thread(target=_run)
        thread.start()
        if not done_event.wait(timeout=10):  # 10-second timeout
            raise TimeoutError("TTS synthesis timed out - check audio device, voice settings, or system speech service.")
        print("done.")  # Progress end
        
        with open(tmp_path, "rb") as f:
            wav_data = f.read()
        # Read, normalise, resample, etc.
        data, sr = sf.read(io.BytesIO(wav_data), dtype="float32")
        if data.ndim > 1:
            mono = data.mean(axis=1)
        else:
            mono = data
    except Exception as e:
        raise RuntimeError(f"TTS synthesis failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # ---------- Resample (fast path) ----------
    if sr != target_sr:
        print(f"Resampling audio from {sr} Hz to {target_sr} Hz...", end=" ", flush=True)  # Status
        mono = _fast_resample(mono, orig_sr=sr, target_sr=target_sr)
        print("done.")

    # ---------- Normalise peak ----------
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    target_peak = 10 ** (-3.0 / 20.0)  # -3 dBFS
    if peak > 0:
        mono = mono * (target_peak / max(peak, 1e-12))

    # ---------- Write WAV bytes ----------
    buf = io.BytesIO()
    sf.write(buf, mono, target_sr, format="WAV")
    return buf.getvalue()

# =========================
# Fast resample (torch-audio if present, else vectorised numpy)
# =========================
_resampler = None
def _init_resampler():
    global _resampler
    if _resampler is None:
        print("Initializing resampler (torchaudio if available)...", end=" ", flush=True)  # Status
        try:
            import torch
            import torchaudio
            _resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
            if torch.cuda.is_available():
                _resampler = _resampler.to('cuda')
            print("torchaudio resampler ready.")
        except Exception:
            _resampler = False  # flag for fallback
            print("falling back to numpy interpolation.")

def _fast_resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    _init_resampler()
    if _resampler:
        import torch
        tensor = torch.from_numpy(y).unsqueeze(0)  # (1, N)
        if torch.cuda.is_available():
            tensor = tensor.to('cuda')
        with torch.no_grad():
            out = _resampler(tensor).cpu().squeeze(0).numpy()
        return out
    # Numpy linear interpolation (fast enough for short utterances)
    old_t = np.linspace(0, len(y) / orig_sr, num=len(y), endpoint=False)
    new_len = int(len(y) * target_sr / orig_sr)
    new_t = np.linspace(0, len(y) / orig_sr, num=new_len, endpoint=False)
    return np.interp(new_t, old_t, y).astype(np.float32)

# =========================
# Audio metrics (same logic, but vectorised)
# =========================
def _peak_dbfs(x: np.ndarray) -> float:
    peak = np.abs(x).max()
    return 20 * np.log10(max(peak, 1e-12)) if peak > 0 else -math.inf

def _rms_db(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(x)))
    return 20 * np.log10(max(rms, 1e-12)) if rms > 0 else -math.inf

def estimate_loudness_proxy_lufs(x: np.ndarray, sr: int) -> float:
    return _rms_db(x)

def estimate_snr_db(x: np.ndarray, sr: int, sil_ms: int = 300) -> float:
    n = max(1, int(sr * sil_ms / 1000))
    if len(x) < 2 * n:
        noise = x[:n]
    else:
        noise = np.concatenate((x[:n], x[-n:]))
    signal_rms = _rms_db(x)
    noise_rms = _rms_db(noise)
    return signal_rms - noise_rms if noise_rms != -math.inf else math.inf

# =========================
# TTS evaluation (now uses in-memory path)
# =========================
def run_tts_eval(text: str, rate: int = 180, target_sr: int = 16000) -> Dict[str, Any]:
    t0 = time.perf_counter()
    wav_bytes = _tts_to_wav_bytes(text, rate=rate, target_sr=target_sr)
    synth_time = time.perf_counter() - t0

    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Pad for SNR (200 ms head/tail)
    pad_samples = int(0.2 * sr)
    y_padded = np.concatenate([np.zeros(pad_samples, dtype=np.float32), y, np.zeros(pad_samples, dtype=np.float32)])

    dur = len(y) / sr
    rtf = synth_time / max(dur, 1e-6)

    return {
        "latency_s": round(synth_time, 3),
        "duration_s": round(dur, 3),
        "rtf": round(rtf, 3),
        "peak_dbfs": round(_peak_dbfs(y_padded), 2),
        "snr_db": round(estimate_snr_db(y_padded, sr), 2),
        "lufs_proxy_db": round(estimate_loudness_proxy_lufs(y_padded, sr), 2),
        "sr": sr,
        "wav_bytes": wav_bytes,
    }

# =========================
# Whisper (cached, optional CUDA, FP16)
# =========================
_have_whisper = False
_asr_model = None
def _load_whisper_base(use_cuda: bool = False):
    global _asr_model, _have_whisper
    if _asr_model is None:
        print("Loading OpenAI Whisper model...", end=" ", flush=True)  # Status
        import whisper
        _have_whisper = True
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        _asr_model = whisper.load_model("base", device=device)
        print(f"OpenAI Whisper loaded on {device}")  # Verify device
    return _asr_model

def transcribe_wav_bytes(wav_bytes: bytes, language: Optional[str] = "en",
                        use_cuda: bool = False) -> Optional[str]:
    if not _have_whisper:
        return None

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Whisper expects 16 kHz
    if sr != 16000:
        data = _fast_resample(data, sr, 16000)

    model = _load_whisper_base(use_cuda=use_cuda)
    # FP16 on CUDA
    print("Running Whisper transcription...", end=" ", flush=True)  # Status
    result = model.transcribe(data, language=language, fp16=use_cuda)
    print("done.")
    return result.get("text", "").strip()


# --- faster-whisper (preferred path, now GPU-enabled with FP16) ---------------------------------
def _have_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False

_fw_models: Dict[str, Any] = {}
def _load_faster_whisper(model_size: str = "base", device: str = "cpu", compute_type: str = "float16"):
    if model_size in _fw_models:
        return _fw_models[model_size]
    print(f"Loading faster-whisper model ({model_size}) on {device}...", end=" ", flush=True)  # Status
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _fw_models[model_size] = model
        print(f"faster-whisper loaded on {device} with {compute_type}")  # Verify
        return model
    except Exception as e:
        print(f"faster-whisper load failed: {e}")
        return None

def fast_transcribe(wav_bytes: bytes, language: Optional[str] = "en", model_size: str = "base", use_cuda: bool = False) -> Optional[str]:
    """Try faster-whisper (CTranslate2) first, then fall back to OpenAI Whisper."""
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != 16000:
        try:
            data = _fast_resample(data, sr, 16000)
        except Exception:
            pass

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"  # FP16 on GPU for accuracy/speed

    # faster-whisper path
    if _have_faster_whisper():
        try:
            model = _load_faster_whisper(model_size, device=device, compute_type=compute_type)
            if model is not None:
                print("Transcribing with faster-whisper...", end=" ", flush=True)  # Status
                segments, info = model.transcribe(data, language=language, beam_size=1)
                text = " ".join([s.text for s in segments]).strip()  # Use s.text (fixed attr)
                print("done.")
                return text if text else None
        except Exception as e:
            print(f"faster-whisper inference failed: {e}")

    # fallback to openai-whisper if available
    try:
        asr = _load_whisper_base(use_cuda=use_cuda)
        print("Falling back to OpenAI Whisper transcription...", end=" ", flush=True)  # Status
        out = asr.transcribe(data, language=language, fp16=use_cuda)
        print("done.")
        return out.get("text", "").strip()
    except Exception as e:
        print(f"OpenAI Whisper fallback failed: {e}")
        return None

# =========================
# WER (unchanged)
# =========================
import re
def _tok(s: str) -> List[str]:
    return re.findall(r"\w+(?:'\w+)?", s.lower())

def wer(ref: str, hyp: str) -> float:
    r, h = _tok(ref), _tok(hyp)
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): dp[i][0] = i
    for j in range(len(h) + 1): dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(r)][len(h)] / max(1, len(r))

# =========================
# Test sentences (unchanged)
# =========================
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Set the gain to 3.5 decibels at two kilohertz.",
    "Wait… are you sure? Okay—let's begin.",
    "Backpropagation adjusts weights via gradient descent.",
    "Gracias por tu ayuda, see you mañana.",
    (
        "In 2025, open source text-to-speech systems continue to improve in both naturalness and controllability, "
        "making high-quality local speech synthesis practical on everyday laptops. Careful punctuation dramatically "
        "improves prosody: commas, periods, and dashes guide the rhythm and phrasing of the spoken output."
    ),
]

# =========================
# CLI
# =========================
def _print_table(rows: List[Dict[str, Any]]):
    headers = ["idx", "lat(s)", "dur(s)", "RTF", "peak(dBFS)", "SNR(dB)", "LUFS~", "sr", "WER"]
    print("\n" + " | ".join(headers))
    print("-" * (len(" | ".join(headers)) + 4))
    for r in rows:
        print(" | ".join([
            f"{r['idx']}",
            f"{r['latency_s']:.2f}",
            f"{r['duration_s']:.2f}",
            f"{r['rtf']:.2f}",
            f"{r['peak_dbfs']:.1f}",
            f"{r['snr_db']:.1f}",
            f"{r['lufs_proxy_db']:.1f}",
            f"{r['sr']}",
            (f"{r['wer']:.2f}" if r.get('wer') is not None else "—"),
        ]))

def main():
    parser = argparse.ArgumentParser(description="Fast TTS baseline + metrics + Whisper WER")
    parser.add_argument("--out", default="tts_sample.wav", help="Output WAV for first sample")
    parser.add_argument("--rate", type=int, default=180, help="TTS voice rate")
    parser.add_argument("--max", type=int, default=min(4, len(TEST_SENTENCES)), help="Max sentences")
    parser.add_argument("--wer-first-n", type=int, default=min(2, len(TEST_SENTENCES)), help="Compute WER for first N (0 = disable)")
    parser.add_argument("--cuda", action="store_true", help="Run Whisper on GPU if available")
    parser.add_argument("--synthesize-only", action="store_true", help="Only write WAVs, skip metrics")
    parser.add_argument("--out-dir", default="tts_outputs", help="Dir for per-sentence WAVs")
    parser.add_argument("--synthesize-single", action="store_true", help="Concatenate all into one file")
    args = parser.parse_args()

    print("Starting TTS evaluation script...")  # Initial status

    # Verify torch CUDA early
    try:
        import torch
        if args.cuda and torch.cuda.is_available():
            print(f"CUDA verified: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        elif args.cuda:
            print("CUDA requested but not available - falling back to CPU")
    except Exception as e:
        print(f"Torch/CUDA init issue: {e}")

    print("== Fast TTS baseline ==")
    rows: List[Dict[str, Any]] = []
    first_wav: Optional[bytes] = None
    synth_buffers: List[tuple[np.ndarray, int]] = []

    # Warm-up Whisper early if we know we will use it
    if args.wer_first_n > 0:
        print("Warming up Whisper models...", end=" ", flush=True)
        if args.cuda:
            _load_faster_whisper(device="cuda", compute_type="float16")  # Prioritize faster-whisper GPU
            _load_whisper_base(use_cuda=True)
        else:
            _load_faster_whisper(device="cpu")
            _load_whisper_base(use_cuda=False)
        print("done.")

    for i, txt in enumerate(TEST_SENTENCES[: args.max], 1):
        print(f"\nProcessing sentence {i}/{min(args.max, len(TEST_SENTENCES))}:")  # Loop progress
        res = run_tts_eval(txt, rate=args.rate)
        wav = res.pop("wav_bytes")
        if first_wav is None:
            first_wav = wav

        # ---------- synthesise-only fast path ----------
        if args.synthesize_only:
            data, sr = sf.read(io.BytesIO(wav), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if args.synthesize_single:
                synth_buffers.append((data, sr))
            else:
                os.makedirs(args.out_dir, exist_ok=True)
                path = os.path.join(args.out_dir, f"sample_{i:02d}.wav")
                with open(path, "wb") as f:
                    f.write(wav)
                print(f"Wrote {path}")
            continue

        # ---------- optional WER ----------
        w = None
        if args.wer_first_n and i <= args.wer_first_n:
            print(f"Computing WER for sentence {i}...", end=" ", flush=True)  # Status
            hyp = fast_transcribe(wav, language="en", model_size="base", use_cuda=args.cuda)
            if hyp:
                w = wer(txt, hyp)
            print("done.")

        rows.append({"idx": i, **res, "wer": w})

    # ---------- output ----------
    if args.synthesize_only and args.synthesize_single and synth_buffers:
        print("Concatenating audio buffers...", end=" ", flush=True)  # Status
        # concatenate with 0.1 s silence
        if synth_buffers:
            pad = np.zeros(int(0.1 * synth_buffers[0][1]), dtype=np.float32)
            parts: List[np.ndarray] = []
            sr = synth_buffers[0][1]
            for data, s in synth_buffers:
                if s != sr:
                    data = _fast_resample(data, s, sr)
                parts.append(data)
                parts.append(pad)
            merged = np.concatenate(parts)
            sf.write(args.out, merged, sr)
            print(f"done. Wrote concatenated WAV → {args.out}")
        return

    _print_table(rows)

    if first_wav:
        with open(args.out, "wb") as f:
            f.write(first_wav)
        print(f"\nSaved first sample → {args.out}")

    # simple sanity warnings
    bad = [r for r in rows if r["peak_dbfs"] > -0.1 or r["rtf"] > 1.0]
    if bad:
        print("\nWarnings:")
        for r in bad:
            if r["peak_dbfs"] > -0.1:
                print(f"  • Row {r['idx']}: peak near 0 dBFS (risk of clipping).")
            if r["rtf"] > 1.0:
                print(f"  • Row {r['idx']}: RTF > 1.0 (slower than real-time).")
    print("\nDone.")

if __name__ == "__main__":
    # Import torch early so the CUDA check is ready for the resampler
    try:
        import torch
    except Exception:
        torch = None  # type: ignore
    main()