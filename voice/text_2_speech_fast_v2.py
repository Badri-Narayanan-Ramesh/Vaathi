# text_2_speech_fast_v2.py
"""
Faster drop-in replacement for text_2_speech.py
- In-memory pyttsx3 → WAV (no temp files)
- Optional Torch-audio resampling (much faster than librosa)
- Whisper warm-up + FP16 / CUDA (optional flag)
- Minor metric speed-ups
- Fixed: Fresh pyttsx3 engine per call to prevent hangs on repeated use
- New: --all-in-one flag to save concatenated audio + run full metrics/WER
"""

from __future__ import annotations
import io
import os
import time
import math
import tempfile
import argparse
import threading  # For timeout
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import soundfile as sf

# Set env vars at top to suppress OMP issues (if not set in shell)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# =========================
# In-memory TTS → WAV bytes (fresh engine per call)
# =========================
import pyttsx3
def _tts_to_wav_bytes(text: str, rate: int = 180, voice: Optional[str] = None,
                    target_sr: int = 16000) -> bytes:
    """
    Synthesise with pyttsx3 (new engine instance) into temp WAV.
    Peak-normalised to -3 dBFS.
    """
    print(f"Synthesizing TTS for: '{text[:50]}{'...' if len(text) > 50 else ''}' ...", end=" ", flush=True)  # Progress start
    
    eng = pyttsx3.init()  # Fresh engine each time to avoid state hangs
    if rate: eng.setProperty("rate", rate)
    if voice: eng.setProperty("voice", voice)
    
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    
    try:
        eng.save_to_file(text, tmp_path)
        
        # Threaded runAndWait with timeout
        done_event = threading.Event()
        def _run():
            eng.runAndWait()
            done_event.set()
        thread = threading.Thread(target=_run)
        thread.start()
        if not done_event.wait(timeout=20):  # Increased to 20s for safety
            raise TimeoutError("TTS synthesis timed out - check audio device or voice.")
        print("done.")
        
        with open(tmp_path, "rb") as f:
            wav_data = f.read()
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
        try:
            eng.stop()  # Ensure cleanup
        except Exception:
            pass

    # ---------- Resample (fast path) ----------
    if sr != target_sr:
        print(f"Resampling audio from {sr} Hz to {target_sr} Hz...", end=" ", flush=True)
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
# Fast resample (torch-audio if available, else vectorised numpy)
# =========================
_resampler: Optional[Any] = None
def _init_resampler():
    global _resampler
    if _resampler is None:
        print("Initializing resampler (torchaudio if available)...", end=" ", flush=True)
        try:
            import torch
            import torchaudio
            _resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
            if torch.cuda.is_available():
                _resampler = _resampler.to('cuda')
            print("torchaudio resampler ready.")
        except Exception as e:
            print(f"torchaudio init failed: {e}")
            _resampler = False
            print("falling back to numpy interpolation.")

def _fast_resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    _init_resampler()
    if _resampler:
        import torch
        tensor = torch.from_numpy(y).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.to('cuda')
        with torch.no_grad():
            out = _resampler(tensor).cpu().squeeze(0).numpy()
        return out
    old_t = np.linspace(0, len(y) / orig_sr, num=len(y), endpoint=False)
    new_len = int(len(y) * target_sr / orig_sr)
    new_t = np.linspace(0, len(y) / orig_sr, num=new_len, endpoint=False)
    return np.interp(new_t, old_t, y).astype(np.float32)

# =========================
# Audio metrics
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
# TTS evaluation
# =========================
def run_tts_eval(text: str, rate: int = 180, target_sr: int = 16000) -> Dict[str, Any]:
    t0 = time.perf_counter()
    wav_bytes = _tts_to_wav_bytes(text, rate=rate, target_sr=target_sr)
    synth_time = time.perf_counter() - t0

    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

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
# Whisper / Transcription
# =========================
_have_whisper = False
_asr_model: Optional[Any] = None
def _load_whisper_base(use_cuda: bool = False):
    global _asr_model, _have_whisper
    if _asr_model is None:
        print("Loading OpenAI Whisper model...", end=" ", flush=True)
        import whisper
        _have_whisper = True
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        _asr_model = whisper.load_model("base", device=device)
        print(f"OpenAI Whisper loaded on {device}")
    return _asr_model

def _have_faster_whisper() -> bool:
    try:
        import faster_whisper
        return True
    except Exception:
        return False

_fw_models: Dict[str, Any] = {}
def _load_faster_whisper(model_size: str = "base", device: str = "cpu", compute_type: str = "float16"):
    if model_size in _fw_models:
        return _fw_models[model_size]
    print(f"Loading faster-whisper model ({model_size}) on {device}...", end=" ", flush=True)
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _fw_models[model_size] = model
        print(f"faster-whisper loaded on {device} with {compute_type}")
        return model
    except Exception as e:
        print(f"faster-whisper load failed: {e}")
        return None

def fast_transcribe(wav_bytes: bytes, language: Optional[str] = "en", model_size: str = "base", use_cuda: bool = False) -> Optional[str]:
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != 16000:
        data = _fast_resample(data, sr, 16000)

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    if _have_faster_whisper():
        try:
            model = _load_faster_whisper(model_size, device=device, compute_type=compute_type)
            if model:
                print("Transcribing with faster-whisper...", end=" ", flush=True)
                segments, _ = model.transcribe(data, language=language, beam_size=1)
                text = " ".join([s.text for s in segments]).strip()
                print("done.")
                return text
        except Exception as e:
            print(f"faster-whisper failed: {e}")

    try:
        asr = _load_whisper_base(use_cuda=use_cuda)
        print("Falling back to OpenAI Whisper...", end=" ", flush=True)
        out = asr.transcribe(data, language=language, fp16=use_cuda)
        print("done.")
        return out.get("text", "").strip()
    except Exception as e:
        print(f"Whisper fallback failed: {e}")
        return None

# =========================
# WER
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
# Test sentences
# =========================
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Set the gain to 3.5 decibels at two kilohertz.",
    "Wait… are you sure? Okay—let's begin.",
    "Backpropagation adjusts weights via gradient descent.",
    "Gracias por tu ayuda, see you mañana.",
    "In 2025, open source text-to-speech systems continue to improve in both naturalness and controllability, making high-quality local speech synthesis practical on everyday laptops. Careful punctuation dramatically improves prosody: commas, periods, and dashes guide the rhythm and phrasing of the spoken output."
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
    parser.add_argument("--out", default="tts_sample.wav", help="Output WAV for first sample or all-in-one")
    parser.add_argument("--rate", type=int, default=180, help="TTS voice rate")
    parser.add_argument("--voice", type=str, default=None, help="TTS voice ID (e.g., HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Microsoft\\\\Speech\\\\Voices\\\\Tokens\\\\TTS_MS_EN-US_ZIRA_11.0)")
    parser.add_argument("--max", type=int, default=min(4, len(TEST_SENTENCES)), help="Max sentences")
    parser.add_argument("--wer-first-n", type=int, default=min(2, len(TEST_SENTENCES)), help="Compute WER for first N (0 = disable)")
    parser.add_argument("--cuda", action="store_true", help="Run Whisper on GPU")
    parser.add_argument("--all-in-one", action="store_true", help="Concatenate all sentences into one WAV (with metrics)")
    parser.add_argument("--silence-sec", type=float, default=0.5, help="Silence seconds between sentences in concat (default 0.5)")
    args = parser.parse_args()

    print("Starting TTS evaluation script...")

    try:
        import torch
        if args.cuda and torch.cuda.is_available():
            print(f"CUDA verified: {torch.cuda.get_device_name(0)}")
        elif args.cuda:
            print("CUDA requested but not available - CPU fallback")
    except Exception as e:
        print(f"Torch issue: {e}")

    print("== Fast TTS baseline ==")
    rows: List[Dict[str, Any]] = []
    first_wav: Optional[bytes] = None
    synth_buffers: List[Tuple[np.ndarray, int]] = []  # For concat: (audio_data, sr)

    if args.wer_first_n > 0:
        print("Warming up Whisper models...", end=" ", flush=True)
        device = "cuda" if args.cuda else "cpu"
        _load_faster_whisper(device=device)
        _load_whisper_base(use_cuda=args.cuda)
        print("done.")

    for i, txt in enumerate(TEST_SENTENCES[:args.max], 1):
        print(f"\nProcessing sentence {i}/{min(args.max, len(TEST_SENTENCES))}:")
        res = run_tts_eval(txt, rate=args.rate)
        wav = res.pop("wav_bytes")
        if first_wav is None:
            first_wav = wav

        # Read audio data for potential concat
        data, sr = sf.read(io.BytesIO(wav), dtype="float32")
        if data.ndim > 1: data = data.mean(axis=1)
        synth_buffers.append((data, sr))

        # Optional WER
        w = None
        if args.wer_first_n and i <= args.wer_first_n:
            print(f"Computing WER for sentence {i}...", end=" ", flush=True)
            hyp = fast_transcribe(wav, use_cuda=args.cuda)
            if hyp: w = wer(txt, hyp)
            print("done.")

        rows.append({"idx": i, **res, "wer": w})

    # Print metrics table
    _print_table(rows)

    # Save outputs
    if args.all_in_one:
        print(f"Concatenating all {len(synth_buffers)} sentences into {args.out}...", end=" ", flush=True)
        if synth_buffers:
            pad_samples = int(args.silence_sec * synth_buffers[0][1])
            pad = np.zeros(pad_samples, dtype=np.float32)
            parts: List[np.ndarray] = []
            target_sr = synth_buffers[0][1]
            for data, s in synth_buffers:
                if s != target_sr:
                    data = _fast_resample(data, s, target_sr)
                parts.append(data)
                parts.append(pad)
            merged = np.concatenate(parts[:-1])  # No trailing pad
            sf.write(args.out, merged, target_sr)
            print("done.")
        print(f"Saved concatenated audio → {args.out}")
    else:
        if first_wav:
            with open(args.out, "wb") as f: f.write(first_wav)
            print(f"\nSaved first sample → {args.out}")

    # Warnings
    bad = [r for r in rows if r["peak_dbfs"] > -0.1 or r["rtf"] > 1.0]
    if bad:
        print("\nWarnings:")
        for r in bad:
            if r["peak_dbfs"] > -0.1: print(f"  • Row {r['idx']}: clipping risk")
            if r["rtf"] > 1.0: print(f"  • Row {r['idx']}: slow RTF")
    print("\nDone.")

if __name__ == "__main__":
    try:
        import torch
    except Exception:
        torch = None
    main()