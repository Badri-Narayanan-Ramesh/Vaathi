# text_2_speech_fast_v2_neural.py
"""
Faster neural TTS baseline (drop-in for your previous script that used pyttsx3)

What it does:
- Coqui TTS (neural) → WAV bytes (no temp files needed)
- Optional fast resampling via torchaudio (falls back to vectorized NumPy)
- Optional faster-whisper (base) for much faster CPU WER; falls back to openai-whisper
- Optional CUDA for both Whisper and TTS
- Engine timeouts not needed (no pyttsx3), same metrics & -3 dBFS headroom
- All-in-one concatenation mode

CLI examples:
  python text_2_speech_fast_v2_neural.py
  python text_2_speech_fast_v2_neural.py --rate 1.0 --max 4 --wer-first-n 2 --tts-gpu
  python text_2_speech_fast_v2_neural.py --all-in-one --silence-sec 0.5 --tts-model tts_models/en/ljspeech/vits
"""

from __future__ import annotations
import io, os, time, math, argparse, re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import soundfile as sf

# Reduce thread contention warnings on some BLAS builds
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# =========================
# Fast resample (torchaudio cached by (orig,target); else vectorized NumPy)
# =========================
_resamplers: Dict[Tuple[int, int], Any] = {}

def _get_resampler(orig_sr: int, target_sr: int):
    key = (orig_sr, target_sr)
    if key in _resamplers:
        return _resamplers[key]
    try:
        import torch
        import torchaudio
        rs = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        if torch.cuda.is_available():
            rs = rs.to("cuda")
        _resamplers[key] = rs
        return rs
    except Exception:
        _resamplers[key] = None
        return None

def _fast_resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or y.size == 0:
        return y.astype(np.float32, copy=False)
    rs = _get_resampler(orig_sr, target_sr)
    if rs is not None:
        import torch
        ten = torch.from_numpy(y).unsqueeze(0)
        if torch.cuda.is_available():
            ten = ten.to("cuda")
        with torch.no_grad():
            out = rs(ten).cpu().squeeze(0).numpy()
        return out.astype(np.float32, copy=False)
    # Fallback: vectorized linear interpolation
    old_t = np.linspace(0, len(y) / orig_sr, num=len(y), endpoint=False)
    new_len = int(round(len(y) * target_sr / orig_sr))
    new_t = np.linspace(0, len(y) / orig_sr, num=new_len, endpoint=False)
    return np.interp(new_t, old_t, y).astype(np.float32)

# =========================
# Metrics
# =========================
def _peak_dbfs(x: np.ndarray) -> float:
    if x.size == 0: return -math.inf
    peak = float(np.max(np.abs(x)))
    return 20 * math.log10(max(peak, 1e-12))

def _rms_db(x: np.ndarray) -> float:
    if x.size == 0: return -math.inf
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    return 20 * math.log10(rms)

def estimate_loudness_proxy_lufs(x: np.ndarray, sr: int) -> float:
    # Lightweight proxy: use RMS dB
    return _rms_db(x)

def estimate_snr_db(x: np.ndarray, sr: int, sil_ms: int = 300) -> float:
    n = max(1, int(sr * sil_ms / 1000))
    if x.size < 2 * n:
        noise = x[:n]
    else:
        noise = np.concatenate([x[:n], x[-n:]])
    return _rms_db(x) - _rms_db(noise)

# =========================
# Neural TTS (Coqui TTS)
# =========================
_tts_model = None
_tts_sr = 22050

def _load_coqui_tts(model_name: str = "tts_models/en/ljspeech/vits", use_gpu: bool = False):
    global _tts_model, _tts_sr
    if _tts_model is not None:
        return _tts_model
    from TTS.api import TTS
    # progress_bar=False avoids tqdm spam in CLI
    _tts_model = TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)
    # Try to get model's native output SR, fallback to 22.05k
    _tts_sr = getattr(getattr(_tts_model, "synthesizer", None), "output_sample_rate", 22050)
    return _tts_model

def _neural_tts_to_wav_bytes(
    text: str,
    rate: float = 1.0,        # playback/speaking rate multiplier (1.0 = normal)
    target_sr: int = 16000,
    model_name: str = "tts_models/en/ljspeech/vits",
    use_gpu: bool = False,
    speaker: Optional[str] = None,
    language: Optional[str] = None,
) -> bytes:
    """
    Synthesize with Coqui TTS into a WAV bytes buffer.
    - Peak-normalized to -3 dBFS
    - Optional resampling to target_sr
    - rate: speaking-rate multiplier (if model supports it; otherwise we post-resample in time)
    """
    tts = _load_coqui_tts(model_name=model_name, use_gpu=use_gpu)

    # Some TTS models support a 'speed' param; TTS.api unifies as 'speed' in many models.
    # We'll try native support; if not, we time-stretch as a fallback (simple resample-in-time).
    supports_speed = "speed" in tts.synthesizer.tts_forward_args if hasattr(tts, "synthesizer") and hasattr(tts.synthesizer, "tts_forward_args") else False

    if supports_speed:
        wav = tts.tts(text=text, speaker=speaker, language=language, speed=rate)
        sr = _tts_sr
    else:
        wav = tts.tts(text=text, speaker=speaker, language=language)
        sr = _tts_sr
        if abs(rate - 1.0) > 1e-3 and len(wav) > 0:
            # Simple time-scale via resampling the time axis
            new_len = int(round(len(wav) / max(rate, 1e-6)))
            t_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False, dtype=np.float32)
            t_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float32)
            wav = np.interp(t_new, t_old, wav).astype(np.float32)

    y = np.asarray(wav, dtype=np.float32)

    # Normalize to -3 dBFS headroom
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    target_peak = 10 ** (-3.0 / 20.0)
    if peak > 0:
        y = y * (target_peak / max(peak, 1e-12))

    # Resample to target_sr if needed
    if sr != target_sr:
        y = _fast_resample(y, sr, target_sr)
        sr = target_sr

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()

# =========================
# TTS eval wrapper (unchanged interface)
# =========================
def run_tts_eval(text: str, rate: float = 1.0, target_sr: int = 16000,
                 model_name: str = "tts_models/en/ljspeech/vits", tts_gpu: bool = False,
                 speaker: Optional[str] = None, language: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    wav_bytes = _neural_tts_to_wav_bytes(
        text, rate=rate, target_sr=target_sr, model_name=model_name,
        use_gpu=tts_gpu, speaker=speaker, language=language,
    )
    synth_time = time.perf_counter() - t0

    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    pad = np.zeros(int(0.2 * sr), dtype=np.float32)
    y_pad = np.concatenate([pad, y, pad])
    dur = len(y) / sr
    rtf = synth_time / max(dur, 1e-6)

    return {
        "latency_s": round(synth_time, 3),
        "duration_s": round(dur, 3),
        "rtf": round(rtf, 3),
        "peak_dbfs": round(_peak_dbfs(y_pad), 2),
        "snr_db": round(estimate_snr_db(y_pad, sr), 2),
        "lufs_proxy_db": round(estimate_loudness_proxy_lufs(y_pad, sr), 2),
        "sr": sr,
        "wav_bytes": wav_bytes,
    }

# =========================
# ASR (prefer faster-whisper base; fallback to openai-whisper base)
# =========================
_fw_models: Dict[Tuple[str, str, str], Any] = {}
_asr_model = None

def _load_faster_whisper(model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
    key = (model_size, device, compute_type)
    if key in _fw_models:
        return _fw_models[key]
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _fw_models[key] = model
        return model
    except Exception:
        return None

def _load_whisper_base(use_cuda: bool = False):
    global _asr_model
    if _asr_model is not None:
        return _asr_model
    import whisper
    device = "cpu"
    try:
        if use_cuda:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
    except Exception:
        pass
    _asr_model = whisper.load_model("base", device=device)
    return _asr_model

def fast_transcribe(
    wav_bytes: bytes,
    language: Optional[str] = "en",
    model_size: str = "base",
    use_cuda: bool = False
) -> Optional[str]:
    # Bytes -> mono float32 @ 16k
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    except Exception:
        return None
    data = data.mean(axis=1)
    if sr != 16000:
        data = _fast_resample(data, sr, 16000)

    # Try faster-whisper first
    device = "cpu"
    compute_type = "int8"
    if use_cuda:
        try:
            import torch
            if torch.cuda.is_available():
                device, compute_type = "cuda", "float16"
        except Exception:
            pass
    fw = _load_faster_whisper(model_size=model_size, device=device, compute_type=compute_type)
    if fw is not None:
        try:
            segments, _ = fw.transcribe(
                data,
                language=language,
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=150),
                condition_on_previous_text=False
            )
            return "".join(seg.text for seg in segments).strip()
        except Exception:
            pass

    # Fallback: openai-whisper base
    try:
        asr = _load_whisper_base(use_cuda=use_cuda)
        out = asr.transcribe(data, language=language, fp16=(device == "cuda"))
        return out.get("text", "").strip()
    except Exception:
        return None

# =========================
# WER
# =========================
def _tok(s: str) -> List[str]:
    return re.findall(r"\w+(?:'\w+)?", s.lower())

def wer(ref: str, hyp: str) -> float:
    r, h = _tok(ref), _tok(hyp)
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): dp[i][0] = i
    for j in range(len(h)+1): dp[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
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
    ("In 2025, open source text-to-speech systems continue to improve in both naturalness and controllability, "
     "making high-quality local speech synthesis practical on everyday laptops. Careful punctuation dramatically "
     "improves prosody: commas, periods, and dashes guide the rhythm and phrasing of the spoken output."),
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
            (f"{r['wer']:.2f}" if r.get("wer") is not None else "—"),
        ]))

def main():
    parser = argparse.ArgumentParser(description="Neural TTS baseline + metrics + Whisper WER")
    parser.add_argument("--out", default="tts_sample.wav", help="Output WAV for first sample or all-in-one")
    parser.add_argument("--rate", type=float, default=1.0, help="Speaking-rate multiplier (1.0 = normal)")
    parser.add_argument("--max", type=int, default=min(4, len(TEST_SENTENCES)), help="Max sentences to run")
    parser.add_argument("--wer-first-n", type=int, default=min(2, len(TEST_SENTENCES)),
                        help="Compute WER for first N sentences (0 disables WER)")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for Whisper if available")
    parser.add_argument("--all-in-one", action="store_true", help="Concatenate all sentences into one WAV")
    parser.add_argument("--silence-sec", type=float, default=0.5, help="Silence between sentences in concat")
    # NEW: neural TTS options
    parser.add_argument("--tts-model", type=str, default="tts_models/en/ljspeech/vits",
                        help="Coqui TTS model name (e.g., tts_models/en/ljspeech/vits, tts_models/en/ljspeech/tacotron2-DDC)")
    parser.add_argument("--tts-gpu", action="store_true", help="Use GPU for neural TTS")
    parser.add_argument("--tts-speaker", type=str, default=None, help="Speaker id/name if the model supports multispeaker")
    parser.add_argument("--tts-language", type=str, default=None, help="Language code if the model supports it (e.g., 'en')")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate for output")
    args = parser.parse_args()

    print("== Neural TTS baseline (Coqui) ==")
    rows: List[Dict[str, Any]] = []
    first_wav: Optional[bytes] = None
    synth_buffers: List[Tuple[np.ndarray, int]] = []

    # Optional warm-up for ASR (small cost, avoids first-call hit)
    if args.wer_first_n > 0:
        device = "cuda" if args.cuda else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _ = _load_faster_whisper(model_size="base", device=device, compute_type=compute_type)
        _ = _load_whisper_base(use_cuda=args.cuda)

    # Warm-up TTS model (avoid first-call download latency/graph build)
    _ = _load_coqui_tts(model_name=args.tts_model, use_gpu=args.tts_gpu)

    for i, txt in enumerate(TEST_SENTENCES[:args.max], 1):
        res = run_tts_eval(
            txt, rate=args.rate, target_sr=args.target_sr,
            model_name=args.tts_model, tts_gpu=args.tts_gpu,
            speaker=args.tts_speaker, language=args.tts_language
        )
        wav = res.pop("wav_bytes")
        if first_wav is None:
            first_wav = wav

        data, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        synth_buffers.append((data, sr))

        w = None
        if args.wer_first_n and i <= args.wer_first_n:
            hyp = fast_transcribe(wav, language="en", model_size="base", use_cuda=args.cuda)
            if hyp:
                w = wer(txt, hyp)

        rows.append({"idx": i, **res, "wer": w})

    _print_table(rows)

    if args.all_in_one and synth_buffers:
        # Concatenate with silence
        target_sr = synth_buffers[0][1]
        pad = np.zeros(int(args.silence_sec * target_sr), dtype=np.float32)
        parts = []
        for j, (d, srr) in enumerate(synth_buffers):
            if srr != target_sr:
                d = _fast_resample(d, srr, target_sr)
            parts.append(d)
            if j != len(synth_buffers) - 1:
                parts.append(pad)
        merged = np.concatenate(parts)
        sf.write(args.out, merged, target_sr)
        print(f"\nSaved concatenated audio → {args.out}")
    elif first_wav:
        with open(args.out, "wb") as f:
            f.write(first_wav)
        print(f"\nSaved first sample → {args.out}")

    # Hints
    bad = [r for r in rows if r["peak_dbfs"] > -0.1 or r["rtf"] > 1.0]
    if bad:
        print("\nWarnings:")
        for r in bad:
            if r["peak_dbfs"] > -0.1:
                print(f"  • Row {r['idx']}: clipping risk (peak too hot)")
            if r["rtf"] > 1.0:
                print(f"  • Row {r['idx']}: synthesis slower than real-time")
    print("\nDone.")

if __name__ == "__main__":
    main()
