import io
import os
import tempfile
import numpy as np
import soundfile as sf
import pyttsx3

def tts_bytes(text: str, rate: int = 180, voice: str | None = None) -> bytes:
    """
    Convert text to spoken audio and return WAV bytes.
    Uses pyttsx3 (offline, cross-platform).

    Args:
        text: Text to be spoken.
        rate: Speaking rate (default 180).
        voice: Optional voice id from system voices.

    Returns:
        WAV audio bytes (16-bit float32).
    """
    engine = pyttsx3.init()
    for v in eng.getProperty("voices"):
        print(v.id)
        
    engine.setProperty("rate", rate)
    if voice:
        engine.setProperty("voice", voice)

    # temporary file for saving synthesized audio
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    engine.save_to_file(text, tmp_path)
    engine.runAndWait()

    # read file → bytes
    data, sr = sf.read(tmp_path, dtype="float32")
    os.remove(tmp_path)

    buf = io.BytesIO()
    sf.write(buf, np.array(data), sr, format="WAV")
    return buf.getvalue()


import torch
import time

def benchmark_tts(tts_model, text):
    # assuming tts_model has a `.synthesize(text)` method returning waveform tensor
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wav = tts_model.synthesize(text)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    mem_alloc = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
    return elapsed_ms, mem_alloc

def run_tests():
    texts = [
        "Hello world, this is a test of text to speech.",
        "In 2025, open source TTS models are rapidly improving in expressiveness."
    ]
    models = {
        "Dia": load_dia_model(),  # your load code
        "XTTS-v2": load_xtts2_model(),
        "Mozilla-tts": load_mozilla_tts_model()
    }
    results = {}
    for name, model in models.items():
        model = model.to('cuda').half()
        for t in texts:
            time_ms, mem = benchmark_tts(model, t)
            print(f"{name} | {len(t)} chars → {time_ms:.1f} ms, {mem:.1f} MB")
        print()
    return results

if __name__ == "__main__":
    run_tests()

# ready-to-run scripts / Docker containers for 2–3 of these TTS models (Dia, XTTS-v2, Mozilla) tailored for your GPU, so you can test them immediately.

# wer_eval.py
import re
def _tok(s): return re.findall(r"\w+(?:'\w+)?", s.lower())
def wer(ref, hyp):
    r, h = _tok(ref), _tok(hyp)
    # Levenshtein distance
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): dp[i][0]=i
    for j in range(len(h)+1): dp[0][j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[len(r)][len(h)] / max(1,len(r))

# audio_checks.py
import io, time, numpy as np, soundfile as sf
from tts_basic import tts_bytes

def peak_dbfs(x):
    peak = np.max(np.abs(x))
    return 20*np.log10(max(peak, 1e-12))

def rms_db(x):
    return 20*np.log10(np.sqrt(np.mean(np.square(x))+1e-12))

def estimate_loudness_lufs(x, sr):
    # lightweight proxy: gated RMS over whole file (OK for hackathon)
    return rms_db(x) - 0  # placeholder offset; treat like relative metric

def estimate_snr_db(x, sr, sil_ms=300):
    n = int(sr*sil_ms/1000)
    head = x[:n]; tail = x[-n:]
    noise = np.concatenate([head, tail])
    return rms_db(x) - rms_db(noise)

def run_tts_eval(text):
    t0 = time.time()
    wav = tts_bytes(text)
    synth_time = time.time() - t0
    y, sr = sf.read(io.BytesIO(wav), dtype="float32")
    dur = len(y)/sr
    return {
        "latency_s": round(synth_time,3),
        "duration_s": round(dur,3),
        "rtf": round(synth_time/dur,3),
        "peak_dbfs": round(peak_dbfs(y),2),
        "snr_db": round(estimate_snr_db(y, sr),2),
        "lufs_proxy_db": round(estimate_loudness_lufs(y, sr),2),
        "sr": sr
    }

# stt_roundtrip.py
import io, soundfile as sf, whisper
asr = whisper.load_model("base")

def transcribe_wav_bytes(wav_bytes, language=None):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        sf.write(f.name, data, sr)
        out = asr.transcribe(f.name, language=language)
    return out.get("text","").strip()