# voice_metrics.py
# Tiny TTS→ASR QA harness (plug-your-own engines)
# This creates /mnt/data/tts_qaharness.py for you to download and run locally.
from typing import List, Dict, Tuple, Optional
import json, math, re, io, os, wave, struct
from dataclasses import dataclass, asdict

import numpy as np

HARNESS_PATH = "/mnt/data/tts_qaharness.py"

code = r'''
"""
Tiny TTS→ASR QA Harness
-----------------------
Plug your own TTS and ASR in `synthesize_tts()` and `transcribe_asr()`.

What it does:
1) Synthesizes audio for each test prompt (TTS).
2) Transcribes the audio back to text (ASR).
3) Computes:
   - WER (word error rate) vs provided reference
   - Speaking rate (words per minute)
   - Pause validation (silence detection, and optional word-timestamp checks if ASR returns them)
   - Approximate SNR (optional, heuristic)
4) Produces a per-item report + overall summary.

Usage (example):
    python tts_qaharness.py inputs.json out_report.json

The `inputs.json` should follow the flattened schema you got earlier, e.g.:
{
  "schema_version": "1.1-flat",
  "items": [
    {"id":"A1","cat":"pangram","text":"The quick brown fox...","ref":"The quick brown fox...","wer_max":0.05,"wpm":"150-190","snr_min":20}
  ]
}

Notes:
- This harness is engine-agnostic. Implement `synthesize_tts` and `transcribe_asr` to connect to your stack.
- If your ASR can return word timestamps, return them as a list of (word, start_sec, end_sec).
- Pause rules are parsed from strings like: "after=Wait;type=ellipsis;min=250;max=600 | after=sure;type=question;min=150;max=400"
"""

import sys, json, math, re, os, io, wave, struct
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np

# --------------------------- Adapters to implement ---------------------------

def synthesize_tts(text: str, voice: str = "default") -> bytes:
    """
    Return raw WAV bytes (16-bit PCM, 16 kHz mono) for the given text.
    IMPLEMENT ME: Replace with your TTS engine call.
    For demonstration, this returns a synthetic "beeps+silences" WAV to exercise pause logic.
    """
    sr = 16000
    # Generate simple tone for each word with short silence gaps; commas/periods create longer gaps
    words = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    audio = []
    def tone(dur_s: float, freq=440.0, amp=0.1):
        t = np.arange(int(sr*dur_s)) / sr
        return (amp*np.sin(2*np.pi*freq*t)).astype(np.float32)
    def silence(dur_s: float):
        return np.zeros(int(sr*dur_s), dtype=np.float32)
    for w in words:
        # beep length ~ 120ms per token
        audio.append(tone(0.12))
        # pause heuristic
        if w in [","]:
            audio.append(silence(0.12))
        elif w in [";", "—", "-"]:
            audio.append(silence(0.20))
        elif w in [".", "!", "?"]:
            audio.append(silence(0.30))
        elif w in ["…", "..."]:
            audio.append(silence(0.35))
        else:
            audio.append(silence(0.06))
    y = np.concatenate(audio) if audio else np.zeros(1, dtype=np.float32)
    # float32 -> int16 PCM WAV
    y_clipped = np.clip(y, -1.0, 1.0)
    pcm = (y_clipped * 32767.0).astype(np.int16).tobytes()
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        return buf.getvalue()

def transcribe_asr(wav_bytes: bytes) -> Tuple[str, Optional[List[Tuple[str, float, float]]]]:
    """
    Return (transcript_text, word_timestamps) from your ASR.
    IMPLEMENT ME: Replace with your ASR engine call.
    For demonstration, we "fake" an ASR by returning a simple constant string.
    """
    # In real use: call your ASR and return words + timestamps (if available).
    # Here we return empty timestamps and an empty transcript (forces WER>0 unless ref empty).
    return "", None

# --------------------------- Metric utilities -------------------------------

def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?", s.lower())

def levenshtein_distance(a: List[str], b: List[str]) -> int:
    # Classic DP
    dp = np.zeros((len(a)+1, len(b)+1), dtype=np.int32)
    for i in range(len(a)+1): dp[i,0] = i
    for j in range(len(b)+1): dp[0,j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i,j] = min(dp[i-1,j] + 1, dp[i,j-1] + 1, dp[i-1,j-1] + cost)
    return int(dp[len(a), len(b)])

def wer(ref: str, hyp: str) -> float:
    r = tokenize_words(ref)
    h = tokenize_words(hyp)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return levenshtein_distance(r, h) / len(r)

def parse_wpm_range(s: str) -> Tuple[Optional[float], Optional[float]]:
    if not s: return (None, None)
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m: return (None, None)
    return float(m.group(1)), float(m.group(2))

def wav_duration_seconds(wav_bytes: bytes) -> float:
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            return frames / float(sr)

def decode_wav_to_np(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            s = wf.readframes(n)
            y = np.frombuffer(s, dtype=np.int16).astype(np.float32) / 32767.0
            return y, sr

def estimate_snr_db(y: np.ndarray, frame_len: int = 1024, hop: int = 512) -> float:
    # Heuristic: noise floor ~ 10th percentile of frame RMS, signal ~ median RMS
    if len(y) < frame_len: return 0.0
    frames = []
    for i in range(0, len(y)-frame_len+1, hop):
        fr = y[i:i+frame_len]
        rms = math.sqrt(float(np.mean(fr**2)) + 1e-12)
        frames.append(rms)
    if not frames: return 0.0
    frames = np.array(frames)
    noise = np.percentile(frames, 10)
    signal = np.percentile(frames, 50)
    if noise <= 1e-9: noise = 1e-9
    return 20.0 * math.log10(signal / noise)

def detect_silences(y: np.ndarray, sr: int, thresh: float = 0.005, min_sil_ms: int = 60) -> List[Tuple[float, float]]:
    """
    Simple energy-based silence detection.
    Returns list of (start_sec, end_sec) where |y| < thresh for >= min_sil_ms.
    """
    mask = np.abs(y) < thresh
    min_len = int(sr * (min_sil_ms / 1000.0))
    silences = []
    in_sil = False
    start = 0
    for i, m in enumerate(mask):
        if m and not in_sil:
            in_sil = True; start = i
        elif not m and in_sil:
            if i - start >= min_len:
                silences.append((start/sr, i/sr))
            in_sil = False
    if in_sil and len(y) - start >= min_len:
        silences.append((start/sr, len(y)/sr))
    return silences

def validate_pauses_rulestring(rule: Optional[str], y: np.ndarray, sr: int,
                               word_timestamps: Optional[List[Tuple[str,float,float]]] = None) -> Dict:
    """
    If word_timestamps provided, we can validate specific post-token pauses.
    Otherwise we fall back to overall silence statistics.
    Rule string format example:
      "after=Wait;type=ellipsis;min=250;max=600 | after=sure;type=question;min=150;max=400"
    Returns dict with pass/fail and details.
    """
    if not rule:
        return {"checked": False, "reason": "no_rules"}

    # Parse rules into list of dicts
    chunks = [c.strip() for c in rule.split("|")]
    rules = []
    for c in chunks:
        d = {}
        for kv in c.split(";"):
            kv = kv.strip()
            if not kv: continue
            if "=" in kv:
                k,v = kv.split("=",1)
                d[k.strip()] = v.strip()
            else:
                # positional type like "comma"
                if "type" not in d:
                    d["type"] = kv.strip()
        if d: rules.append(d)

    results = []
    if word_timestamps:
        # Build quick index of word->last end time (case-insensitive)
        last_end = {}
        for w, s, e in word_timestamps:
            last_end[w.lower()] = max(last_end.get(w.lower(), 0.0), e)
        # Need first speech start after that word; approximate from timestamps
        starts = sorted([s for _, s, _ in word_timestamps])
        for r in rules:
            tok = r.get("after", "").lower()
            t_end = last_end.get(tok, None)
            if t_end is None:
                results.append({"rule": r, "pass": False, "reason": f"token '{tok}' not found"})
                continue
            # Next start after t_end
            next_starts = [s for s in starts if s > t_end]
            if not next_starts:
                # pause until end of audio
                pause_dur_ms = (len(y)/sr - t_end)*1000.0
            else:
                pause_dur_ms = (next_starts[0] - t_end)*1000.0
            min_ms = float(r.get("min", 0))
            max_ms = float(r.get("max", 1e9))
            ok = (pause_dur_ms >= min_ms) and (pause_dur_ms <= max_ms)
            results.append({"rule": r, "pass": ok, "measured_ms": round(pause_dur_ms,1)})
    else:
        # Fallback: measure silences and check if any silence falls within ranges
        sils = detect_silences(y, sr)
        sil_ms = [ (b-a)*1000.0 for (a,b) in sils ]
        for r in rules:
            min_ms = float(r.get("min", 0))
            max_ms = float(r.get("max", 1e9))
            ok = any((d>=min_ms and d<=max_ms) for d in sil_ms)
            results.append({"rule": r, "pass": ok, "matched_silences_ms": [round(d,1) for d in sil_ms if (d>=min_ms and d<=max_ms)]})

    all_ok = all(x.get("pass", False) for x in results) if results else False
    return {"checked": True, "all_rules_pass": all_ok, "details": results}

# --------------------------- Main evaluation loop ---------------------------

def evaluate_item(item: Dict) -> Dict:
    text = item["text"]
    ref  = item.get("ref", text)
    wer_max = float(item.get("wer_max", 0.1))
    wpm_range = parse_wpm_range(item.get("wpm",""))
    snr_min = float(item.get("snr_min", 0))

    wav_bytes = synthesize_tts(text)
    hyp, word_ts = transcribe_asr(wav_bytes)

    # Metrics
    dur_s = wav_duration_seconds(wav_bytes)
    n_words = max(1, len(tokenize_words(hyp or text)))  # use hyp if available else rough estimate from text
    wpm = (n_words / max(dur_s, 1e-6)) * 60.0

    y, sr = decode_wav_to_np(wav_bytes)
    snr_db = estimate_snr_db(y)

    # WER (vs ref)
    item_wer = wer(ref, hyp)

    # Pauses
    pauses_rule = item.get("pauses", None)
    pause_eval = validate_pauses_rulestring(pauses_rule, y, sr, word_ts)

    # Pass/fail
    pass_wer = (item_wer <= wer_max)
    pass_wpm = True
    if all(v is not None for v in wpm_range):
        lo, hi = wpm_range
        pass_wpm = (wpm >= lo) and (wpm <= hi)
    pass_snr = (snr_db >= snr_min) if snr_min > 0 else True
    pass_pauses = (pause_eval.get("all_rules_pass", True) if pause_eval.get("checked") else True)

    return {
        "id": item.get("id"),
        "cat": item.get("cat"),
        "text": text,
        "ref": ref,
        "hyp": hyp,
        "duration_s": round(dur_s, 3),
        "wpm": round(wpm, 1),
        "wpm_bounds": list(wpm_range) if all(v is not None for v in wpm_range) else None,
        "snr_db": round(snr_db, 1),
        "snr_min": snr_min,
        "wer": round(item_wer, 3),
        "wer_max": wer_max,
        "pauses_checked": pause_eval.get("checked", False),
        "pauses_result": pause_eval,
        "pass_wer": pass_wer,
        "pass_wpm": pass_wpm,
        "pass_snr": pass_snr,
        "pass_pauses": pass_pauses,
        "pass_overall": bool(pass_wer and pass_wpm and pass_snr and pass_pauses)
    }

def run_eval(inp_path: str, out_path: Optional[str] = None) -> Dict:
    data = json.load(open(inp_path, "r", encoding="utf-8"))
    items = data.get("items", data.get("tts_test_inputs", []))
    results = [evaluate_item(it) for it in items]
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["pass_overall"]),
        "failed": sum(1 for r in results if not r["pass_overall"]),
        "avg_wer": round(float(np.mean([r["wer"] for r in results])) if results else 0.0, 3),
        "avg_wpm": round(float(np.mean([r["wpm"] for r in results])) if results else 0.0, 1),
        "avg_snr_db": round(float(np.mean([r["snr_db"] for r in results])) if results else 0.0, 1)
    }
    report = {"summary": summary, "results": results}
    if out_path:
        json.dump(report, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tts_qaharness.py <inputs.json> [out_report.json]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    rep = run_eval(inp, out)
    print(json.dumps(rep["summary"], indent=2))
'''

# Write the harness file
with open(HARNESS_PATH, "w", encoding="utf-8") as f:
    f.write(code)

HARNESS_PATH
