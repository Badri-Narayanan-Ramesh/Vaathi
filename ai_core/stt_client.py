"""Small wrapper around whisper_hf_asr.transcribe to return cleaned text, language, and confidence.

Always returns a dict: {text, language, confidence, raw, error}
"""
from __future__ import annotations

from typing import Dict, Any, Optional




def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    s = text.strip()
    # remove role tokens like 'user:' or 'assistant:' if present at start
    lower = s.lower()
    if lower.startswith("user:") or lower.startswith("assistant:") or lower.startswith("system:"):
        s = s.split(":", 1)[1].strip()
    # collapse multiple spaces
    s = " ".join(s.split())
    return s


def transcribe_wav_bytes(wav_bytes: bytes, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe WAV bytes using whisper_hf_asr.transcribe.

    Returns a dict with keys: text, language, confidence, raw, error.
    Always returns without raising.
    """
    try:
        import whisper_hf_asr  # lazy import to avoid hard dependency at module import time

        res = whisper_hf_asr.transcribe(wav_bytes, language=language, return_segments=False)
    except Exception as e:
        return {"text": "", "language": language or "", "confidence": None, "raw": None, "error": str(e)}

    # Expected to be a dict with 'text' and maybe 'language' and 'confidence'
    text = _clean_text(res.get("text") if isinstance(res, dict) else None)
    lang = res.get("language") if isinstance(res, dict) else language
    conf = res.get("confidence") if isinstance(res, dict) else None

    return {"text": text, "language": lang, "confidence": conf, "raw": res, "error": None}
