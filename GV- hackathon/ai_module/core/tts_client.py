"""Wrapper for TTS that calls into text_2_speech_fast_v2._tts_to_wav_bytes and returns structured result.

Returns: {wav_bytes, sample_rate, length_seconds, error, meta}
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import io



def synthesize_text(text: str, rate: int = 180, voice: Optional[str] = None, target_sr: int = 16000) -> Dict[str, Any]:
    try:
        safe_text = (text or "").strip()
        if not safe_text:
            return {"wav_bytes": b"", "sample_rate": None, "length_seconds": 0.0, "error": "empty input", "meta": {}}
        # Safety trim to avoid extremely long synthesis
        if len(safe_text) > 5000:
            safe_text = safe_text[:5000]

        try:
            from text_2_speech_fast_v2 import _tts_to_wav_bytes  # lazy import
        except Exception as e:
            return {"wav_bytes": b"", "sample_rate": None, "length_seconds": 0.0, "error": f"tts import error: {e}", "meta": {}}

        wav_bytes = _tts_to_wav_bytes(safe_text, rate=rate, voice=voice, target_sr=target_sr)
        # try to read basic info via lazy import
        try:
            import whisper_hf_asr  # lazy

            audio, sr = whisper_hf_asr._read_audio_any(wav_bytes)
            length = float(len(audio)) / float(sr) if sr and len(audio) else 0.0
        except Exception:
            sr = target_sr
            length = 0.0

        return {"wav_bytes": wav_bytes, "sample_rate": sr, "length_seconds": length, "error": None, "meta": {"rate": rate, "voice": voice}}
    except Exception as e:
        return {"wav_bytes": b"", "sample_rate": None, "length_seconds": 0.0, "error": str(e), "meta": {}}
