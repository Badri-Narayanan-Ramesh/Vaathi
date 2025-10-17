"""Speech-to-Text backends.

Local STT: prefer Whisper (small) if available; fallback to Vosk if installed.
Cloud STT: stub only.
"""
from __future__ import annotations

import os
from typing import Optional


def transcribe_local(wav_path: str) -> str:
    """Transcribe using local models.

    Preference order: whisper (if installed) -> vosk (if installed) -> empty string.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    # Try Whisper first
    try:  # pragma: no cover - environment dependent
        import whisper  # type: ignore

        model = whisper.load_model("small")
        res = model.transcribe(wav_path)
        return (res.get("text") or "").strip()
    except Exception:
        pass

    # Try Vosk next
    try:  # pragma: no cover - environment dependent
        import wave
        import json
        from vosk import Model, KaldiRecognizer  # type: ignore

        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 44100]:
            # For simplicity, we don't resample here
            return ""
        model = Model(lang="en-us")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        text_parts = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                ans = rec.Result()
                text_parts.append(json.loads(ans).get("text", ""))
        final = rec.FinalResult()
        text_parts.append(json.loads(final).get("text", ""))
        return " ".join([t for t in text_parts if t]).strip()
    except Exception:
        pass

    return ""


def transcribe_cloud(wav_path: str) -> str:
    """Cloud STT stub (no implementation)."""
    # TODO: integrate cloud STT provider
    return ""
