"""High-level orchestrator linking STT -> Tutor chain -> TTS.

Functions return a dict with keys: transcript, answer_text, wav_bytes, ok, reasons, metadata
and never raise uncaught exceptions.
"""
from __future__ import annotations

from typing import Optional, Dict, Any

from .stt_client import transcribe_wav_bytes
from .tutor_chain import answer_once
from .tts_client import synthesize_text


def handle_audio_turn(wav_bytes: bytes, language: Optional[str] = None, contexts: Optional[str] = None) -> Dict[str, Any]:
    try:
        stt = transcribe_wav_bytes(wav_bytes, language=language)
        transcript = stt.get("text", "")
        metadata: Dict[str, Any] = {"stt": {k: stt.get(k) for k in ("language", "confidence", "error")}}

        if not transcript:
            return {"ok": False, "reasons": ["no_transcript"], "transcript": transcript, "answer_text": "", "wav_bytes": b"", "metadata": metadata}

        answer = answer_once(transcript, contexts=contexts)
        answer_text = answer.get("text", "") if isinstance(answer, dict) else ""

        tts = synthesize_text(answer_text)

        metadata["llm"] = answer.get("meta") if isinstance(answer, dict) else {}
        metadata["tts"] = {k: tts.get(k) for k in ("sample_rate", "length_seconds", "error", "meta")}

        ok = True if answer.get("ok") and (tts.get("error") is None) else False

        return {"ok": ok, "reasons": [], "transcript": transcript, "answer_text": answer_text, "wav_bytes": tts.get("wav_bytes", b""), "metadata": metadata}
    except Exception as e:
        return {"ok": False, "reasons": ["exception"], "transcript": "", "answer_text": "", "wav_bytes": b"", "metadata": {"error": str(e)}}


def handle_text_turn(text: str, contexts: Optional[str] = None) -> Dict[str, Any]:
    try:
        answer = answer_once(text, contexts=contexts)
        answer_text = answer.get("text", "") if isinstance(answer, dict) else ""
        tts = synthesize_text(answer_text)
        metadata = {"llm": answer.get("meta"), "tts": {k: tts.get(k) for k in ("sample_rate", "length_seconds", "error", "meta")}}
        ok = True if answer.get("ok") and (tts.get("error") is None) else False
        return {"ok": ok, "reasons": [], "transcript": text, "answer_text": answer_text, "wav_bytes": tts.get("wav_bytes", b""), "metadata": metadata}
    except Exception as e:
        return {"ok": False, "reasons": ["exception"], "transcript": text, "answer_text": "", "wav_bytes": b"", "metadata": {"error": str(e)}}
