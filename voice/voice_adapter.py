# voice_adapter.py
"""
Unified Voice Adapter: STT (Whisper local/API) + TTS (pyttsx3 / Coqui)

Features
- STT:
  - whisper_local: runs OpenAI Whisper locally (tiny/base/small/...).
  - whisper_api  : uses OpenAI Whisper API (requires OPENAI_API_KEY).
  - Auto-resamples incoming audio bytes to 16 kHz mono (librosa if available).
- TTS:
  - pyttsx3      : fully offline, OS voices; returns WAV bytes.
  - coqui        : local neural TTS via Coqui TTS; model configurable.
- Utilities:
  - ensure_mono_16k()     : normalize WAV bytes to mono 16 kHz.
  - list_tts_voices()     : list available voices for current engine.
  - set_voice() / set_rate() helpers.
  - synth_to_file()       : save audio to disk easily.
  - Safe fallbacks & clear exceptions with actionable messages.

Env Vars (optional)
- STT_ENGINE, STT_MODEL, OPENAI_API_KEY
- TTS_ENGINE, TTS_VOICE, TTS_RATE, COQUI_MODEL
"""

from __future__ import annotations
import io
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import soundfile as sf

__all__ = [
    "STTConfig",
    "STT",
    "TTSConfig",
    "TTS",
    "ensure_mono_16k",
    "list_tts_voices",  # Exposed for external use if needed
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _have_librosa() -> bool:
    try:
        import librosa  # noqa: F401
        return True
    except Exception:
        return False


def _resample_mono(x: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Resample 1D mono to target_sr using librosa if available; else passthrough."""
    if sr == target_sr:
        return x, sr
    if _have_librosa():
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        return x.astype(np.float32, copy=False), target_sr
    # Fallback: return original to avoid hard dependency
    return x, sr


def ensure_mono_16k(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Read arbitrary WAV bytes and return mono float32 at (≈)16 kHz.

    If librosa is available the audio will be resampled; otherwise the
    original sample rate is returned (caller should handle mismatches).
    """
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y, sr = _resample_mono(y, sr, 16000)
    return y, sr


# -----------------------------------------------------------------------------
# STT
# -----------------------------------------------------------------------------


@dataclass
class STTConfig:
    engine: str = os.getenv("STT_ENGINE", "whisper_local")
    model: str = os.getenv("STT_MODEL", "base")
    language: Optional[str] = None


class STT:
    """Speech-to-Text adapter.

    Supported engines: whisper_local (requires openai-whisper) and
    whisper_api (requires openai SDK and OPENAI_API_KEY). The adapter
    normalizes incoming audio to mono/16k when possible.
    """

    def __init__(self, cfg: STTConfig = STTConfig()):
        self.cfg = cfg
        self._mode = self.cfg.engine.strip().lower()
        if self._mode not in {"whisper_local", "whisper_api"}:
            raise ValueError(f"Unsupported STT engine: {self._mode}")

        self._whisper = None
        self._client = None

        if self._mode == "whisper_api":
            try:
                from openai import OpenAI  # type: ignore

                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is required for whisper_api STT.")
                self._client = OpenAI(api_key=api_key)
            except Exception as e:
                raise RuntimeError("whisper_api selected but openai SDK failed to initialize") from e

    def _ensure_whisper_local(self):
        if self._whisper is None:
            try:
                import whisper  # type: ignore

                self._whisper = whisper.load_model(self.cfg.model)
            except Exception as e:
                raise RuntimeError("Local whisper model load failed or 'openai-whisper' is not installed") from e

    def transcribe(self, wav_bytes: bytes) -> str:
        """Return transcription for given WAV bytes (string)."""
        audio, sr = ensure_mono_16k(wav_bytes)
        if self._mode == "whisper_local":
            self._ensure_whisper_local()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            try:
                sf.write(tmp_path, audio, sr)
                out = self._whisper.transcribe(tmp_path, language=self.cfg.language)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            return (out.get("text") or "").strip()

        # whisper_api
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio, sr)
            with open(tmp_path, "rb") as file:
                resp = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    language=self.cfg.language,
                )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return (getattr(resp, "text", "") or "").strip()


# -----------------------------------------------------------------------------
# TTS
# -----------------------------------------------------------------------------


@dataclass
class TTSConfig:
    engine: str = os.getenv("TTS_ENGINE", "pyttsx3")
    voice: Optional[str] = os.getenv("TTS_VOICE", None)
    rate: Optional[int] = os.getenv("TTS_RATE", "180")
    coqui_model: str = os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits")
    target_sr: int = 16000

    def __post_init__(self):
        if self.rate is not None:
            self.rate = int(self.rate)


class TTS:
    """Text-to-Speech adapter with pyttsx3 and optional Coqui support."""

    def __init__(self, cfg: TTSConfig = TTSConfig()):
        self.cfg = cfg
        self._mode = self.cfg.engine.strip().lower()
        if self._mode not in {"pyttsx3", "coqui"}:
            raise ValueError(f"Unsupported TTS engine: {self._mode}")

        self.engine = None
        self.coqui = None

        if self._mode == "pyttsx3":
            try:
                import pyttsx3  # type: ignore

                self.engine = pyttsx3.init()
                if self.cfg.rate is not None:
                    self.engine.setProperty("rate", self.cfg.rate)
                if self.cfg.voice:
                    self._set_pyttsx3_voice(self.cfg.voice)
            except Exception as e:
                raise RuntimeError("pyttsx3 not available or failed to initialize") from e

        elif self._mode == "coqui":
            try:
                from TTS.api import TTS as COQUI  # type: ignore

                self.coqui = COQUI(model_name=self.cfg.coqui_model)  # Use model_name for newer TTS versions
            except Exception as e:
                raise RuntimeError("Coqui TTS not available or failed to initialize") from e

    # ------------------- pyttsx3 helpers -------------------

    def _set_pyttsx3_voice(self, query: str) -> None:
        """Try to select a pyttsx3 voice by id or (contains) name."""
        if not self.engine:
            return
        try:
            voices = self.engine.getProperty("voices") or []
            for v in voices:
                if getattr(v, "id", "") == query:
                    self.engine.setProperty("voice", v.id)
                    return
            qlow = query.lower()
            for v in voices:
                name = getattr(v, "name", "") or ""
                if qlow in name.lower():
                    self.engine.setProperty("voice", v.id)
                    return
            if voices:
                self.engine.setProperty("voice", voices[0].id)
        except Exception:
            pass

    def list_tts_voices(self) -> List[Dict[str, Any]]:
        """List available voices for the current engine."""
        out: List[Dict[str, Any]] = []
        if self._mode == "pyttsx3" and self.engine:
            try:
                for v in self.engine.getProperty("voices") or []:
                    out.append({
                        "id": getattr(v, "id", None),
                        "name": getattr(v, "name", None),
                        "languages": getattr(v, "languages", None),
                        "gender": getattr(v, "gender", None),
                        "age": getattr(v, "age", None),
                    })
            except Exception:
                pass
        elif self._mode == "coqui" and self.coqui:
            # For Coqui, list speakers if multi-speaker model
            speakers = getattr(self.coqui, "speakers", None)
            if speakers:
                for s in speakers:
                    out.append({"speaker": s})
            else:
                out.append({"model": self.cfg.coqui_model})
        return out

    def set_voice(self, voice_id_or_name: str) -> None:
        """Change voice at runtime (engine-specific implementation)."""
        if self._mode == "pyttsx3":
            self._set_pyttsx3_voice(voice_id_or_name)
        elif self._mode == "coqui":
            self.cfg.voice = voice_id_or_name  # Store for use in synth if applicable

    def set_rate(self, rate: int) -> None:
        self.cfg.rate = rate
        if self._mode == "pyttsx3" and self.engine:
            try:
                self.engine.setProperty("rate", int(rate))
            except Exception:
                pass
        # Coqui does not support rate change via this API easily; ignore or extend if needed

    # ------------------- TTS core -------------------

    def synth(self, text: str) -> bytes:
        """Synthesize text to WAV bytes. Output is mono ~16 kHz when possible."""
        if self._mode == "pyttsx3":
            return self._synth_pyttsx3(text)
        elif self._mode == "coqui":
            return self._synth_coqui(text)
        else:
            raise RuntimeError(f"Unsupported TTS engine at runtime: {self._mode}")

    def synth_to_file(self, text: str, path: str) -> None:
        """Synthesize and save directly to a file (WAV)."""
        audio_bytes = self.synth(text)
        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        sf.write(path, y, sr)

    # ------------------- Engine implementations -------------------

    def _synth_pyttsx3(self, text: str) -> bytes:
        if not self.engine:
            raise RuntimeError("pyttsx3 engine not initialized.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()

            audio, sr = sf.read(tmp_path, dtype="float32")
            if audio.ndim > 1:
                mono = audio.mean(axis=1)
            else:
                mono = audio
            mono, sr = _resample_mono(mono, sr, target_sr=self.cfg.target_sr)

            buf = io.BytesIO()
            sf.write(buf, mono, sr, format="WAV")
            return buf.getvalue()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _synth_coqui(self, text: str) -> bytes:
        if not self.coqui:
            raise RuntimeError("Coqui TTS model not initialized.")
        # Handle speaker if set and model supports it
        kwargs = {}
        if self.cfg.voice and getattr(self.coqui, "speakers", None):
            kwargs["speaker"] = self.cfg.voice
        wav = self.coqui.tts(text=text, **kwargs)
        if isinstance(wav, list):
            wav = np.array(wav)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        output_sr = self.coqui.synthesizer.output_sample_rate
        mono, sr = _resample_mono(wav, output_sr, self.cfg.target_sr)
        buf = io.BytesIO()
        sf.write(buf, mono, sr, format="WAV")
        return buf.getvalue()