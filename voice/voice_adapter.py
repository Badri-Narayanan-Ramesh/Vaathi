# voice_adapter.py
"""
Unified Voice Adapter: STT (Whisper local/API) + TTS (pyttsx3 / Coqui / ElevenLabs)

Features
- STT:
  - whisper_local: runs OpenAI Whisper locally (tiny/base/small/...).
  - whisper_api  : uses OpenAI Whisper API (requires OPENAI_API_KEY).
  - Auto-resamples incoming audio bytes to 16 kHz mono (librosa if available).
- TTS:
  - pyttsx3      : fully offline, OS voices; returns WAV bytes.
  - coqui        : local neural TTS via Coqui TTS; model configurable.
  - elevenlabs   : cloud TTS; requires ELEVENLABS_API_KEY.
- Utilities:
  - ensure_mono_16k()     : normalize WAV bytes to mono 16 kHz.
  - list_tts_voices()     : list available voices for current engine.
  - set_voice() / set_rate() helpers.
  - synth_to_file()       : save audio to disk easily.
  - Safe fallbacks & clear exceptions with actionable messages.

Env Vars (optional)
- STT_ENGINE=whisper_local|whisper_api     (default: whisper_local)
- STT_MODEL=tiny|base|small|medium|large   (default: base)
- STT_LANGUAGE=en|...                      (default: None->auto)
- TTS_ENGINE=pyttsx3|coqui|elevenlabs      (default: pyttsx3)
- TTS_RATE=integer speaking rate           (default: 180)
- TTS_VOICE=engine-specific voice id/name  (default: None)
- COQUI_MODEL=tts_models/en/vctk/vits      (default: vits above)
- OPENAI_API_KEY=...                       (for whisper_api)
- ELEVENLABS_API_KEY=...                   (for elevenlabs)
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
    "STTConfig", "STT",
    "TTSConfig", "TTS",
    "ensure_mono_16k",
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
    """
    Read arbitrary WAV/AIFF/FLAC bytes -> return mono float32 at (≈)16 kHz.
    Uses librosa when available. If resample not possible, returns original sr.
    """
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    y = y.mean(axis=1)  # to mono
    y, sr = _resample_mono(y, sr, 16000)
    return y, sr

# -----------------------------------------------------------------------------
# STT
# -----------------------------------------------------------------------------

@dataclass
class STTConfig:
    engine: str = os.getenv("STT_ENGINE", "whisper_local")  # whisper_local | whisper_api
    model: str = os.getenv("STT_MODEL", "base")             # tiny/base/small/...
    language: Optional[str] = os.getenv("STT_LANGUAGE", None)

class STT:
    """
    Speech-to-Text adapter.
    - whisper_local: uses local Whisper (pip install openai-whisper)
    - whisper_api  : uses OpenAI Whisper API (OPENAI_API_KEY required)
    """
    def __init__(self, cfg: STTConfig = STTConfig()):
        self.cfg = cfg
        self._mode = self.cfg.engine.strip().lower()
        if self._mode not in {"whisper_local", "whisper_api"}:
            raise ValueError(f"Unsupported STT engine: {self._mode}")

        if self._mode == "whisper_local":
            try:
                import whisper  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "whisper_local selected but 'openai-whisper' is not installed. "
                    "Install with: pip install openai-whisper"
                ) from e
            self._whisper = whisper.load_model(self.cfg.model)
            self._client = None
        else:
            # whisper_api
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "whisper_api selected but 'openai' SDK is not installed. "
                    "Install with: pip install openai"
                ) from e
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for whisper_api STT.")
            self._client = OpenAI(api_key=api_key)
            self._whisper = None

    def transcribe(self, wav_bytes: bytes) -> str:
        """
        Transcribe audio bytes into text. Normalizes to mono ~16 kHz for stability.
        Returns a UTF-8 string (may be empty).
        """
        audio, sr = ensure_mono_16k(wav_bytes)
        if self._mode == "whisper_local":
            # whisper local needs a file path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, sr)
                out = self._whisper.transcribe(f.name, language=self.cfg.language)
            return (out.get("text") or "").strip()
        else:
            # OpenAI Whisper API
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, sr)
                with open(f.name, "rb") as file:
                    resp = self._client.audio.transcriptions.create(
                        model="whisper-1",
                        file=file,
                        language=self.cfg.language
                    )
            # SDK returns .text
            return (getattr(resp, "text", "") or "").strip()

# -----------------------------------------------------------------------------
# TTS
# -----------------------------------------------------------------------------

@dataclass
class TTSConfig:
    engine: str = os.getenv("TTS_ENGINE", "pyttsx3")  # pyttsx3 | coqui | elevenlabs
    voice: Optional[str] = os.getenv("TTS_VOICE", None)
    rate: Optional[int] = int(os.getenv("TTS_RATE", "180"))
    coqui_model: str = os.getenv("COQUI_MODEL", "tts_models/en/vctk/vits")
    target_sr: int = 16000  # normalize outputs for downstream STT

class TTS:
    """
    Text-to-Speech adapter.
    Engines:
      - pyttsx3: offline, OS voices (good baseline).
      - coqui : local neural TTS via Coqui TTS (pip install TTS).
      - elevenlabs: cloud neural TTS (pip install elevenlabs).
    """
    def __init__(self, cfg: TTSConfig = TTSConfig()):
        self.cfg = cfg
        self._mode = self.cfg.engine.strip().lower()
        if self._mode not in {"pyttsx3", "coqui", "elevenlabs"}:
            raise ValueError(f"Unsupported TTS engine: {self._mode}")

        self.engine = None
        self.coqui = None
        self.eleven = None

        if self._mode == "pyttsx3":
            try:
                import pyttsx3  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "TTS engine 'pyttsx3' selected but module not installed. "
                    "Install with: pip install pyttsx3"
                ) from e
            self.engine = pyttsx3.init()
            if self.cfg.rate:
                self.engine.setProperty("rate", int(self.cfg.rate))
            if self.cfg.voice:
                # Try to set by exact id or name (fallback fuzzy)
                self._set_pyttsx3_voice(self.cfg.voice)

        elif self._mode == "coqui":
            try:
                from TTS.api import TTS as COQUI  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "TTS engine 'coqui' selected but Coqui TTS is not installed. "
                    "Install with: pip install TTS"
                ) from e
            self.coqui = COQUI(self.cfg.coqui_model)

        else:  # elevenlabs
            try:
                import elevenlabs  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "TTS engine 'elevenlabs' selected but SDK not installed. "
                    "Install with: pip install elevenlabs"
                ) from e
            api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("ELEVENLABS_API_KEY is required for ElevenLabs TTS.")
            elevenlabs.set_api_key(api_key)
            self.eleven = elevenlabs

    # ------------------- pyttsx3 helpers -------------------

    def _set_pyttsx3_voice(self, query: str) -> None:
        """Try to select a pyttsx3 voice by id or (contains) name."""
        if not self.engine:
            return
        try:
            voices = self.engine.getProperty("voices") or []
            # exact id
            for v in voices:
                if getattr(v, "id", "") == query:
                    self.engine.setProperty("voice", v.id)
                    return
            # name contains
            qlow = query.lower()
            for v in voices:
                name = getattr(v, "name", "") or ""
                if qlow in name.lower():
                    self.engine.setProperty("voice", v.id)
                    return
            # fallback: first voice
            if voices:
                self.engine.setProperty("voice", voices[0].id)
        except Exception:
            # Safe to ignore: keep engine defaults
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
        elif self._mode == "coqui":
            out.append({"model": self.cfg.coqui_model})
        elif self._mode == "elevenlabs":
            try:
                # Ensure elevenlabs client exists
                if self.eleven:
                    # SDK voice list:
                    # new SDKs: elevenlabs.voices() or elevenlabs.get_voices()
                    get_voices = getattr(self.eleven, "voices", None) or getattr(self.eleven, "get_voices", None)
                    if get_voices:
                        voices = get_voices()
                        for v in voices:
                            out.append({"id": getattr(v, "voice_id", None) or getattr(v, "id", None),
                                        "name": getattr(v, "name", None)})
            except Exception:
                pass
        return out

    def set_voice(self, voice_id_or_name: str) -> None:
        """Change voice at runtime (engine-specific implementation)."""
        if self._mode == "pyttsx3":
            self._set_pyttsx3_voice(voice_id_or_name)
        else:
            # For coqui/elevenlabs, voice selection can be part of synth() call or config.
            self.cfg.voice = voice_id_or_name

    def set_rate(self, rate: int) -> None:
        self.cfg.rate = rate
        if self._mode == "pyttsx3" and self.engine:
            try:
                self.engine.setProperty("rate", int(rate))
            except Exception:
                pass

    # ------------------- TTS core -------------------

    def synth(self, text: str) -> bytes:
        """
        Synthesize text to WAV bytes. Output is mono ~16 kHz when possible.
        """
        if self._mode == "pyttsx3":
            return self._synth_pyttsx3(text)
        elif self._mode == "coqui":
            return self._synth_coqui(text)
        else:
            return self._synth_elevenlabs(text)

    def synth_to_file(self, text: str, path: str) -> None:
        """Synthesize and save directly to a file (WAV)."""
        audio = self.synth(text)
        y, sr = sf.read(io.BytesIO(audio), dtype="float32")
        sf.write(path, y, sr)

    # ------------------- Engine implementations -------------------

    def _synth_pyttsx3(self, text: str) -> bytes:
        if not self.engine:
            raise RuntimeError("pyttsx3 engine not initialized.")

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            # (Rate/voice already set in __init__/set_rate/set_voice)
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()

            audio, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            mono, sr = _resample_mono(mono, sr, target_sr=self.cfg.target_sr)

            buf = io.BytesIO()
            sf.write(buf, mono, sr, format="WAV")
            return buf.getvalue()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def _synth_coqui(self, text: str) -> bytes:
        if not self.coqui:
            raise RuntimeError("Coqui TTS model not initialized.")
        # Many coqui models return a numpy array (float32) at 22.05k or 24k
        wav = self.coqui.tts(text)  # can also pass speaker_id/language if model supports
        wav = np.asarray(wav, dtype=np.float32)
        mono, sr = _resample_mono(wav, getattr(self.coqui, "output_sample_rate", 22050), self.cfg.target_sr)
        buf = io.BytesIO()
        sf.write(buf, mono, sr, format="WAV")
        return buf.getvalue()

    def _synth_elevenlabs(self, text: str) -> bytes:
        if not self.eleven:
            raise RuntimeError("ElevenLabs client not initialized.")
        # SDK generate() -> bytes or iterable of bytes (depending on SDK/version).
        voice = self.cfg.voice or "Rachel"
        try:
            audio = self.eleven.generate(text=text, voice=voice)
        except TypeError:
            # Fallback in case newer SDK signature requires model/options:
            audio = self.eleven.generate(text=text, voice=voice)  # adapt here as needed

        # Many SDK versions return bytes directly; some return iterable chunks
        if isinstance(audio, (bytes, bytearray)):
            raw = bytes(audio)
        else:
            # concatenate chunks
            raw = b"".join(audio)

        # Normalize to mono ~16 kHz for consistency
        y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        y, sr = _resample_mono(y, sr, self.cfg.target_sr)
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV")
        return buf.getvalue()
