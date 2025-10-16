# voice_io.py
import io, os, tempfile, numpy as np
from dataclasses import dataclass
from typing import Optional
import soundfile as sf

# ---------- STT ----------
@dataclass
class STTConfig:
    engine: str = os.getenv("STT_ENGINE", "whisper_local")  # whisper_local|whisper_api
    model: str = "base"  # tiny/base/small (local)
    language: Optional[str] = None

class STT:
    def __init__(self, cfg: STTConfig = STTConfig()):
        self.cfg = cfg
        if self.cfg.engine == "whisper_local":
            import whisper
            self._model = whisper.load_model(self.cfg.model)
        else:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe(self, wav_bytes: bytes) -> str:
        # ensure 16k mono float32
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        if self.cfg.engine == "whisper_local":
            import tempfile, soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, sr)
                out = self._model.transcribe(f.name, language=self.cfg.language)
            return out.get("text","").strip()
        else:
            # OpenAI Whisper API
            from openai import OpenAI
            client = self._client
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, sr)
                with open(f.name, "rb") as file:
                    resp = client.audio.transcriptions.create(
                        model="whisper-1", file=file, language=self.cfg.language
                    )
            return resp.text.strip()

# ---------- TTS ----------
@dataclass
class TTSConfig:
    engine: str = os.getenv("TTS_ENGINE", "pyttsx3")  # pyttsx3|coqui|elevenlabs
    voice: Optional[str] = None  # engine-specific id/name
    rate: Optional[int] = 180

class TTS:
    def __init__(self, cfg: TTSConfig = TTSConfig()):
        self.cfg = cfg
        if cfg.engine == "pyttsx3":
            import pyttsx3
            self.engine = pyttsx3.init()
            if cfg.rate: self.engine.setProperty('rate', cfg.rate)
        elif cfg.engine == "coqui":
            from TTS.api import TTS as COQUI
            # small, fast multi-speaker English:
            self.coqui = COQUI("tts_models/en/vctk/vits")
        elif cfg.engine == "elevenlabs":
            import elevenlabs
            elevenlabs.set_api_key(os.getenv("ELEVENLABS_API_KEY"))
            self.eleven = elevenlabs

    def synth(self, text: str) -> bytes:
        if self.cfg.engine == "pyttsx3":
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            self.engine.save_to_file(text, tmp.name)
            self.engine.runAndWait()
            data, _ = sf.read(tmp.name, dtype='float32')
            os.remove(tmp.name)
            buf = io.BytesIO()
            sf.write(buf, data, 22050, format='WAV')
            return buf.getvalue()
        elif self.cfg.engine == "coqui":
            wav = self.coqui.tts(text)
            buf = io.BytesIO()
            sf.write(buf, np.array(wav), 22050, format='WAV')
            return buf.getvalue()
        else:  # elevenlabs
            audio = self.eleven.generate(text=text, voice=self.cfg.voice or "Rachel")
            return b"".join(audio)
# Example usage: