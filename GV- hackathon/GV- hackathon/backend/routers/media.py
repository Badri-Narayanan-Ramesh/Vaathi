from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from backend.services.ai_adapter import text_to_speech, speech_to_text
from backend.utils.files import save_upload
import os
import uuid
from pathlib import Path

router = APIRouter()

AUDIO_DIR = Path("./audio")
AUDIO_DIR.mkdir(exist_ok=True, parents=True)

def temp_audio_path():
    return str(AUDIO_DIR / f"{uuid.uuid4()}.mp3")

@router.post("/tts")
async def tts(text: str):
    out_path = temp_audio_path()
    text_to_speech(text, out_path)
    return FileResponse(out_path, media_type="audio/mpeg", filename=os.path.basename(out_path))

@router.post("/stt")
async def stt(audio: UploadFile = File(...)):
    path = save_upload(audio)
    text = speech_to_text(path)
    return {"text": text}
