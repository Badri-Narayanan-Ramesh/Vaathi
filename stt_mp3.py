# stt_mp3.py
import json, os, subprocess, tempfile, time, soundfile as sf, torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = "openai/whisper-small"         # open-source MIT
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float32                  # safe, avoids dtype mismatches

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)


def convert_to_wav_16k_mono(path: str) -> str:
    """Use ffmpeg to convert any input (mp3, mp4, etc.) to 16 kHz mono WAV."""
    tmp_wav = os.path.join(
        tempfile.gettempdir(), os.path.splitext(os.path.basename(path))[0] + "_16k.wav"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", path, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", tmp_wav
    ]
    subprocess.run(cmd, check=True)
    return tmp_wav


def transcribe(audio_path: str, language="en", task="transcribe"):
    # 1️⃣ Convert MP3 (or anything) → clean 16 kHz WAV
    wav_path = convert_to_wav_16k_mono(audio_path)

    # 2️⃣ Load audio
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000, f"expected 16 kHz, got {sr}"

    # 3️⃣ Preprocess and move to device
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    inputs["input_features"] = inputs["input_features"].to(model.dtype)

    # 4️⃣ Generate
    t0 = time.time()
    prompt_ids = processor.get_decoder_prompt_ids(language=language, task=task) or []
    max_new = model.config.max_target_positions - len(prompt_ids) - 2  # stay within 448
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            language=language,
            task=task,
            max_new_tokens=max_new,
            num_beams=1,
            do_sample=False,
        )

    text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return {
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(model.dtype),
        "language": language,
        "task": task,
        "duration_sec": round(len(audio) / sr, 3),
        "processing_sec": round(time.time() - t0, 3),
        "text": text,
        "source": audio_path,
    }


if __name__ == "__main__":
    result = transcribe("data/test_audio.mp3")   # <- any mp3 file
    print(json.dumps(result, ensure_ascii=False, indent=2))
