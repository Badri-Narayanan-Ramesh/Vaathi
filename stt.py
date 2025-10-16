import json, time, torch, soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

MODEL_ID = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,     # or torch.float16 for GPU speed
).to(DEVICE)


def transcribe(path, language=None, task="transcribe"):
    # 1. Load audio (16 kHz mono recommended)
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr} Hz")

    # 2. Preprocess
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    inputs["input_features"] = inputs["input_features"].to(model.dtype)

    # 3. Generate
    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            language=language,   # e.g. "en"
            task=task,           # "transcribe" or "translate"
            max_new_tokens=444,
            num_beams=1,
            do_sample=False,
        )
    txt = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return {
        "text": txt,
        "duration_sec": round(len(audio) / sr, 3),
        "processing_sec": round(time.time() - t0, 3),
        "dtype": str(model.dtype),
        "device": DEVICE,
    }


if __name__ == "__main__":
    result = transcribe("data/out.wav", language="en")
    print(json.dumps(result, ensure_ascii=False, indent=2))
