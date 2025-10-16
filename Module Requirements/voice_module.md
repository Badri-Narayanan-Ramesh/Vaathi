wrap it behind a small FastAPI /tts route so your quiz or collaboration clients can request spoken audio dynamically.

# QUIZ REQUIREMENTS

- push-to-talk STT (local Whisper or OpenAI Whisper API)
- “Speak it” TTS (local pyttsx3 or Coqui / ElevenLabs)
- a tiny Streamlit widget that fills quiz answers by voice and reads stems/rationales
- hooks for Study-Jam (broadcast transcripts & “speaking” events)

# Latency & quality tips (works well on laptops)

- 16 kHz mono end-to-end (keep resampling once).
- VAD endpointing at ~0.5s silence to trim tail.
- Local Whisper base is a sweet spot for speed/accuracy; use tiny on low CPUs.
- If you need near-realtime, chunk 1–2s buffers and run incremental STT (stretch goal).

# Fallbacks 

- Mic not available → show a text box and a note: “Press 🔊 to hear; type to answer.”
- TTS engine fails → failover order: pyttsx3 → Coqui → ElevenLabs.

--- 

# TEXT TO SPEECH

Naturalness & expressiveness: how human-like, smooth prosody, emotional tone, control over emphasis, pauses, etc.

Zero-shot / voice cloning: ability to adapt to a new speaker with minimal data.

Inference speed / latency: how fast it runs (especially for streaming or near-real-time use).

Memory & compute footprint: VRAM / GPU / CPU needs.

Language / accent support: how many languages, dialects.

Control / conditioning: ability to influence style, prosody, pitch, rhythm.

Ease of integration, tooling, community support.

Dia (1.6B)
XTTS-v2
Mozilla TTS
Coqui TTS
Piper (by Rhasspy)
VoxCPM
IndexTTS
F5R-TTS
Muyan-TTS

Start with inference-only, single-speaker, FP16 / mixed precision

Most of these models will run inference in FP16 / half precision to reduce VRAM. Use batch size = 1 or small.
Focus on models with efficient vocoders

The vocoder (mel → waveform) is often the memory/time bottleneck, especially with high fidelity (HiFi-GAN, WaveGlow, etc.). Some newer systems bundle efficient vocoders (lightweight or neural decoders optimized).

3. Profile memory and latency early

Run small text inputs (a few sentences) and measure:

Peak VRAM usage

Time to synthesize

Warm-up overhead

This gives you a baseline before scaling to longer texts or more voices.

4. Use streaming or chunked synthesis if needed

If full sequence generation is heavy, chunk your text, synthesize in parts, and stitch audio (with overlap) — many libs support that.

5. Consider pruning / quantization if needed

If a model is just slightly over memory, techniques like model quantization or pruning layers might make it feasible.

You can also measure audio quality (MOS, similarity, prosody) by listening or using metrics (if you have reference audio).

Use FP16, batch size = 1, chunked streaming if needed.

--- 
# SPEECH TO TEXT (Open Source, Locally)
- Whisper / Faster-Whisper Supports streaming, batching, multilingual, timestamps.
- FunASR (Paraformer-based) For low-latency ASR with endpoint detection, punctuation, diarization. Non-autoregressive = ~3× faster decoding.
- Whisper.cpp (for C++ or offline edge). Quantize models to int8/int4 for CPU deployment.
- wav2vec 2.0 (fine-tuneable) Good for custom domains (technical jargon, accents).

Best overall accuracy (English + multilingual)	faster-whisper + large-v3-turbo, float16
Fast inference / low latency	FunASR Paraformer or Whisper medium float16
Edge / offline no Python	whisper.cpp quantized
Custom vocabulary / domain	wav2vec 2.0 fine-tuned with LM rescore

Whisper small / medium, Whisper.cpp int8/float16, faster-whisper base
Whisper large-v3 or large-v3-turbo
FunASR Paraformer, OWSM v3.1
wav2vec 2.0 (base/large) + custom head OK for inference; fine-tuning needs gradient checkpointing, Great if you have domain data

