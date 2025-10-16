Cautions / quick fixes for your code

These are small but important to make it “rock-solid”:

Private driver hook (fragile):
Setting eng._driver._audio_stream = _on_audio relies on pyttsx3 internals; it will break on some OS/driver combos.
✅ Keep your current temp-file fallback, and consider defaulting to the safe path on non-Windows or when the hook isn’t present.

Assumed PCM format & sample rate:
You assume 16-bit mono @ ~22.05 kHz from pyttsx3. That’s often true but not guaranteed. Better: detect format (or just read the produced WAV in the fallback path consistently).

torchaudio Resample on GPU:
torchaudio.transforms.Resample is CPU-bound; calling .cuda() on the transform/tensor can raise errors.
✅ Keep tensors on CPU for resampling; only move to CUDA if you use a GPU-capable resampler.

Whisper FP16 on CPU:
fp16=True does not speed up CPU and can fail; FP16 is a CUDA thing.
✅ Use fp16=use_cuda only when device=="cuda".

Whisper lazy-load flag:
transcribe_wav_bytes() returns None unless _have_whisper is set by a prior warm-up.
✅ If Whisper is requested, call _load_whisper_base() inside transcribe_wav_bytes() instead of gating on _have_whisper.

faster-whisper defaults:
compute_type="int8" is fast but can reduce accuracy; provide a knob (int8_float16/float16) and auto-pick based on device.

Edge-case metrics:
Guard against empty audio (avoid log(0)), and consider ITU-R BS.1770 if you later need real LUFS (your RMS proxy is fine for now).


Great! With your NVIDIA RTX 4070 (8GB VRAM) and CUDA 12.6 installed, you're well-equipped to leverage GPU acceleration in this script, particularly for the resampling (via torchaudio) and Whisper transcription parts. The script already has optional CUDA support baked in, but it needs a few tweaks to fully utilize your setup:

Torch + torchaudio with CUDA: These need to be installed with CUDA 12.6 support (PyTorch doesn't auto-detect pre-installed system CUDA; it bundles its own runtime). This enables FP16 on GPU for faster ops.
faster-whisper: This is the preferred path in the script (via CTranslate2) and supports CUDA directly for much faster transcription than OpenAI's Whisper. It's CPU-fallback otherwise, but we can make it scream on your GPU.
Potential issues:

The in-memory TTS capture hack assumes pyttsx3's default driver behavior (works on Windows SAPI, may need testing on Linux/macOS with espeak/NSS). If it falls back to temp files, it's still fine but slower.
Whisper/faster-whisper models like "base" fit easily in 8GB VRAM.
No major conflicts with CUDA 12.6—PyTorch supports it via nightly or specific builds, but stable wheels work with bundled runtime.



Installation Steps
Run these in your Python environment (e.g., via venv or conda). Assume Python 3.10+.

Install PyTorch with CUDA 12.6 support:
textpip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

This gets torch 2.4+ with CUDA 12.6 kernels. Verify after install:
textpython -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
Should output True for CUDA.


Install faster-whisper (GPU-accelerated):
textpip install faster-whisper

It auto-detects CUDA if torch is GPU-enabled. For your 4070 (Ampere arch), it uses optimized beams. Test: It will offload to GPU in the script.


Other deps (if not already installed):
textpip install pyttsx3 soundfile numpy whisper  # whisper is fallback, optional if faster-whisper works

pyttsx3 on Linux may need espeak via apt; on Windows it's SAPI.


Optional: For even faster CUDA in faster-whisper:

Install cuDNN if not already (PyTorch bundles basics, but for max speed: download from NVIDIA and add to PATH).



Script Tweaks for Optimal GPU Use
Your script is mostly ready—--cuda flag enables GPU for OpenAI Whisper fallback. To prioritize faster-whisper on GPU:

In _load_faster_whisper, hardcode GPU:
pythondef _load_faster_whisper(model_size: str = "base", device: str = "cuda", compute_type: str = "float16"):
    # ... rest unchanged

Use "float16" for FP16 on GPU (faster, fits your VRAM). "int8" for quantized if VRAM tight, but 8GB handles base in FP16 fine.
In fast_transcribe, add model_size="base.en" for English-only (faster/smaller).


Warm-up resampling on GPU: Add this before main loop in main():
python_init_resampler()  # Forces torch load early
if torch.cuda.is_available():
    print("CUDA ready for resampling & ASR")

CLI tip: Use --cuda (enables GPU for fallback), but faster-whisper will auto-GPU.

Running the Script
Save your script as text_2_speech_fast.py and run:
textpython text_2_speech_fast.py --cuda --wer-first-n 6 --max 6 --rate 200

--cuda: GPU for Whisper fallback (and enables FP16).
Expect RTF <0.5 on synthesis, transcription in ~0.1-0.5s per sentence on GPU (vs 2-5s on CPU).
For a bigger model: Change model_size="small" or "medium" in fast_transcribe (still fits in 8GB, better accuracy but slower).
If synthesizing only: --synthesize-only --synthesize-single --out all_samples.wav for a concat file.

Expected Performance

TTS synthesis: CPU-bound (pyttsx3), ~0.1-0.5s latency per sentence on modern CPU. Your GPU won't help here.
Resample: Torchaudio on GPU = near-instant for short audio.
Transcription (WER): faster-whisper on 4070 = real-time or faster (e.g., 5-10x speedup vs CPU). WER should be low (~0.1-0.3 for these clear sentences).
Monitor GPU: Use nvidia-smi in another terminal—should show torch processes spiking to ~1-2GB VRAM.
Bottlenecks: If TTS capture falls back (exception in hack), synthesis slows 2-3x. Test on your OS.

Troubleshooting


CUDA not found: If torch.cuda.is_available() False, reinstall torch with the cu126 index. Mismatch with system CUDA is fine (PyTorch uses bundled).


pyttsx3 audio hook fails: Comment out the try block in _tts_to_wav_bytes and force fallback (uses temp files). Or use Coqui TTS for fully GPU TTS alternative:
If you want end-to-end GPU TTS (more natural than pyttsx3), replace the TTS part with Coqui:
textpip install TTS
Then adapt _tts_to_wav_bytes:
pythonfrom TTS.api import TTS
_tts_model = None
def _get_tts():
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC").cuda()
    return _tts_model

# In func:
tts = _get_tts()
buf = io.BytesIO()
tts.tts_to_file(text, file_path=buf, format="wav")  # Wait, api uses files; hack with temp or in-mem via numpy
# Better: y = tts.synthesize(text); then sf.write...
Coqui is slower to init but GPU-fast inference, better quality.


WER high? pyttsx3 prosody isn't perfect; try rate=150 for clarity.


Out of memory: Stick to "base" model; your 8GB is ample.


This should get you 5-10x overall speedup on ASR. If errors paste traceback, or share OS details for more help!




-----------

\Use Built-in FP16 (No Changes Needed):

Run with --cuda: This enables GPU + FP16 in OpenAI Whisper.
Example command: python text_2_speech_fast.py --wer-first-n 2 --cuda --max 6
This offloads to GPU, uses FP16 automatically for inference speed.


Verify GPU Usage:

Add this debug print in fast_transcribe or _load_faster_whisper:
pythonimport torch
print(f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

Monitor with nvidia-smi during runs.


Other Tips for Your Setup:

Torch CUDA Alignment: Your resampler (torchaudio.transforms.Resample) already uses CUDA if available. Ensure import torch succeeds with CUDA (run python -c "import torch; print(torch.cuda.is_available())"—should be True).
Batch for Efficiency: If testing more sentences, batch transcribe calls to amortize GPU warm-up.
Potential Issues: If OOM ever hits (unlikely), downgrade to "tiny" model via --model_size tiny in fast_transcribe calls (edit the function default).
Accuracy Boost: For mixed-language sentences in TEST_SENTENCES, set language="multilingual" or detect auto via Whisper's info.language.

Great! Since you're opting for GPU acceleration without quantization (using FP16 on CUDA for optimal speed/accuracy balance), your existing script is already well-positioned for this. The OpenAI Whisper backend (via whisper library) will handle FP16 natively on your RTX 4070, and faster-whisper can fallback or be configured similarly. Here's how to set it up and run seamlessly.
Step 1: Verify/Install Dependencies
Ensure everything is CUDA-compatible. Run these in your environment (assuming Python 3.10+ and pip):
bash# Core deps (if not already installed)
pip install pyttsx3 soundfile numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install whisper  # OpenAI Whisper (supports CUDA FP16 out-of-the-box)

# Optional: faster-whisper for even faster transcription (still FP16 on GPU, no quant)
pip install faster-whisper

CUDA Check: After install, run this snippet to confirm:
pythonimport torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should show 'NVIDIA GeForce RTX 4070'
If False, reinstall PyTorch with the CUDA 12.6 index URL above, or check NVIDIA drivers (need >=550 for CUDA 12.6).

Step 2: Minor Code Edits for Reliable GPU Offload
Your script has some small issues with the in-memory TTS capture (the driver hack may not work reliably across platforms, potentially falling back to temp files). Also, ensure Whisper uses GPU explicitly. Apply these patches:

Fix/Improve TTS Capture (Optional but Recommended):

The raw PCM hooking can fail silently. Replace the try block in _tts_to_wav_bytes with a more robust temp-file fallback, or use pyttsx3's event loop properly. For now, force the fallback if needed, but here's a safer version of the try-except:
pythontry:
    # ... (existing hook code)
except Exception as e:
    print(f"TTS hook failed ({e}), falling back to temp file...")
    # Existing fallback code...

This ensures it works on your setup (Windows/Linux/macOS vary).


Force GPU in Whisper Loads:

Update _load_whisper_base to always use CUDA if available (your CLI flag already does, but hardcode for testing):
pythondef _load_whisper_base(use_cuda: bool = True):  # Default to True
    global _asr_model, _have_whisper
    if _asr_model is None:
        import whisper
        _have_whisper = True
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        _asr_model = whisper.load_model("base", device=device)
        print(f"Whisper loaded on {device}")  # Debug print
    return _asr_model

In transcribe_wav_bytes, ensure fp16=use_cuda is passed (already there).


For faster-whisper (GPU FP16, No Quant):

Update _load_faster_whisper to use CUDA and FP16 compute type (non-quantized):
pythondef _load_faster_whisper(model_size: str = "base", device: str = "cuda" if torch.cuda.is_available() else "cpu", compute_type: str = "float16"):
    if model_size in _fw_models:
        return _fw_models[model_size]
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _fw_models[model_size] = model
        print(f"faster-whisper loaded on {device} with {compute_type}")  # Debug
        return model
    except Exception as e:
        print(f"faster-whisper load failed: {e}")
        return None

This uses float16 (full precision equivalent on GPU) for max accuracy/speed without int8 quantization.


Resampler GPU Ensure:

In _init_resampler, add device placement:
pythonif torch.cuda.is_available():
    _resampler = _resampler.cuda()

Already in your code—good!



Save the script as text_2_speech_fast.py after edits.
Step 3: Run the Script on GPU
Use the --cuda flag to enable GPU for Whisper (and auto FP16). It will warm up on GPU during the first WER compute.

Basic Run (Metrics + WER on GPU):
bashpython text_2_speech_fast.py --wer-first-n 2 --cuda --max 6 --rate 180

This synthesizes TTS, computes metrics, transcribes first 2 sentences on GPU, and calculates WER.
Expected output: Table with latency/RTF (TTS on CPU, but fast), WER via GPU Whisper.
Monitor GPU: Open another terminal and run watch -n 1 nvidia-smi—you'll see VRAM spike to ~1-2 GB during transcription.


Synthesize Only (Save WAVs, No Metrics):
bashpython text_2_speech_fast.py --synthesize-only --cuda --out-dir tts_outputs --max 6

Generates WAVs quickly; Whisper warmup skips if no WER.


Concatenated Output:
bashpython text_2_speech_fast.py --synthesize-single --synthesize-only --out all_samples.wav --max 6


Expected Performance on Your Hardware

TTS Synthesis: CPU-bound (pyttsx3), ~0.1-0.5s per sentence (RTF <0.5).
Transcription (GPU): <0.5s per short clip in FP16 on RTX 4070—2-5x faster than CPU.
Total Run: <10s for all tests, with GPU utilization only during WER.
VRAM Usage: Peaks at ~1.5 GB (Whisper base in FP16 + torchaudio). Plenty left for multitasking.

Troubleshooting Tips

If Whisper Stays on CPU: Check torch.cuda.is_available(). If issues, set export CUDA_VISIBLE_DEVICES=0 before running.
Errors on Import: If faster-whisper fails (CUDA mismatch), stick to OpenAI Whisper—it works fine.
Slower than Expected: Ensure no other GPU apps running. For even faster, upgrade model to "small" (edit model_size="small" in fast_transcribe—still fits in 8GB).
TTS Voice Issues: pyttsx3 uses system voices; test rates/voices via CLI if needed.
Logs: Add print for times, e.g., in transcribe_wav_bytes:
pythont0 = time.perf_counter()
# ... transcription
print(f"Transcription time: {time.perf_counter() - t0:.2f}s on GPU")


This will give you full GPU-accelerated transcription workflow without quantization trade-offs. If you hit any errors (paste them), or want benchmarks, let me know!

Enhance Metrics/Eval:

Add MOS proxy (e.g., via DNSMOS API if online allowed).
GPU Accelerate: --cuda if NVIDIA GPU (faster Whisper/resample).
Custom Text: Edit TEST_SENTENCES or add CLI --text "Your sentence".


Test Edge Cases:

Long text: --max 6 (sentence 6 tests punctuation).
Different rates/voices: --rate 150 --voice "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0".
Error Handling: Mute speakers? Script still works (saves WAV without playback).