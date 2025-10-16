# WER (Word Error Rate)
Formula: WER = edit_distance(ref_words, hyp_words) / len(ref_words)

Tokenization: alphanumerics + apostrophes (lowercased).

Pass criteria: WER <= wer_max (per item threshold in your JSON).

# Speaking Rate (WPM: words per minute)
Formula: WPM = (#words) / (audio_duration_seconds) * 60

By default uses ASR word count; if ASR is empty, falls back to the prompt’s words (so you still get a rough rate).

Pass criteria: within the item’s wpm range, e.g., "150-190".

# Pause Validation (Prosody pauses)

Silence detection is energy-based: segments below a small amplitude threshold for ≥ minimum duration are “silences”.

If your ASR returns word timestamps, the harness can check pause after specific tokens (e.g., “after=Wait;min=250;max=600”).

If timestamps are unavailable, it falls back to checking whether any detected silences fall within each rule’s duration range.

Pass criteria: all pause rules in the item are satisfied.

# SNR (Signal-to-Noise Ratio, dB) (optional heuristic)

Estimates noise floor from the 10th percentile of frame RMS; signal level from the median RMS.

SNR_dB = 20 * log10(signal_rms / noise_rms)

Pass criteria: SNR_dB >= snr_min (if provided).

# Pitch variability (optional check in your JSON)

The harness is ready to read a pitch_stdev_min_semitones key, but you’ll need to add a pitch extractor if you want this measured objectively. (You can also keep it as a manual listening note.)

Metrics (objective + quick human rubric)
A) Intelligibility (objective)

Round-trip WER: Transcribe TTS audio with Whisper (local) and compute Word Error Rate vs. input.

Goal: WER ≤ 10% for clean sentences; ≤ 15% for long paragraph.

B) Audio quality (objective)

Clipping: Peak amplitude ≤ -0.1 dBFS (no samples at 0 dBFS).

Loudness (integrated LUFS): target -16 LUFS ± 3 dB (podcast/voice norm).

Noise/SNR: estimate leading/trailing silence RMS vs speech RMS; SNR ≥ 20 dB.

Sample format: mono, ≥16 kHz sample rate; consistent across runs.

C) Latency & throughput (objective)

Latency to first byte (synthesis time): < 1.5 s for 100–150 chars on your laptop.

Real-time factor (RTF) = synth_time / audio_duration: ≤ 1.0 (ideally < 0.7).

D) Stability (objective-ish)

Determinism: same input, same voice/rate → identical or near-identical waveform (pyttsx3 is usually stable).

Resource use: CPU avg < 120% of one core, RAM delta < 200 MB per synth.

E) Naturalness & prosody (human rubric)

Rate 1–5 on:

Pronunciation clarity

Pausing & punctuation respect

Emphasis/intonation (not monotone)

Overall pleasantness (fatigue)

Goal: Mean Opinion Score (MOS) ≥ 3.5/5, no item < 3.

3) Acceptance Gates (simple pass/fail)

A test passes if ALL are true:

WER ≤ 10% (≤15% long paragraph)

No clipping; loudness in -19 to -13 LUFS

SNR ≥ 20 dB

RTF ≤ 1.0; latency ≤ 1.5 s

MOS ≥ 3.5 and no single rubric item < 3

If any fail: tweak rate, punctuation in text (add commas), or switch engine (e.g., Coqui) for quality.

Prepare a list of 10 test sentences (above).

For each:

Call run_tts_eval(text) → check peak_dbfs ≤ -0.1, SNR ≥ 20, RTF ≤ 1.0, sr ≥ 16000.

(If using Whisper) transcribe and compute wer(text, transcript).

Listen briefly and score MOS (1–5) on the 4 rubric items.

If any metric fails, adjust:

Add commas/periods for better pausing,

Lower rate (e.g., 160) for clarity,

If quality is still weak, switch to Coqui for nicer timbre.
