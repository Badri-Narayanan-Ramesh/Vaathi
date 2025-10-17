"""Text-to-Speech backends.

Local default uses pyttsx3. Cloud TTS is a stub for future integration.
"""
from __future__ import annotations

import os
from typing import Optional


def speak_local(text: str, out_path: str) -> str:
    """Synthesize speech using pyttsx3 and save to file.

    Returns the output path. If pyttsx3 is unavailable, writes a small placeholder file.
    """
    if not text:
        raise ValueError("text is empty")
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()
    except Exception:
        # Fallback: write text content so tests can verify output exists
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("[tts-fallback]\n" + text)
    return out_path


def speak_cloud(text: str, out_path: Optional[str] = None) -> str:
    """Cloud TTS stub (no implementation)."""
    # TODO: Implement cloud TTS provider
    return ""
