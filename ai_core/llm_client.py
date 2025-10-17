"""LLM client abstraction supporting cloud (OpenAI gpt-4o-mini) and local (Ollama).

APP_MODE environment variable controls backend: 'cloud' | 'local'.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

import requests


MAX_TOKENS_APPROX = 800  # simple safety to avoid huge outputs in tests


def _truncate_prompt(text: str, max_chars: int = 6000) -> str:
    return text[:max_chars]


def _retry_sleep(attempt: int) -> None:
    time.sleep(min(0.5 * (2 ** attempt), 4.0))


def generate(prompt: str, sys_prompt: Optional[str] = None) -> str:
    """Generate completion using configured backend (Gemini for cloud, Ollama for local)."""
    app_mode = os.getenv("APP_MODE", "cloud").lower()
    if app_mode == "local":
        return _generate_ollama(prompt, sys_prompt)
    # Only Gemini supported for cloud
    out = _generate_gemini(prompt, sys_prompt)
    fallback_enabled = os.getenv("CLOUD_FALLBACK_TO_LOCAL", "1").lower() in {"1", "true", "yes"}
    gemini_failed = out.startswith("[gemini-error]") if isinstance(out, str) else False
    if fallback_enabled and (not out or gemini_failed):
        local_out = _generate_ollama(prompt, sys_prompt)
        return local_out or out
    return out


def _generate_gemini(prompt: str, sys_prompt: Optional[str]) -> str:
    # Support both GOOGLE_API_KEY (AI Studio) and GEMINI_API_KEY env names
    api_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return "[gemini-error] Missing GOOGLE_API_KEY/GEMINI_API_KEY"
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return "[gemini-missing] " + prompt[:80]

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    full_prompt = (sys_prompt + "\n\n" if sys_prompt else "") + _truncate_prompt(prompt)
    out_dir = os.path.abspath(os.path.join(os.getcwd(), "out"))
    os.makedirs(out_dir, exist_ok=True)
    err_log = os.path.join(out_dir, "last_gemini_error.txt")
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(model_name)
            r = model.generate_content(full_prompt)
            txt = getattr(r, "text", None)
            if not txt and hasattr(r, "parts"):
                # older SDK styles
                txt = "".join(getattr(p, "text", "") for p in getattr(r, "parts", [])).strip() or None
            if not txt:
                # Log empty response too
                with open(err_log, "w", encoding="utf-8") as f:
                    f.write("[gemini-empty] No text in response. Raw repr:\n" + repr(r))
            return (txt or "").strip()
        except Exception as e:
            # Write last error for debugging
            try:
                with open(err_log, "w", encoding="utf-8") as f:
                    f.write(f"[gemini-exception attempt={attempt}] {type(e).__name__}: {e}\n")
            except Exception:
                pass
            _retry_sleep(attempt)
    return "[gemini-error] Unable to get response"


    # OpenAI cloud support removed


def _generate_ollama(prompt: str, sys_prompt: Optional[str]) -> str:
    model = os.getenv("OLLAMA_MODEL", "phi3:mini")
    url = "http://localhost:11434/api/generate"
    full_prompt = (sys_prompt + "\n\n" if sys_prompt else "") + prompt
    payload = {"model": model, "prompt": _truncate_prompt(full_prompt), "stream": False}
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return (data.get("response") or "").strip()
        except Exception:
            _retry_sleep(attempt)
    # Test-friendly stub
    return "[ollama-stub] " + prompt[:80]
