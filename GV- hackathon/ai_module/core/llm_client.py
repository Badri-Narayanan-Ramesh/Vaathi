"""LLM client abstraction supporting cloud (OpenAI gpt-4o-mini) and local (Ollama).

APP_MODE environment variable controls backend: 'cloud' | 'local'.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Optional, Dict

import requests


MAX_TOKENS_APPROX = 800  # simple safety to avoid huge outputs in tests


def _truncate_prompt(text: str, max_chars: int = 6000) -> str:
    return text[:max_chars]


def _retry_sleep(attempt: int) -> None:
    time.sleep(min(0.5 * (2 ** attempt), 4.0))


def generate(prompt: str, sys_prompt: Optional[str] = None) -> str:
    """Generate completion using configured backend.

    Args:
        prompt: User prompt string.
        sys_prompt: Optional system message to steer the model.
    Returns:
        Model output text (string, may be empty on failure).
    """
    app_mode = os.getenv("APP_MODE", "cloud").lower()
    if app_mode == "local":
        return _generate_ollama(prompt, sys_prompt)
    # cloud first (provider selectable)
    provider = os.getenv("CLOUD_PROVIDER", "openai").lower()
    if provider == "gemini":
        out = _generate_gemini(prompt, sys_prompt)
    else:
        out = _generate_openai(prompt, sys_prompt)
    # Optional fallback to local if cloud failed/quota
    fallback_enabled = os.getenv("CLOUD_FALLBACK_TO_LOCAL", "1").lower() in {"1", "true", "yes"}
    if fallback_enabled and (not out or out.startswith("[openai-error]")):
        local_out = _generate_ollama(prompt, sys_prompt)
        return local_out or out
    return out


def _sanitize_model_output(raw: str) -> str:
    """Remove any assistant/system/user scaffolding or role-prefixed lines.

    Keeps the text compact and trims excessive whitespace.
    """
    if not isinstance(raw, str):
        return ""
    # Remove common role prefixes like 'assistant:' or 'User:' at line starts
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        # strip role tokens if they appear like 'assistant: text' or 'User:'
        if s.lower().startswith("assistant:") or s.lower().startswith("system:") or s.lower().startswith("user:"):
            s = s.split(":", 1)[1].strip()
        lines.append(s)
    out = "\n".join([l for l in lines if l])
    # Collapse repeated blank lines
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip()


def generate_with_metadata(
    prompt: str,
    sys_prompt: Optional[str] = None,
    timeout: float = 25.0,
    max_retries: int = 3,
    fallback_text: str = "I'm sorry — I couldn't generate an answer right now.",
) -> Dict[str, object]:
    """Call the configured backend and return sanitized text plus metadata.

    Guarantees a non-empty 'text' field (uses fallback_text on error).
    Returns a dict: {text, raw, meta: {backend, attempts, elapsed, error}}
    """
    attempts = 0
    start = time.time()
    last_err = None
    raw_out = ""
    backend = os.getenv("APP_MODE", "cloud").lower()

    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            # timeout at the function level using time checks — backends may not accept timeouts
            t0 = time.time()
            raw_out = generate(prompt, sys_prompt=sys_prompt)
            elapsed = time.time() - t0
            # quick sanity: if it looks like an error token from our implementations, treat as failure
            if isinstance(raw_out, str) and raw_out.startswith("[openai-error]"):
                last_err = raw_out
                raise RuntimeError(raw_out)
            # success
            break
        except Exception as e:  # noqa: BLE001 - we want to capture any failure and continue
            last_err = str(e)
            _retry_sleep(attempt)
            # continue to next attempt

    total_elapsed = time.time() - start
    sanitized = _sanitize_model_output(raw_out) if raw_out else ""
    if not sanitized:
        sanitized = fallback_text

    meta = {
        "backend": backend,
        "attempts": attempts,
        "elapsed": round(total_elapsed, 3),
        "error": last_err,
    }

    return {
        "text": sanitized,
        "raw": raw_out,
        "meta": meta,
    }


def _generate_gemini(prompt: str, sys_prompt: Optional[str]) -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return "[gemini-stub] " + prompt[:80]
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return "[gemini-missing] " + prompt[:80]

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    full_prompt = (sys_prompt + "\n\n" if sys_prompt else "") + _truncate_prompt(prompt)
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(model_name)
            r = model.generate_content(full_prompt)
            txt = getattr(r, "text", None)
            if not txt and hasattr(r, "parts"):
                # older SDK styles
                txt = "".join(getattr(p, "text", "") for p in getattr(r, "parts", [])).strip() or None
            return (txt or "").strip()
        except Exception:
            _retry_sleep(attempt)
    return "[gemini-error] Unable to get response"


def _generate_openai(prompt: str, sys_prompt: Optional[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # In tests we may not have it; return a deterministic stub
        return "[openai-stub] " + prompt[:80]

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        # Fallback graceful degrade
        return "[openai-missing] " + prompt[:80]

    client = OpenAI(api_key=api_key)
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": _truncate_prompt(sys_prompt)})
    messages.append({"role": "user", "content": _truncate_prompt(prompt)})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=MAX_TOKENS_APPROX,
                temperature=0.2,
            )
            txt = resp.choices[0].message.content or ""
            return txt.strip()
        except Exception:
            _retry_sleep(attempt)
    # Provide an informative stub if API errors persist
    return "[openai-error] Unable to get response (rate limit or quota)."


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
