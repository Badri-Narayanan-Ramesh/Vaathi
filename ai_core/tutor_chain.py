"""Tutor chain: builds system prompt, validates queries, and calls LLM client.

Returns structured output: {text, raw, meta, ok, reason}
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from . import llm_client
from .prompts import EXPLAIN_PAGE

import time
import uuid
from typing import Any, List

from . import stt_client
from . import tts_client
from .prompts import render_voice_prompt


SYSTEM_PROMPT = (
    "You are an expert, friendly tutor. Be concise, use plain language, provide 3-5 bullets when explaining, "
    "and finish with a 2-3 sentence summary. Avoid quoting source text verbatim."
)


def _validate_query(text: Optional[str]) -> Tuple[bool, str]:
    if not text:
        return False, "empty"
    t = text.strip()
    if not t:
        return False, "empty"
    if len(t) > 5000:
        return False, "too_long"
    return True, "ok"


def answer_once(query: str, contexts: Optional[str] = None) -> Dict[str, object]:
    """Process a single-turn question.

    contexts: optional string providing page or slide context. If provided, it's appended to prompt.
    Returns: {ok, reason, text, raw, meta}
    """
    try:
        ok, reason = _validate_query(query)
        if not ok:
            return {"ok": False, "reason": reason, "text": "", "raw": "", "meta": {}}

        prompt = query
        if contexts:
            # Lean: put contexts after system message
            prompt = f"[CONTEXT]\n{contexts}\n\n[QUESTION]\n{query}"

        resp = llm_client.generate_with_metadata(prompt, sys_prompt=SYSTEM_PROMPT)
        # Ensure text sanitized (generate_with_metadata already sanitizes)
        out_text = resp.get("text", "")
        return {"ok": True, "reason": "generated", "text": out_text, "raw": resp.get("raw"), "meta": resp.get("meta", {})}
    except Exception as e:
        return {"ok": False, "reason": "exception", "text": "", "raw": "", "meta": {"error": str(e)}}


# --------------------------
# Conversation manager
# --------------------------


class ConversationManager:
    """In-memory conversation manager for short-lived sessions.

    Stores recent messages and enforces simple size limits. This is intentionally
    lightweight and not persistent; you can replace with a DB later.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def start(self, session_id: Optional[str] = None) -> str:
        sid = session_id or str(uuid.uuid4())
        self._sessions[sid] = {"created": time.time(), "messages": []}
        return sid

    def end(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def add_user(self, session_id: str, text: str) -> None:
        sess = self._sessions.setdefault(session_id, {"created": time.time(), "messages": []})
        sess["messages"].append({"role": "user", "text": text})
        # keep recent messages only
        if len(sess["messages"]) > 40:
            sess["messages"] = sess["messages"][-40:]

    def add_assistant(self, session_id: str, text: str) -> None:
        sess = self._sessions.setdefault(session_id, {"created": time.time(), "messages": []})
        sess["messages"].append({"role": "assistant", "text": text})
        if len(sess["messages"]) > 40:
            sess["messages"] = sess["messages"][-40:]

    def recent(self, session_id: str, limit: int = 8) -> List[Dict[str, str]]:
        sess = self._sessions.get(session_id)
        if not sess:
            return []
        return sess["messages"][-limit:]


# single global manager for simplicity
_CONV_MANAGER = ConversationManager()


def chat_turn(
    session_id: Optional[str] = None,
    user_text: Optional[str] = None,
    wav_bytes: Optional[bytes] = None,
    language: Optional[str] = None,
    contexts: Optional[str] = None,
    voice: bool = False,
    tts_rate: int = 180,
    tts_voice: Optional[str] = None,
    max_history_messages: int = 8,
) -> Dict[str, object]:
    """Process a conversation turn (text or audio) for a session.

    Returns a dict: {ok, session_id, user_text, answer_text, wav_bytes, meta, reasons}
    """
    try:
        # ensure session
        sid = session_id or _CONV_MANAGER.start()

        # 1) If audio provided, transcribe
        stt_meta = None
        if wav_bytes is not None:
            stt = stt_client.transcribe_wav_bytes(wav_bytes, language=language)
            stt_meta = stt
            user_text = stt.get("text", "")
            if not user_text:
                return {"ok": False, "session_id": sid, "reasons": ["no_transcript"], "user_text": "", "answer_text": "", "wav_bytes": b"", "meta": {"stt": stt_meta}}

        # validate user_text
        if not user_text or not user_text.strip():
            return {"ok": False, "session_id": sid, "reasons": ["empty_input"], "user_text": user_text or "", "answer_text": "", "wav_bytes": b"", "meta": {"stt": stt_meta}}

        user_text = user_text.strip()

        # append user message to session history
        _CONV_MANAGER.add_user(sid, user_text)

        # 2) Build prompt including system prompt and recent messages
        # Choose system prompt differently for voice vs text
        if voice:
            # import voice style from prompts
            from .prompts import VOICE_ANSWER_SHORT

            sys_prompt = render_voice_prompt(" ".join(VOICE_ANSWER_SHORT))
        else:
            sys_prompt = "You are a helpful tutor. Be concise and explanatory."

        # Build conversation block from recent messages
        recent = _CONV_MANAGER.recent(sid, limit=max_history_messages)
        convo_lines = []
        for m in recent:
            role = m.get("role", "user")
            text = m.get("text", "")
            # prefix roles to help LLM understand structure
            if role == "user":
                convo_lines.append(f"User: {text}")
            else:
                convo_lines.append(f"Assistant: {text}")

        convo_block = "\n".join(convo_lines)

        prompt_parts = ["[SYSTEM]\n" + sys_prompt]
        if contexts:
            prompt_parts.append("[CONTEXT]\n" + contexts)
        prompt_parts.append("[CONVERSATION]\n" + convo_block)
        prompt_parts.append("[NEW_USER]\n" + user_text)

        full_prompt = "\n\n".join(p for p in prompt_parts if p)

        # Safety: cap prompt length
        if len(full_prompt) > 10000:
            full_prompt = full_prompt[-10000:]

        # 3) Call LLM
        resp = llm_client.generate_with_metadata(full_prompt, sys_prompt=sys_prompt)
        answer_text = resp.get("text", "")

        # 4) store assistant reply
        if answer_text:
            _CONV_MANAGER.add_assistant(sid, answer_text)

        # 5) Optionally synthesize TTS
        wav_out = b""
        tts_meta = None
        if voice and answer_text:
            tts = tts_client.synthesize_text(answer_text, rate=tts_rate, voice=tts_voice)
            wav_out = tts.get("wav_bytes", b"")
            tts_meta = tts

        meta = {"stt": stt_meta, "llm": resp.get("meta"), "tts": tts_meta}

        return {"ok": True, "session_id": sid, "user_text": user_text, "answer_text": answer_text, "wav_bytes": wav_out, "meta": meta, "reasons": []}
    except Exception as e:
        return {"ok": False, "session_id": session_id or "", "reasons": ["exception"], "user_text": user_text or "", "answer_text": "", "wav_bytes": b"", "meta": {"error": str(e)}}
