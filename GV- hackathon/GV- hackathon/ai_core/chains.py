"""LangChain-like lightweight chains that call llm_client under the hood.

We avoid heavy agent machinery; these are thin adapters building prompts and
parsing outputs. Results cached in-memory keyed by (fn_name, prompt_hash).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from . import llm_client
from .prompts import (
    EXPLAIN_PAGE,
    ANSWER_WITH_CITATIONS,
    FLASHCARDS_FROM_CONTEXT,
    QUIZ_FROM_CONTEXT,
    CHEATSHEET_FROM_CONTEXT,
    render_ctx_blocks,
)


_CACHE: Dict[Tuple[str, str], str] = {}


def _cache_key(name: str, content: str) -> Tuple[str, str]:
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return (name, h)


def explain_page(page_context: str) -> str:
    prompt = EXPLAIN_PAGE.format(page_context=page_context)
    key = _cache_key("explain_page", prompt)
    if key in _CACHE:
        return _CACHE[key]
    out = llm_client.generate(prompt, sys_prompt="You are a precise teaching assistant.")
    _CACHE[key] = out
    return out


def answer_question(question: str, ctx_texts: List[str]) -> str:
    contexts = render_ctx_blocks(ctx_texts)
    prompt = ANSWER_WITH_CITATIONS.format(contexts=contexts, question=question)
    key = _cache_key("answer_question", prompt)
    if key in _CACHE:
        return _CACHE[key]
    out = llm_client.generate(prompt, sys_prompt="Cite slides as [Slide n]. Only use given contexts.")
    _CACHE[key] = out
    return out


def make_flashcards(page_context: str) -> List[Dict]:
    prompt = FLASHCARDS_FROM_CONTEXT.format(page_context=page_context)
    key = _cache_key("make_flashcards", prompt)
    if key in _CACHE:
        raw = _CACHE[key]
    else:
        raw = llm_client.generate(prompt)
        _CACHE[key] = raw
    cards: List[Dict] = []
    q, a = None, None
    import os
    # Log raw output for debugging
    if os.getenv("APP_MODE", "local").lower() == "cloud":
        with open("./out/last_flashcards_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw)
    for line in raw.splitlines():
        s = line.strip()
        if s.lower().startswith("q:"):
            if q and a:
                cards.append({"q": q, "a": a})
            q = s[2:].strip()
            a = None
        elif s.lower().startswith("a:"):
            a = s[2:].strip()
    if q and a:
        cards.append({"q": q, "a": a})
    # If no cards parsed, fallback to raw text
    if not cards and raw.strip():
        cards.append({"q": "RAW", "a": raw.strip()})
    return cards


def make_quiz(page_context: str) -> List[Dict]:
    prompt = QUIZ_FROM_CONTEXT.format(page_context=page_context)
    key = _cache_key("make_quiz", prompt)
    if key in _CACHE:
        raw = _CACHE[key]
    else:
        raw = llm_client.generate(prompt)
        _CACHE[key] = raw
    import os
    # Log raw output for debugging
    if os.getenv("APP_MODE", "local").lower() == "cloud":
        with open("./out/last_quiz_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw)
    # Parse simple format: numbered questions with A-D options, then Answers: 1) X, ...
    items: List[Dict] = []
    current: Dict = {}
    for line in raw.splitlines():
        s = line.strip()
        if (len(s) >= 2 and s[0].isdigit() and (s[1] in ").")):
            if current:
                items.append(current)
            current = {"q": s, "options": []}
        elif (len(s) >= 2 and s[0] in "ABCD" and s[1] in "):"):
            # Keep simple: treat any line starting with A) or A: as option
            current.setdefault("options", []).append(s)
        elif s.lower().startswith("answers:"):
            # attach answers to the last element for simplicity
            current["answers"] = s
    if current:
        items.append(current)
    # If no items parsed, fallback to raw text
    if not items and raw.strip():
        items.append({"q": "RAW", "options": [raw.strip()]})
    return items


def make_cheatsheet(page_context: str) -> str:
    prompt = CHEATSHEET_FROM_CONTEXT.format(page_context=page_context)
    key = _cache_key("make_cheatsheet", prompt)
    if key in _CACHE:
        return _CACHE[key]
    out = llm_client.generate(prompt)
    _CACHE[key] = out
    return out
