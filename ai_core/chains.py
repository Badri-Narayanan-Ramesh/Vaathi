"""LangChain-like lightweight chains that call llm_client under the hood.

We avoid heavy agent machinery; these are thin adapters building prompts and
parsing outputs. Results cached in-memory keyed by (fn_name, prompt_hash).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple, Optional
import logging
import re
import os

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
logger = logging.getLogger("ai_core.chains")


def _cache_key(name: str, content: str) -> Tuple[str, str]:
    # Include model info in cache key to prevent cross-model caching
    app_mode = os.getenv("APP_MODE", "local")
    model = os.getenv("GEMINI_MODEL" if app_mode == "cloud" else "OLLAMA_MODEL", "default")
    combined = f"{app_mode}:{model}:{content}"
    h = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return (name, h)


def explain_page(page_context: str) -> str:
    prompt = EXPLAIN_PAGE.format(page_context=page_context)
    key = _cache_key("explain_page", prompt)
    app_mode = os.getenv("APP_MODE", "local")
    model = os.getenv("GEMINI_MODEL" if app_mode == "cloud" else "OLLAMA_MODEL", "default")
    logger.debug(f"explain_page: app_mode={app_mode}, model={model}, cache_key={key[1][:16]}")
    if key in _CACHE:
        logger.debug(f"explain_page: CACHE HIT")
        return _CACHE[key]
    logger.debug(f"explain_page: CACHE MISS, generating")
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

    if os.getenv("APP_MODE", "local").lower() == "cloud":
        try:
            with open("./out/last_flashcards_raw.txt", "w", encoding="utf-8") as f:
                f.write(raw)
        except Exception:
            pass

    # Parse into Q/A pairs
    cards: List[Dict] = []
    current_q: Optional[str] = None
    current_a: Optional[str] = None
    q_pattern = re.compile(r"^(?:q(?:uestion)?\s*\d*\s*[:.)-])\s*(.+)$", re.IGNORECASE)
    a_pattern = re.compile(r"^(?:a(?:nswer)?\s*\d*\s*[:.)-])\s*(.+)$", re.IGNORECASE)

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        mq = q_pattern.match(s)
        ma = a_pattern.match(s)
        if mq:
            if current_q and current_a:
                cards.append({"q": current_q.strip(), "a": current_a.strip()})
                current_q, current_a = None, None
            current_q = mq.group(1).strip()
        elif ma:
            current_a = ma.group(1).strip()
            if current_q:
                cards.append({"q": current_q.strip(), "a": current_a.strip()})
                current_q, current_a = None, None

    if current_q and current_a:
        cards.append({"q": current_q.strip(), "a": current_a.strip()})

    # Block-based fallback parsing
    if not cards:
        block_pattern = re.compile(
            r"(q(?:uestion)?\s*\d*\s*[:.)-].*?)\s+(a(?:nswer)?\s*\d*\s*[:.)-].*?)(?=\n\s*q(?:uestion)?\s*\d*\s*[:.)-]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        for m in block_pattern.finditer(raw):
            q_block = m.group(1)
            a_block = m.group(2)
            q_text = q_pattern.sub("", q_block).strip()
            a_text = a_pattern.sub("", a_block).strip()
            if q_text and a_text:
                cards.append({"q": q_text, "a": a_text})

    # Final fallback: sentence-based simple cards (ensure at least 3)
    if not cards and raw.strip():
        sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
        for s in sentences[:3]:
            if s:
                cards.append({"q": "What is a key point from the page?", "a": s.strip()})

    # Ensure at least 3 cards by synthesizing summaries
    while len(cards) < 3 and page_context.strip():
        snippet = page_context[len(cards)*80:(len(cards)+1)*80].strip()
        if not snippet:
            break
        cards.append({"q": "Summarize this part", "a": snippet})

    logger.debug(f"Generated {len(cards)} flashcards after parsing")
    return cards


def make_quiz(page_context: str) -> List[Dict]:
    prompt = QUIZ_FROM_CONTEXT.format(page_context=page_context)
    key = _cache_key("make_quiz", prompt)
    if key in _CACHE:
        raw = _CACHE[key]
    else:
        raw = llm_client.generate(prompt)
        _CACHE[key] = raw

    # Log raw output for debugging when in cloud mode
    if os.getenv("APP_MODE", "local").lower() == "cloud":
        with open("./out/last_quiz_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw)

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Extract answer key if present
    answer_map: Dict[int, str] = {}
    key_re = re.compile(r"^(answers?|answer\s*key)\s*[:\-]\s*(.*)$", re.IGNORECASE)
    pair_re = re.compile(r"(\d+)\s*[):.\-]\s*([A-D])")
    for l in lines:
        m = key_re.match(l)
        if m:
            rest = m.group(2)
            for num, letter in pair_re.findall(rest):
                answer_map[int(num)] = letter.upper()

    # Parse questions/options
    items: List[Dict] = []
    current: Optional[Dict] = None
    qnum = 0
    q_start_re = re.compile(r"^(?:q(?:uestion)?\s*(\d+)|([0-9]+))\s*[).:\-]\s*(.+)$", re.IGNORECASE)
    opt_re = re.compile(r"^([A-D])[).:\-]\s*(.+)$")
    answer_line_re = re.compile(r"^(?:answer|correct\s*answer)\s*[:\-]\s*([A-D])", re.IGNORECASE)

    for l in lines:
        mq = q_start_re.match(l)
        if mq:
            # Flush previous
            if current and current.get("question") and current.get("options"):
                items.append(current)
            qnum = int(mq.group(1) or mq.group(2) or len(items) + 1)
            question_text = mq.group(3).strip()
            current = {"question": question_text, "options": [], "_qnum": qnum}
            continue

        mo = opt_re.match(l)
        if mo and current is not None:
            letter = mo.group(1)
            text = mo.group(2).strip()
            current.setdefault("options", []).append(text)
            if current.get("_qnum") in answer_map and answer_map[current["_qnum"]] == letter:
                current["answer"] = text
            continue

        ma = answer_line_re.match(l)
        if ma and current is not None and current.get("options"):
            letter = ma.group(1).upper()
            idx = ord(letter) - ord('A')
            if 0 <= idx < len(current["options"]):
                current["answer"] = current["options"][idx]
            continue

    # Flush tail
    if current and current.get("question") and current.get("options"):
        items.append(current)

    # Cleanup helper keys and ensure answers
    for it in items:
        it.pop("_qnum", None)
        if not it.get("answer") and it.get("options"):
            it["answer"] = it["options"][0]

    # Ensure at least 3-4 items by synthesizing
    def synthesize_q(prompt_text: str) -> Dict:
        base = prompt_text.strip()[:80] or "the page content"
        correct = f"Mentions: {base}"
        return {
            "question": "Which of the following is supported by the page?",
            "options": [correct, "Contradictory claim 1", "Contradictory claim 2", "Not mentioned"],
            "answer": correct,
        }

    while len(items) < 4:
        items.append(synthesize_q(page_context[len(items)*60:]))

    return items


def make_cheatsheet(page_context: str) -> str:
    prompt = CHEATSHEET_FROM_CONTEXT.format(page_context=page_context)
    key = _cache_key("make_cheatsheet", prompt)
    app_mode = os.getenv("APP_MODE", "local")
    model = os.getenv("GEMINI_MODEL" if app_mode == "cloud" else "OLLAMA_MODEL", "default")
    logger.debug(f"make_cheatsheet: app_mode={app_mode}, model={model}, cache_key={key[1][:16]}")
    if key in _CACHE:
        logger.debug(f"make_cheatsheet: CACHE HIT for key={key[1][:16]}")
        return _CACHE[key]
    logger.debug(f"make_cheatsheet: CACHE MISS, generating new content")
    out = llm_client.generate(prompt)
    _CACHE[key] = out
    logger.debug(f"make_cheatsheet: cached result with key={key[1][:16]}")
    return out
