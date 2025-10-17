"""Prompt templates for AI Tutor chains."""
from __future__ import annotations

from typing import List


EXPLAIN_PAGE = (
    "You are a friendly tutor. Read the page context and explain it to a student in plain English.\n"
    "Write 4–6 clear bullets that teach the idea (no jargon unless necessary),\n"
    "use examples or analogies when helpful, avoid repeating phrases from the slide,\n"
    "and prefer short, varied sentences. Then add a brief 2–3 sentence summary.\n\n"
    "Rules:\n- Do not copy text verbatim; paraphrase naturally.\n- Avoid repeating the same term more than twice.\n"
    "- Prefer everyday language over academic phrasing.\n- Keep it actionable and memorable.\n\n"
    "[PAGE CONTEXT]\n{page_context}\n\n"
    "[OUTPUT FORMAT]\n- Bullet 1\n- Bullet 2\n- Bullet 3\n- Bullet 4\n- Bullet 5 (optional)\n- Bullet 6 (optional)\n\nSummary:\n"
)


ANSWER_WITH_CITATIONS = (
    "Answer the question STRICTLY using only the provided contexts.\n"
    "Be concise and synthesized (no copy-paste), and cite slide numbers in square brackets like [Slide 3, Slide 5].\n"
    "If information is missing, state that clearly.\n\n"
    "[CONTEXTS]\n{contexts}\n\n[QUESTION]\n{question}\n\n[ANSWER]"
)


FLASHCARDS_FROM_CONTEXT = (
    "From the context, produce 5 high-quality flashcards in the exact format:\n"
    "Q: <short question>\nA: <concise answer>\n"
    "Focus on definitions, why/how, and key contrasts.\n\n"
    "[CONTEXT]\n{page_context}\n\n[FLASHCARDS]"
)


QUIZ_FROM_CONTEXT = (
    "From the context, generate 3 multiple-choice questions with options A–D.\n"
    "Keep stems short and options plausible (one best answer).\n"
    "After the questions, include an 'Answers:' line like: Answers: 1) B, 2) D, 3) A.\n\n"
    "[CONTEXT]\n{page_context}\n\n[QUIZ]"
)


CHEATSHEET_FROM_CONTEXT = (
    "Create a concise cheatsheet with tight bullets: definitions, key formulas, rules of thumb, and steps.\n"
    "Prefer one-line bullets that someone can revise quickly.\n\n"
    "[CONTEXT]\n{page_context}\n\n[CHEATSHEET]"
)


# === Voice-friendly prompts ===
VOICE_EXPLAIN_SHORT = (
    "You are a friendly spoken-word tutor. Speak in short, clear sentences suitable for Text-to-Speech.",
    "Use natural rhythm, simple vocabulary, and include brief pauses where appropriate.",
    "Provide 3–5 short bullets that can be spoken one at a time, and finish with a single 1–2 sentence summary.",
    "Do not use markup or lists characters like '-' or '*' — produce plain text lines separated by newlines.",
)


VOICE_ANSWER_SHORT = (
    "Answer the question as if speaking to a student. Use short sentences and a calm, neutral tone.",
    "If you must cite slides, place citations in parentheses like (Slide 3). Keep them brief.",
    "Avoid long paragraphs; prefer 2–4 short sentences. Make the response TTS-friendly (no extra brackets or role tags).",
)


def render_voice_prompt(base_text: str, extra_instructions: list[str] | None = None) -> str:
    """Compose a voice-friendly system prompt combining base text and optional extra instructions.

    Keeps outputs plain and TTS-ready.
    """
    parts = [base_text.strip()]
    if extra_instructions:
        parts.extend(i.strip() for i in extra_instructions if i)
    return "\n".join(parts)


def render_ctx_blocks(contexts: List[str]) -> str:
    """Join context blocks using a clear separator for the LLM."""
    ctxs = [c.strip() for c in contexts if c and isinstance(c, str)]
    return "\n---\n".join(ctxs)
