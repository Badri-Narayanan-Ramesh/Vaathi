"""Prompt templates for AI Tutor chains."""
from __future__ import annotations

from typing import List


EXPLAIN_PAGE = (
    "You are a friendly, clear, and intuitive tutor.\n"
    "Explain the topic in a very simple, short, and intuitive way.\n\n"

    "Format:\n"
    "# Title\n\n"
    "- 4–6 short bullet points (no long sentences)\n"
    "**Key Takeaway:** One-line summary.\n\n"

    "Guidelines:\n"
    "- Use everyday analogies when helpful\n"
    "- No jargon unless absolutely needed\n"
    "- No paragraphs; use clean line breaks\n"
    "- Calm, confident, and clear tone\n\n"

    "[PAGE CONTEXT]\n{page_context}\n\n"
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
    "You are a structured content generator. Create a visually clean, minimal, and concise cheatsheet.\n\n"

    "Guidelines:\n"
    "- Use clear section headers with icon pointers (e.g., 📌 Concepts, 🔑 Formulas, 💡 Syntax, 📋 Examples, ⚡ Shortcuts)\n"
    "- Prefer bullet points, tables, and code blocks over paragraphs\n"
    "- Keep text short and crisp, but informative\n"
    "- Include common pitfalls, best practices, or mnemonics if relevant\n"
    "- Use ✅, ⚠️, 📝 icons for quick scanning where helpful\n"
    "- Avoid unnecessary explanations — assume the reader knows basics\n\n"

    "Format:\n"
    "# Title\n\n"
    "## 📌 Concepts\n"
    "- Bullet\n"
    "- Bullet\n\n"
    "## 🔑 Formulas (omit if none)\n"
    "| Name | Formula |\n|------|---------|\n|  |  |\n\n"
    "## 💡 Syntax / Examples (optional)\n"
    "```\nshort example\n```\n\n"
    "## ⚡ Shortcuts / Best Practices / Pitfalls (optional)\n"
    "- ✅ Best practice\n- ⚠️ Pitfall\n- 📝 Mnemonic\n\n"

    "Constraints:\n"
    "- Keep it compact; no paragraphs\n"
    "- Prefer 1-line bullets; total length <= 150 words\n\n"

    "[CONTEXT]\n{page_context}\n\n[CHEATSHEET]"
)


def render_ctx_blocks(contexts: List[str]) -> str:
    """Join context blocks using a clear separator for the LLM."""
    ctxs = [c.strip() for c in contexts if c and isinstance(c, str)]
    return "\n---\n".join(ctxs)
