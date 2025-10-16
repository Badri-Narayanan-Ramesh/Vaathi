# Question Types (MVP → Extensible)

MCQ-1 (single correct, 3–4 distractors)

MCQ-N (multi-select)

TF (true/false + rationale)

Cloze (fill-in-the-blank; exact/regex grading)

ShortAnswer (open response → rubric + semantic similarity)

Optional (stretch): Ordering, Match-pairs, Image-based MCQ (from slide crops + BLIP captions)

# Adaptive Difficulty (Optional, fast)

Maintain skill[q.blooms][topic] ∈ [0,1]

After grading, update with EMA:
skill ← 0.8*skill + 0.2*(correct ? 1 : 0)

Next generation prompt: raise proportion of weak Bloom levels & topics:
mix = "mcq_single:50,true_false:20,cloze:30", bias blooms toward weaker ones.

# Guardrails & UX Details

Determinism: seed distractor order; shuffle per attempt.

Clarity: 12–18 words average per stem; avoid negations (“NOT”).

Rationales: always keep; drives learning.

Accessibility: keyboard navigation for options; TTS button for stems & rationale.