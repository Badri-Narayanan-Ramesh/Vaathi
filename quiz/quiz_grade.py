# quiz_grade.py
import re
from typing import List, Dict, Any
from difflib import SequenceMatcher
from quiz_schema import Question

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter/union

def semantic_similarity(a: str, b: str) -> float:
    # Placeholder: plug in embeddings (local/all-MiniLM) for real use.
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def grade_question(q: Question, user_answer: Any) -> Dict[str, Any]:
    correct = False; score = 0.0; max_score = 1.0; feedback = ""
    if q.qtype == "mcq_single":
        correct = (user_answer == q.answer)
        score = 1.0 if correct else 0.0
    elif q.qtype == "mcq_multi":
        ua = set(user_answer or [])
        ca = set(q.answer if isinstance(q.answer, list) else [])
        if ua == ca: score = 1.0; correct = True
        else:
            # partial credit
            tp = len(ua & ca); fp = len(ua - ca); fn = len(ca - ua)
            score = max(0.0, (tp - 0.5*fp) / (tp+fn or 1))
            correct = score > 0.99
        feedback = f"Expected {sorted(ca)}"
    elif q.qtype == "true_false":
        correct = (bool(user_answer) == bool(q.answer))
        score = 1.0 if correct else 0.0
    elif q.qtype == "cloze":
        rule = q.grading or {"type":"exact","value":q.answer}
        if rule["type"] == "exact":
            score = 1.0 if _norm(user_answer)==_norm(rule["value"]) else 0.0
        elif rule["type"] == "regex":
            score = 1.0 if re.search(rule["value"], user_answer, re.I) else 0.0
        correct = score == 1.0
    elif q.qtype == "short_answer":
        rule = q.grading or {"type":"keywords","value":[str(q.answer)]}
        if rule["type"] == "keywords":
            kws = [ _norm(k) for k in rule["value"] ]
            hit = sum(1 for k in kws if k in _norm(user_answer))
            score = min(1.0, hit/max(1,len(kws)))
        elif rule["type"] == "semantic":
            score = semantic_similarity(user_answer, q.answer)
            score = 1.0 if score >= 0.82 else 0.0
        correct = score >= 0.99
        feedback = f"Model answer: {q.answer}"
    return {"correct": bool(correct), "score": float(round(score,3)), "max":1.0, "feedback": feedback, "rationale": q.rationale or ""}
