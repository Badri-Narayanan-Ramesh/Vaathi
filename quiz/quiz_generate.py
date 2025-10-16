# quiz_generate.py
import json, uuid, random
from quiz_schema import Quiz
from typing import Dict

class LLMInterface:
    def __init__(self, provider="openai", model="gpt-4o-mini", local=False):
        self.provider = provider
        self.model = model
        self.local = local
    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict:
        # Pseudocode — plug your LLM client here. Ensure JSON-only output via tools/json mode if available.
        # return client.chat.completions.create(..., response_format={"type":"json_object"})
        raise NotImplementedError

def make_quiz(context: str, n=6, mix="mcq_single:70,true_false:20,cloze:10",
              source_id="slide_1", title="Quick Check", llm: LLMInterface=None) -> Quiz:
    sys = "You are an expert pedagogue... Return valid JSON for the schema Quiz."
    user = f"CONTEXT:\n{context}\n\nTASK:\nCreate {n} questions with mix {mix} ... OUTPUT: Return a single JSON object exactly of type Quiz."
    raw = llm.generate_json(sys, user)
    quiz = Quiz(**raw)
    # Safety: inject IDs if missing
    for i,q in enumerate(quiz.questions,1):
        if not q.id: q.id = f"Q{i}"
    if not quiz.id: quiz.id = f"{source_id}-{uuid.uuid4().hex[:8]}"
    return quiz
