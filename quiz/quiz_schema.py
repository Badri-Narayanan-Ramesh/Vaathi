# quiz_schema.py
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel

QType = Literal["mcq_single", "mcq_multi", "true_false", "cloze", "short_answer"]

class Choice(BaseModel):
    id: str         # "A","B","C","D"
    text: str

class GradingRule(BaseModel):
    type: Literal["exact","regex","keywords","semantic"]
    value: Any      # string | list[str] | regex pattern

class Question(BaseModel):
    id: str
    qtype: QType
    prompt: str
    choices: Optional[List[Choice]] = None
    answer: Any                      # "B" | ["A","D"] | True | "Faraday" | {"keywords":["..."]}
    rationale: Optional[str] = None
    difficulty: Literal["easy","medium","hard"]
    blooms: Literal["remember","understand","apply","analyze","evaluate","create"]
    grading: Optional[GradingRule] = None
    metadata: Dict[str, Any] = {}

class Quiz(BaseModel):
    id: str              # session_id + timestamp
    source_id: str       # slide_id / doc_hash
    title: str
    questions: List[Question]
    time_limit_sec: Optional[int] = None
    seed: Optional[int] = None
    version: int = 1
