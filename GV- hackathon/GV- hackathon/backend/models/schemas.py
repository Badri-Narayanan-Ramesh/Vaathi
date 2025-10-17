# Pydantic models for FastAPI endpoints
from pydantic import BaseModel
from typing import List, Optional, Any

class UploadResp(BaseModel):
    doc_id: str
    name: str
    page_count: int

class PageInfo(BaseModel):
    page_id: int

class PagesResp(BaseModel):
    pages: List[PageInfo]

class ExplainResp(BaseModel):
    page_id: int
    explanation: str

class QAReq(BaseModel):
    question: str
    k: int = 3

class QAResp(BaseModel):
    answer: str

class FlashcardsResp(BaseModel):
    items: List[Any]

class QuizResp(BaseModel):
    items: List[Any]

class CheatsheetResp(BaseModel):
    content: str

class TTSResp(BaseModel):
    audio_url: str

class STTResp(BaseModel):
    text: str
