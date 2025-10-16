"""Thin retrieval wrapper around the Chroma index."""
from __future__ import annotations

from typing import Any, List, Dict

from .embeddings import query_index


def retrieve_for_question(index: Any, question: str, k: int = 3) -> List[Dict]:
    """Retrieve top-k page chunks for a question.

    Returns a list of dicts with keys: page_id, text, score.
    """
    return query_index(index, question, k=k)
