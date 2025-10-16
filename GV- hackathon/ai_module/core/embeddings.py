"""Embedding index management using sentence-transformers and Chroma.

This keeps a simple API: build_index and query_index.
"""
from __future__ import annotations

from typing import Any, List, Dict


def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "sentence-transformers is required. Please install 'sentence-transformers'."
        ) from e
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _load_chroma():
    try:
        import chromadb  # type: ignore
        from chromadb.utils import embedding_functions  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("chromadb is required. Please install 'chromadb'.") from e
    return chromadb


def build_index(page_contexts: List[Dict], index_dir: str) -> Any:
    """Build a Chroma index storing page contexts.

    Each page_context dict must contain keys: 'page_id' and 'page_context'.
    Returns a Chroma collection instance (acts as our index).
    """
    chromadb = _load_chroma()
    client = chromadb.PersistentClient(path=index_dir)
    collection = client.get_or_create_collection(name="pages")

    embedder = _load_embedder()

    documents = [pc.get("page_context", "") for pc in page_contexts]
    ids = [str(pc.get("page_id", i)) for i, pc in enumerate(page_contexts)]
    metadatas = [{"page_id": int(pc.get("page_id", i))} for i, pc in enumerate(page_contexts)]

    # Clear and add fresh to keep idempotent for tests
    try:
        existing = collection.count()
        if existing:
            collection.delete(where={})
    except Exception:
        pass

    # Pre-compute embeddings to avoid multiple model loads
    vectors = embedder.encode(documents, show_progress_bar=False, normalize_embeddings=True)

    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=vectors)
    return collection


def query_index(index: Any, text: str, k: int = 3) -> List[Dict]:
    """Query the Chroma index and return top-k results with page_id and score."""
    if not text:
        return []
    try:
        results = index.query(query_texts=[text], n_results=max(1, int(k)))
        out: List[Dict] = []
        for i in range(len(results.get("ids", [[]])[0])):
            pid = int(results["metadatas"][0][i]["page_id"])  # type: ignore[index]
            doc = results["documents"][0][i]
            dist = results.get("distances")
            score = float(dist[0][i]) if dist else 0.0
            out.append({"page_id": pid, "text": doc, "score": score})
        return out
    except Exception:
        return []
