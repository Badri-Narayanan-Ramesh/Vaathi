# Document store service for AI Tutor backend
# Handles in-memory and disk-based document storage, retrieval, and deletion

from typing import Dict, Any, Optional
import threading
from pathlib import Path
import json


DOCS_DIR = Path("./data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

class DocStore:
    # Process-wide shared storage so multiple instances across modules share state
    _store: Dict[str, Dict[str, Any]] = {}
    _lock: threading.Lock = threading.Lock()
    _loaded_from_disk: bool = False

    def __init__(self):
        # Load any persisted docs on first instantiation only
        if not DocStore._loaded_from_disk:
            self._load_from_disk()
            DocStore._loaded_from_disk = True

    def save(self, doc_id: str, page_contexts: Any, index: Any, name: str):
        with DocStore._lock:
            DocStore._store[doc_id] = {
                'page_contexts': page_contexts,
                'index': index,
                'name': name
            }
            self._save_to_disk(doc_id)

    def get(self, doc_id: str) -> Optional[Any]:
        with DocStore._lock:
            return DocStore._store.get(doc_id)

    def delete(self, doc_id: str):
        with DocStore._lock:
            if doc_id in DocStore._store:
                del DocStore._store[doc_id]
            f = DOCS_DIR / f"{doc_id}.json"
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass

    def list_ids(self):
        with DocStore._lock:
            return list(DocStore._store.keys())

    def update(self, doc_id: str, **kwargs):
        """Update fields for a document and persist to disk."""
        with DocStore._lock:
            if doc_id not in DocStore._store:
                return
            DocStore._store[doc_id].update({k: v for k, v in kwargs.items() if v is not None})
            self._save_to_disk(doc_id)

    def _save_to_disk(self, doc_id: str):
        data = DocStore._store.get(doc_id)
        if not data:
            return
        # Persist only what we can serialize easily
        serializable = {
            'doc_id': doc_id,
            'name': data.get('name'),
            'page_contexts': data.get('page_contexts'),
            # Do not attempt to store the index (rebuild on demand)
        }
        try:
            out_path = DOCS_DIR / f"{doc_id}.json"
            out_path.write_text(json.dumps(serializable, ensure_ascii=False))
        except Exception:
            # Non-fatal
            pass

    def _load_from_disk(self):
        try:
            for f in DOCS_DIR.glob("*.json"):
                try:
                    obj = json.loads(f.read_text(encoding="utf-8"))
                    doc_id = obj.get('doc_id') or f.stem
                    DocStore._store[doc_id] = {
                        'name': obj.get('name', 'Untitled'),
                        'page_contexts': obj.get('page_contexts', []),
                        'index': None,  # build lazily when needed
                    }
                except Exception:
                    continue
        except Exception:
            # Non-fatal
            pass
