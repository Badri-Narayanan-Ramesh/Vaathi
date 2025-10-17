from fastapi import APIRouter, HTTPException
from backend.services.doc_store import DocStore
from backend.services.ai_adapter import generate_explanation, build_index
from backend.models.schemas import PagesResp, ExplainResp
import logging

logger = logging.getLogger("backend.pages")
router = APIRouter()
doc_store = DocStore()

@router.get("/{doc_id}/pages", response_model=PagesResp)
async def list_pages(doc_id: str):
    try:
        logger.info(f"📄 List pages request: doc_id={doc_id}")
        doc = doc_store.get(doc_id)
        if not doc:
            logger.warning(f"⚠️ Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        pages = [{"page_id": i} for i in range(len(doc["page_contexts"]))]
        logger.info(f"✅ Found {len(pages)} pages")
        return {"pages": pages}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ List pages failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{doc_id}/pages/{page_id}/explain", response_model=ExplainResp)
async def explain_page(doc_id: str, page_id: int):
    try:
        logger.info(f"💡 Explain page request: doc_id={doc_id}, page_id={page_id}")
        doc = doc_store.get(doc_id)
        if not doc:
            logger.warning(f"⚠️ Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        page_contexts = doc["page_contexts"]
        # Rebuild index lazily if missing (e.g., after server restart)
        if not doc.get("index"):
            logger.info("ℹ️ Index missing for doc; rebuilding now...")
            try:
                idx = build_index(page_contexts)
                doc["index"] = idx
            except Exception as e:
                logger.warning(f"⚠️ Failed to rebuild index: {e}")
        if page_id < 0 or page_id >= len(page_contexts):
            logger.warning(f"⚠️ Invalid page_id: {page_id} (total pages: {len(page_contexts)})")
            raise HTTPException(status_code=404, detail="Page not found")
        
        page_context = page_contexts[page_id]
        logger.debug(f"Generating explanation for page {page_id}...")
        # our ingest uses 'page_context' field for combined text
        base_text = page_context.get('page_context') or page_context.get('text', '')
        explanation = generate_explanation(base_text)
        logger.info(f"✅ Explanation generated: {len(explanation)} chars")
        
        return {"page_id": page_id, "explanation": explanation}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Explain page failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
