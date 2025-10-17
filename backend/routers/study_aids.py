from fastapi import APIRouter, HTTPException
from backend.services.doc_store import DocStore
from backend.services.ai_adapter import generate_flashcards, generate_quiz, generate_cheatsheet
from backend.models.schemas import FlashcardsResp, QuizResp, CheatsheetResp
import logging

logger = logging.getLogger("backend.study_aids")
router = APIRouter()
doc_store = DocStore()

@router.get("/{doc_id}/pages/{page_id}/flashcards", response_model=FlashcardsResp)
async def flashcards(doc_id: str, page_id: int, model: str = None):
    try:
        logger.info(f"🎴 Flashcards request: doc_id={doc_id}, page_id={page_id}, model={model}")
        doc = doc_store.get(doc_id)
        if not doc:
            logger.warning(f"⚠️ Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        page_contexts = doc["page_contexts"]
        if page_id < 0 or page_id >= len(page_contexts):
            logger.warning(f"⚠️ Invalid page_id: {page_id}")
            raise HTTPException(status_code=404, detail="Page not found")
        
        page_text = page_contexts[page_id].get('page_context', '')
        logger.debug(f"Generating flashcards for page {page_id}...")
        items = generate_flashcards(page_text, model)
        logger.info(f"✅ Generated {len(items)} flashcards")
        return {"items": items}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Flashcards failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{doc_id}/pages/{page_id}/quiz", response_model=QuizResp)
async def quiz(doc_id: str, page_id: int, model: str = None):
    try:
        logger.info(f"📝 Quiz request: doc_id={doc_id}, page_id={page_id}, model={model}")
        doc = doc_store.get(doc_id)
        if not doc:
            logger.warning(f"⚠️ Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        page_contexts = doc["page_contexts"]
        if page_id < 0 or page_id >= len(page_contexts):
            logger.warning(f"⚠️ Invalid page_id: {page_id}")
            raise HTTPException(status_code=404, detail="Page not found")
        
        page_text = page_contexts[page_id].get('page_context', '')
        logger.debug(f"Generating quiz for page {page_id}...")
        items = generate_quiz(page_text, model)
        logger.info(f"✅ Generated {len(items)} quiz questions")
        return {"items": items}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Quiz failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{doc_id}/pages/{page_id}/cheatsheet", response_model=CheatsheetResp)
async def cheatsheet(doc_id: str, page_id: int, model: str = None):
    try:
        logger.info(f"📋 Cheatsheet request: doc_id={doc_id}, page_id={page_id}, model={model}")
        doc = doc_store.get(doc_id)
        if not doc:
            logger.warning(f"⚠️ Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        page_contexts = doc["page_contexts"]
        if page_id < 0 or page_id >= len(page_contexts):
            logger.warning(f"⚠️ Invalid page_id: {page_id}")
            raise HTTPException(status_code=404, detail="Page not found")
        
        page_text = page_contexts[page_id].get('page_context', '')
        logger.debug(f"Generating cheatsheet for page {page_id}...")
        content = generate_cheatsheet(page_text, model)
        logger.info(f"✅ Cheatsheet generated: {len(content)} chars")
        return {"content": content}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Cheatsheet failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
