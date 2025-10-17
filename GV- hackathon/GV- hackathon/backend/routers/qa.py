from fastapi import APIRouter, HTTPException
from backend.services.doc_store import DocStore
from backend.services.ai_adapter import query_index, answer_question_from_context, build_index
from backend.models.schemas import QAReq, QAResp
import logging

logger = logging.getLogger("backend.qa")
router = APIRouter()
doc_store = DocStore()

@router.post("/{doc_id}/qa", response_model=QAResp)
async def qa(doc_id: str, req: QAReq):
    try:
        logger.info(f"❓ Q&A request: doc_id={doc_id}, question={req.question}, k={req.k}")
        doc = doc_store.get(doc_id)
        if not doc:
            logger.warning(f"⚠️ Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Query vector index
        logger.debug(f"Querying vector index with k={req.k}...")
        index = doc.get("index")
        if not index:
            try:
                logger.info("ℹ️ Index missing for doc; rebuilding now...")
                index = build_index(doc.get("page_contexts", []))
                doc["index"] = index
            except Exception as e:
                logger.warning(f"⚠️ Failed to rebuild index: {e}")
        results = query_index(index, req.question, k=req.k)
        logger.debug(f"Found {len(results)} relevant chunks")
        
        # Build context from results
        context = "\n\n".join([r.get('text', '') or r.get('page_context', '') for r in results])
        if not context:
            # Fallback to concatenated page contexts
            pcs = doc.get("page_contexts", [])
            context = "\n\n".join([(pc.get('page_context') or pc.get('text') or '') for pc in pcs])
        
        # Generate answer
        logger.debug("Generating answer...")
        answer = answer_question_from_context(context, req.question)
        logger.info(f"✅ Answer generated: {len(answer)} chars")
        
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Q&A failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
