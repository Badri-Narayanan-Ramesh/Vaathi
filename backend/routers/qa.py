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
        logger.info(f"❓ Q&A request: doc_id={doc_id}, question={req.question}, k={req.k}, page_id={req.page_id}, model={req.model}")
        doc = doc_store.get(doc_id)
        if not doc:
            logger.warning(f"⚠️ Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        used_contexts = []
        citations = []

        # Build context: if page_id provided, prioritize that page's context
        pcs = doc.get("page_contexts", [])
        page_context = ""
        if req.page_id is not None and 0 <= req.page_id < len(pcs):
            page_context = pcs[req.page_id].get("page_context") or pcs[req.page_id].get("text") or ""
            if page_context:
                used_contexts.append(page_context[:300])
                citations.append({"page_id": req.page_id})

        # If no page context or to enrich, query vector index
        logger.debug(f"Querying vector index with k={req.k}...")
        index = doc.get("index")
        if not index:
            try:
                logger.info("ℹ️ Index missing for doc; rebuilding now...")
                index = build_index(pcs)
                doc["index"] = index
            except Exception as e:
                logger.warning(f"⚠️ Failed to rebuild index: {e}")
        results = query_index(index, req.question, k=req.k)
        logger.debug(f"Found {len(results)} relevant chunks")

        # Merge context from results
        for r in results:
            txt = r.get('text', '') or r.get('page_context', '')
            if txt:
                used_contexts.append(txt[:300])
            pid = r.get('page_id')
            if pid is not None:
                try:
                    citations.append({"page_id": max(0, int(pid) - 1)})  # normalize to 0-based
                except Exception:
                    pass

        # Deduplicate citations while preserving order
        seen = set()
        dedup_citations = []
        for c in citations:
            pid = c.get("page_id")
            if pid not in seen:
                dedup_citations.append(c)
                seen.add(pid)

        # Build final context string
        if used_contexts:
            context = "\n\n".join(used_contexts)
        else:
            # Fallback to concatenated page contexts
            context = "\n\n".join([(pc.get('page_context') or pc.get('text') or '') for pc in pcs])

        # Generate answer
        logger.debug("Generating answer...")
        answer = answer_question_from_context(context, req.question, req.model)
        logger.info(f"✅ Answer generated: {len(answer)} chars")
        
        return {"answer": answer, "citations": dedup_citations, "used_contexts": used_contexts}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Q&A failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
