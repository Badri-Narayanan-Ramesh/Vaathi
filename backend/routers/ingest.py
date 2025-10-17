from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.services.doc_store import DocStore
from backend.services.ai_adapter import ingest_pdf, build_index
from backend.models.schemas import UploadResp
from backend.utils.files import save_upload
import uuid
import logging

logger = logging.getLogger("backend.ingest")
router = APIRouter()

doc_store = DocStore()

@router.post("/upload", response_model=UploadResp)
async def upload(file: UploadFile = File(...), name: str = Form(None)):
    try:
        logger.info(f"📥 Upload request: filename={file.filename}, name={name}")
        
        # Save uploaded file
        logger.debug("Saving uploaded file...")
        path = save_upload(file)
        logger.info(f"✅ File saved to: {path}")
        
        # Ingest PDF
        logger.debug("Starting PDF ingestion...")
        page_contexts = ingest_pdf(path)
        logger.info(f"✅ PDF ingested: {len(page_contexts)} pages")
        
        # Build index
        logger.debug("Building vector index...")
        index = build_index(page_contexts)
        logger.info(f"✅ Vector index built")
        
        # Store document with PDF path
        doc_id = str(uuid.uuid4())
        doc_store.save(doc_id, page_contexts, index, name or file.filename, str(path))
        logger.info(f"✅ Document stored: doc_id={doc_id}")
        
        result = {"doc_id": doc_id, "name": name or file.filename, "page_count": len(page_contexts)}
        logger.info(f"📤 Upload complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
