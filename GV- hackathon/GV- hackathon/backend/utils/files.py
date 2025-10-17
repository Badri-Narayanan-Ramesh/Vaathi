# File utilities for backend
import os
from fastapi import UploadFile
from pathlib import Path
import uuid
import logging

logger = logging.getLogger("backend.files")

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

def save_upload(file: UploadFile) -> str:
    """Save an uploaded file and return its path."""
    try:
        file_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix
        filename = f"{file_id}{ext}"
        filepath = UPLOAD_DIR / filename
        
        with open(filepath, "wb") as f:
            content = file.file.read()
            f.write(content)
        
        logger.info(f"Saved upload to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving upload: {e}")
        raise
