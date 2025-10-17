# Utility functions for backend (logging, file ops, etc.)
import logging
import os
from typing import Optional

LOG_FORMAT = "% (asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("backend")

def save_file(file, path: str):
    with open(path, "wb") as f:
        f.write(file)

def get_env_var(key: str, default: Optional[str] = None) -> str:
    return os.environ.get(key, default)
