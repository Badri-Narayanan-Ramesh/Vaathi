# AI Adapter Service - wraps ai_core module for backend
# Provides clean interface for PDF ingestion, embeddings, LLM chains, TTS/STT

from ai_core.ingest import load_pdf
from ai_core.embeddings import build_index as build_chroma_index, query_index as query_chroma_index
from ai_core.chains import explain_page, answer_question, make_flashcards, make_quiz, make_cheatsheet
from ai_core.tts import speak_local, speak_cloud
from ai_core.stt import transcribe_local, transcribe_cloud
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger("backend.ai_adapter")

# Configuration
USE_CLOUD_TTS = os.getenv("USE_CLOUD_TTS", "false").lower() == "true"
USE_CLOUD_STT = os.getenv("USE_CLOUD_STT", "false").lower() == "true"
INDEX_DIR = os.getenv("VECTOR_STORE_DIR", "./data/chroma")

def ingest_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Ingest a PDF and return page contexts with text and images."""
    try:
        logger.info(f"Ingesting PDF: {pdf_path}")
        page_contexts = load_pdf(pdf_path)
        logger.info(f"PDF ingested successfully: {len(page_contexts)} pages")
        return page_contexts
    except Exception as e:
        logger.error(f"Error ingesting PDF: {e}")
        logger.exception(e)
        raise

def build_index(page_contexts: List[Dict[str, Any]]) -> Any:
    """Build vector index from page contexts."""
    try:
        logger.info(f"Building vector index for {len(page_contexts)} pages")
        index = build_chroma_index(page_contexts, INDEX_DIR)
        logger.info("Vector index built successfully")
        return index
    except Exception as e:
        logger.error(f"Error building index: {e}")
        logger.exception(e)
        raise

def query_index(index: Any, query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Query the vector index and return top k results."""
    try:
        logger.debug(f"Querying index: query='{query}', k={k}")
        results = query_chroma_index(index, query, k=k)
        logger.debug(f"Found {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error querying index: {e}")
        logger.exception(e)
        raise

def generate_explanation(page_text: str, model: Optional[str] = None) -> str:
    """Generate detailed explanation for a page."""
    try:
        logger.debug(f"Generating explanation for {len(page_text)} chars, model={model}")
        # Temporarily override model/app mode for this call if requested
        prev = {
            "APP_MODE": os.getenv("APP_MODE"),
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
        }
        try:
            if model:
                if ":" in model or model.lower() in {"llama3", "llama3.1", "llama3.2", "phi3", "phi3:mini"}:
                    os.environ["APP_MODE"] = "local"
                    os.environ["OLLAMA_MODEL"] = model
                else:
                    os.environ["APP_MODE"] = "cloud"
                    os.environ["GEMINI_MODEL"] = model
            result = explain_page(page_text)
        finally:
            # Restore env
            for k, v in prev.items():
                if v is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = v
        logger.debug(f"Explanation generated: {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        logger.exception(e)
        raise

def answer_question_from_context(context: str, question: str, model: Optional[str] = None) -> str:
    """Answer a question based on context, with optional model override (Gemini or Ollama)."""
    try:
        logger.debug(f"Answering question: '{question}' model={model}")
        # Temporarily override model/app mode for this call if requested
        prev = {
            "APP_MODE": os.getenv("APP_MODE"),
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
        }
        try:
            if model:
                if ":" in model or model.lower() in {"llama3", "llama3.1", "llama3.2", "phi3", "phi3:mini"}:
                    os.environ["APP_MODE"] = "local"
                    os.environ["OLLAMA_MODEL"] = model
                else:
                    os.environ["APP_MODE"] = "cloud"
                    os.environ["GEMINI_MODEL"] = model

            # answer_question from chains expects question first, then context list
            result = answer_question(question, [context])
        finally:
            # Restore env
            for k, v in prev.items():
                if v is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = v
        logger.debug(f"Answer generated: {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        logger.exception(e)
        raise

def generate_flashcards(page_text: str, model: Optional[str] = None) -> List[Dict]:
    """Generate flashcards from page text."""
    try:
        logger.debug(f"Generating flashcards for {len(page_text)} chars, model={model}")
        # Temporarily override model/app mode for this call if requested
        prev = {
            "APP_MODE": os.getenv("APP_MODE"),
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
        }
        try:
            if model:
                if ":" in model or model.lower() in {"llama3", "llama3.1", "llama3.2", "phi3", "phi3:mini"}:
                    os.environ["APP_MODE"] = "local"
                    os.environ["OLLAMA_MODEL"] = model
                else:
                    os.environ["APP_MODE"] = "cloud"
                    os.environ["GEMINI_MODEL"] = model
            result = make_flashcards(page_text)
        finally:
            # Restore env
            for k, v in prev.items():
                if v is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = v
        logger.debug(f"Generated {len(result)} flashcards")
        return result
    except Exception as e:
        logger.error(f"Error generating flashcards: {e}")
        logger.exception(e)
        raise

def generate_quiz(page_text: str, model: Optional[str] = None) -> List[Dict]:
    """Generate quiz questions from page text."""
    try:
        logger.debug(f"Generating quiz for {len(page_text)} chars, model={model}")
        # Temporarily override model/app mode for this call if requested
        prev = {
            "APP_MODE": os.getenv("APP_MODE"),
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
        }
        try:
            if model:
                if ":" in model or model.lower() in {"llama3", "llama3.1", "llama3.2", "phi3", "phi3:mini"}:
                    os.environ["APP_MODE"] = "local"
                    os.environ["OLLAMA_MODEL"] = model
                else:
                    os.environ["APP_MODE"] = "cloud"
                    os.environ["GEMINI_MODEL"] = model
            result = make_quiz(page_text)
        finally:
            # Restore env
            for k, v in prev.items():
                if v is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = v
        logger.debug(f"Generated {len(result)} quiz questions")
        return result
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        logger.exception(e)
        raise

def generate_cheatsheet(page_text: str, model: Optional[str] = None) -> str:
    """Generate cheatsheet from page text."""
    try:
        logger.debug(f"Generating cheatsheet for {len(page_text)} chars, model={model}")
        # Temporarily override model/app mode for this call if requested
        prev = {
            "APP_MODE": os.getenv("APP_MODE"),
            "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
            "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL"),
        }
        try:
            if model:
                if ":" in model or model.lower() in {"llama3", "llama3.1", "llama3.2", "phi3", "phi3:mini"}:
                    os.environ["APP_MODE"] = "local"
                    os.environ["OLLAMA_MODEL"] = model
                else:
                    os.environ["APP_MODE"] = "cloud"
                    os.environ["GEMINI_MODEL"] = model
            result = make_cheatsheet(page_text)
        finally:
            # Restore env
            for k, v in prev.items():
                if v is None:
                    if k in os.environ:
                        del os.environ[k]
                else:
                    os.environ[k] = v
        logger.debug(f"Cheatsheet generated: {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"Error generating cheatsheet: {e}")
        logger.exception(e)
        raise

def text_to_speech(text: str, output_path: str) -> str:
    """Convert text to speech and save to file."""
    try:
        logger.debug(f"Converting text to speech: {len(text)} chars")
        if USE_CLOUD_TTS:
            result = speak_cloud(text, output_path)
        else:
            result = speak_local(text, output_path)
        logger.debug(f"Audio saved to: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in TTS: {e}")
        logger.exception(e)
        raise

def speech_to_text(audio_path: str) -> str:
    """Convert speech to text."""
    try:
        logger.debug(f"Transcribing audio: {audio_path}")
        if USE_CLOUD_STT:
            result = transcribe_cloud(audio_path)
        else:
            result = transcribe_local(audio_path)
        logger.debug(f"Transcribed: {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"Error in STT: {e}")
        logger.exception(e)
        raise
