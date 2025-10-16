# AI Tutor Backbone (Module Only)

This module provides the core AI backbone for an AI Tutor: PDF ingestion, embeddings + retrieval,
LLM chains (Explain, Answer w/ Citations, Flashcards, Quiz, Cheatsheet), and simple TTS/STT.
No UI, no collaboration code.

## Features
- Cloud LLM: OpenAI gpt-4o-mini (assume free tier)
- Local LLM: Ollama `phi3:mini` (CPU)
- PDF per-page ingestion: PyMuPDF text + EasyOCR for raster/handwritten + BLIP captions for diagrams
- Merge to page_context, embed with `sentence-transformers/all-MiniLM-L6-v2`, store in Chroma
- Chains: ExplainPage, AnswerWithCitations (retrieval Top-k), Flashcards, Quiz, Cheatsheet
- Local TTS (pyttsx3) and local STT (Whisper small or Vosk) with cloud stubs
- In-memory caching by (fn_name, prompt_hash)

## Install

Python 3.10+ recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ai_module/requirements.txt
```

On Windows, `pdf2image` may require Poppler. Install Poppler and set POPPLER_PATH in env or
ensure `pdf2image` can locate it.

## Configure
Copy `.env.example` to `.env` (or set environment variables).

- `APP_MODE`: `cloud` or `local`
- `OPENAI_API_KEY`: required for cloud mode
- `OLLAMA_MODEL`: default `phi3:mini`
- `INDEX_DIR`: where to store Chroma index

## Usage

1) Ingest a PDF and build an index:

```python
from ai_module.core.ingest import load_pdf
from ai_module.core.embeddings import build_index, query_index
from ai_module.core.retriever import retrieve_for_question
from ai_module.core.chains import explain_page, answer_question, make_flashcards, make_quiz, make_cheatsheet

pages = load_pdf("/path/to/notes.pdf")
index = build_index(pages, index_dir="./data/index")

# Explain a page
print(explain_page(pages[0]["page_context"]))

# Answer with citations
hits = retrieve_for_question(index, "What is backpropagation?", k=3)
ctx_texts = [h["text"] for h in hits]
print(answer_question("What is backpropagation?", ctx_texts))

# Study aids
print(make_flashcards(pages[0]["page_context"]))
print(make_quiz(pages[0]["page_context"]))
print(make_cheatsheet(pages[0]["page_context"]))
```

2) TTS/STT:

```python
from ai_module.core.tts import speak_local
from ai_module.core.stt import transcribe_local

wav = speak_local("Hello world", out_path="./out/hello.wav")
print("Saved:", wav)

text = transcribe_local(wav)
print("Transcript:", text)
```

## Switching cloud ↔ local

Set `APP_MODE=cloud` for OpenAI or `APP_MODE=local` for Ollama. For local, ensure Ollama is running and
`phi3:mini` is pulled.

## Notes
- This is the AI module only; UI and Study-Jam will integrate later via clean callable functions.
- Chains are intentionally lightweight (no agents) and cache results in-memory.
- Tests use mocks; no external calls required.
