# AI Tutor Backend API

FastAPI backend wrapper for the AI Tutor core module. Provides RESTful endpoints for PDF ingestion, LLM-powered study aids, Q&A, TTS/STT, and more.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
- `GEMINI_API_KEY`: Your Google Gemini API key (required for cloud LLM)
- `OLLAMA_BASE_URL`: Base URL for local Ollama server (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model to use with Ollama (default: llama2)

### 3. Run the Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## API Endpoints

### Ingest
- `POST /ingest/upload` - Upload PDF/PPT document
  - Request: multipart/form-data with `file` and optional `name`
  - Response: `{ doc_id, name, page_count }`

### Pages
- `GET /pages/{doc_id}/pages` - List all pages
- `GET /pages/{doc_id}/pages/{page_id}/explain` - Get detailed explanation for a page

### Q&A
- `POST /qa/{doc_id}/qa` - Ask a question about the document
  - Request body: `{ question: string, k?: number }`
  - Response: `{ answer: string }`

### Study Aids
- `GET /study/{doc_id}/pages/{page_id}/flashcards` - Generate flashcards
- `GET /study/{doc_id}/pages/{page_id}/quiz` - Generate quiz questions
- `GET /study/{doc_id}/pages/{page_id}/cheatsheet` - Generate cheatsheet

### Media
- `POST /media/tts?text=<text>` - Convert text to speech (returns audio file)
- `POST /media/stt` - Convert speech to text
  - Request: multipart/form-data with `audio` file
  - Response: `{ text: string }`

## API Documentation

Once running, visit:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## File Structure

```
backend/
├── main.py                 # FastAPI app & router registration
├── routers/
│   ├── ingest.py          # Document upload endpoints
│   ├── pages.py           # Page listing & explanation
│   ├── qa.py              # Question answering
│   ├── study_aids.py      # Flashcards, quiz, cheatsheet
│   └── media.py           # TTS/STT endpoints
├── services/
│   ├── doc_store.py       # In-memory document storage
│   ├── ai_adapter.py      # Wrapper for ai_core module
│   └── user_service.py    # User/session management (placeholder)
├── models/
│   └── schemas.py         # Pydantic request/response models
├── utils/
│   ├── common.py          # Logging & utilities
│   └── files.py           # File handling utilities
├── requirements.txt       # Python dependencies
└── .env.example           # Environment template
```

## TypeScript Client

A fully-typed TypeScript client is available at `web/lib/api.ts` for Next.js integration:

```typescript
import { AITutorClient, uploadDocument, askQuestion } from '@/lib/api';

// Upload a document
const result = await uploadDocument(file, 'My Document');
console.log(result.doc_id, result.page_count);

// Ask a question
const qa = await askQuestion(result.doc_id, { 
  question: 'What is the main topic?',
  k: 3 
});
console.log(qa.answer);
```

## Development

The server runs with auto-reload enabled. Edit any file and the server will restart automatically.

To run in production:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Notes

- Uploaded files are stored in `./uploads/`
- Generated audio is stored in `./audio/`
- Document storage is in-memory by default (extend `DocStore` for persistence)
- All AI processing is handled by the `ai_core` module
