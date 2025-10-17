# AI Tutor - Complete System

A production-ready AI-powered tutoring system with PDF ingestion, LLM-based explanations, Q&A, study aids generation, and voice capabilities.

## Project Structure

```
.
├── ai_core/              # Core AI module (PDF processing, LLM chains, embeddings, TTS/STT)
├── backend/              # FastAPI REST API wrapper
└── web/                  # TypeScript client for Next.js integration
```

## Quick Start

### 1. Install Core Dependencies

```bash
pip install -r ai_core/requirements.txt
```

### 2. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Configure Environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and add your API keys
```

### 4. Run Backend Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## Components

### AI Core (`ai_core/`)

The backbone AI module providing:
- **PDF Ingestion**: PyMuPDF + OCR + image captioning
- **Embeddings**: Sentence transformers + ChromaDB vector store
- **LLM Chains**: Explanations, Q&A, flashcards, quiz, cheatsheet generation
- **TTS**: Text-to-speech with pyttsx3
- **STT**: Speech-to-text with Whisper/Vosk
- **Multi-provider support**: Gemini (cloud) and Ollama (local)

See `ai_core/README.md` for details.

### Backend (`backend/`)

FastAPI REST API wrapping the core module with:
- Document upload & management
- Page-level explanations
- Q&A with vector search
- Study aids generation (flashcards, quiz, cheatsheet)
- TTS/STT endpoints
- Full CORS support

See `backend/README.md` for API documentation.

### TypeScript Client (`web/lib/api.ts`)

Fully-typed client for Next.js with:
- All API endpoints covered
- Type-safe request/response models
- Environment-based configuration
- Singleton and functional API

## Features

✅ **PDF Ingestion**: Extract text, images, and structure from PDFs  
✅ **Smart OCR**: Fallback OCR for scanned documents  
✅ **Image Captioning**: BLIP-based visual understanding  
✅ **Vector Search**: Semantic search with ChromaDB  
✅ **LLM Chains**: Generate explanations, answers, flashcards, quizzes, cheatsheets  
✅ **Voice I/O**: Text-to-speech and speech-to-text  
✅ **Multi-LLM**: Gemini (cloud) or Ollama (local)  
✅ **REST API**: FastAPI with auto-docs  
✅ **TypeScript Client**: Type-safe Next.js integration  

## API Usage Examples

### Python

```python
from ai_core import ingest, chains

# Ingest PDF
pages = ingest.process_pdf("document.pdf")

# Generate explanation
explanation = chains.explain_page(pages[0]['text'])

# Answer question
answer = chains.answer_question(pages[0]['text'], "What is this about?")
```

### TypeScript (Next.js)

```typescript
import { uploadDocument, askQuestion, getFlashcards } from '@/lib/api';

// Upload document
const doc = await uploadDocument(file);

// Ask question
const qa = await askQuestion(doc.doc_id, { 
  question: "Explain the key concepts",
  k: 3 
});

// Get flashcards
const cards = await getFlashcards(doc.doc_id, 0);
```

### cURL

```bash
# Upload document
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@document.pdf" \
  -F "name=My Document"

# Ask question
curl -X POST http://localhost:8000/qa/{doc_id}/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "k": 3}'
```

## Configuration

### Environment Variables

Create `backend/.env`:

```env
# LLM Provider (cloud or local)
LLM_MODE=cloud
CLOUD_PROVIDER=gemini
CLOUD_MODEL=gemini-1.5-flash

# API Keys
GEMINI_API_KEY=your_api_key_here

# Ollama (for local mode)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Vector Store
VECTOR_STORE_DIR=./data/chroma
```

## Development

All modules support hot-reload:

```bash
# Backend (auto-reload)
uvicorn backend.main:app --reload

# Test core module directly
python ai_core/demo.py --pdf test.pdf
```

## Testing

```bash
# Test core AI module
pytest ai_core/tests/

# Test backend API
pytest backend/tests/
```

## Production Deployment

```bash
# Run backend in production
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Consider:
- Use Gunicorn/Uvicorn workers for scalability
- Set up Redis for persistent document storage
- Configure CORS for your frontend domain
- Use environment-based config management
- Set up logging and monitoring

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

For detailed documentation:
- **AI Core**: See `ai_core/README.md`
- **Backend API**: See `backend/README.md` or visit `http://localhost:8000/docs`
- **TypeScript Client**: See inline JSDoc in `web/lib/api.ts`
