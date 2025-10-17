# Debug Console & Logging Guide

## 🔍 Debug Features Added

Your backend now has comprehensive debug logging and monitoring capabilities!

### 1. Enhanced Backend Logging

The backend (`backend/main.py`) now includes:

✅ **Request Logging Middleware**
- Logs every incoming request with method, path, headers, query params
- Tracks request processing time
- Color-coded emojis for easy identification

✅ **Global Exception Handler**
- Catches all unhandled exceptions
- Logs full traceback
- Returns detailed error responses

✅ **Startup/Shutdown Events**
- Logs when the server starts and stops
- Shows environment configuration
- Confirms router registration

✅ **Health Check Endpoints**
- `GET /health` - Simple health check
- `GET /` - Detailed service info

### 2. Router-Level Debug Logging

All routers now have detailed logging:

**Ingest Router** (`backend/routers/ingest.py`)
- 📥 File upload tracking
- ✅ PDF ingestion progress
- 📤 Document storage confirmation

**Pages Router** (`backend/routers/pages.py`)
- 📄 Page listing
- 💡 Explanation generation with timing

**Q&A Router** (`backend/routers/qa.py`)
- ❓ Question tracking
- 🔍 Vector search results
- 💬 Answer generation

### 3. Debug Console

**Interactive Tool**: `debug_console.py`

Run it with:
```bash
python debug_console.py
```

Features:
- 🏥 Health checks
- 📤 Upload documents
- 📄 List pages
- 💡 Test explanations
- ❓ Ask questions
- 🎴 Generate flashcards
- 🔊 Test TTS
- Interactive menu for easy testing

## 🚀 How to Use

### Step 1: Start Backend with Debug Logging

```bash
uvicorn backend.main:app --reload --log-level debug
```

You'll see:
```
2025-10-16 16:31:49,924 - backend - INFO - 🚀 AI Tutor Backend API starting up...
2025-10-16 16:31:49,924 - backend - INFO -    Environment: development
2025-10-16 16:31:49,925 - backend - INFO -    CORS Origins: ['http://localhost:3000']
2025-10-16 16:31:49,925 - backend - INFO -    Routers registered: ingest, pages, qa, study_aids, media
INFO:     Application startup complete.
```

### Step 2: Open Debug Console (Optional)

In a new terminal:
```bash
python debug_console.py
```

### Step 3: Make Requests

Watch the backend terminal for detailed logs:

**Example Request Log:**
```
2025-10-16 16:35:12,345 - backend - INFO - 🔵 REQUEST START: POST /ingest/upload
2025-10-16 16:35:12,346 - backend.ingest - INFO - 📥 Upload request: filename=test.pdf, name=Test Doc
2025-10-16 16:35:12,347 - backend.ingest - INFO - ✅ File saved to: ./uploads/abc123.pdf
2025-10-16 16:35:14,123 - backend.ingest - INFO - ✅ PDF ingested: 5 pages
2025-10-16 16:35:16,789 - backend.ingest - INFO - ✅ Vector index built
2025-10-16 16:35:16,790 - backend.ingest - INFO - ✅ Document stored: doc_id=xyz789
2025-10-16 16:35:16,791 - backend - INFO - ✅ REQUEST END: POST /ingest/upload - Status: 200 - Time: 4.446s
```

## 📊 Log Levels & Emojis

| Emoji | Level | Meaning |
|-------|-------|---------|
| 🔵 | INFO | Request started |
| ✅ | INFO | Success/Completion |
| 📥 | INFO | Incoming data |
| 📤 | INFO | Outgoing data |
| 💡 | INFO | Processing/Generation |
| ❓ | INFO | Question/Query |
| 🔍 | DEBUG | Search/Lookup |
| ⚠️ | WARNING | Warning/Not found |
| ❌ | ERROR | Error occurred |
| 🔥 | ERROR | Unhandled exception |

## 🛠️ Troubleshooting with Debug Console

### Issue: Can't connect to backend
```bash
# Check if backend is running
python debug_console.py
# Select option 1 (Health Check)
```

### Issue: Upload fails
```bash
# Start debug console
python debug_console.py
# Select option 3 (Upload PDF)
# Watch backend terminal for detailed error logs
```

### Issue: LLM not responding
```bash
# Backend terminal will show:
# - API call attempts
# - Response times
# - Error messages from LLM provider
```

### Issue: Vector search problems
```bash
# Q&A logs will show:
# - Query text
# - Number of results found
# - Chunks retrieved
```

## 📋 Example Debug Session

**Terminal 1 (Backend):**
```bash
PS C:\Users\Vivin\GV- hackathon> uvicorn backend.main:app --reload --log-level debug
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
2025-10-16 16:31:49,924 - backend - INFO - 🚀 AI Tutor Backend API starting up...
```

**Terminal 2 (Debug Console):**
```bash
PS C:\Users\Vivin\GV- hackathon> python debug_console.py
🚀 AI Tutor Backend Debug Console
📡 Connecting to: http://127.0.0.1:8000

🎮 AI TUTOR BACKEND DEBUG CONSOLE
1. Health Check
2. Root Endpoint
3. Upload PDF
...

Select option: 1
```

**Backend Terminal Shows:**
```
2025-10-16 16:35:12,345 - backend - INFO - 🔵 REQUEST START: GET /health
2025-10-16 16:35:12,346 - backend - DEBUG - Health check endpoint called
2025-10-16 16:35:12,347 - backend - INFO - ✅ REQUEST END: GET /health - Status: 200 - Time: 0.002s
```

## 🔧 Advanced Debugging

### Enable More Verbose Logging

Edit `backend/main.py` and change:
```python
logging.basicConfig(level=logging.DEBUG)  # Already set
```

### Log to File

Add to `backend/main.py`:
```python
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend_debug.log'),
        logging.StreamHandler()
    ]
)
```

### Custom Logger for Specific Module

```python
import logging
logger = logging.getLogger("backend.custom")
logger.setLevel(logging.DEBUG)
```

## 📚 API Documentation

While debugging, you can also use:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

These show all endpoints and let you test them interactively!

## 🎯 Next Steps

1. ✅ Backend is running with debug logging
2. ✅ Debug console ready for testing
3. ✅ All requests are logged with emojis
4. ✅ Errors show full tracebacks

Now you can:
- Test endpoints using debug console
- Monitor logs in real-time
- Troubleshoot issues with detailed traces
- See exactly what's happening in the backend

Happy debugging! 🐛🔍
