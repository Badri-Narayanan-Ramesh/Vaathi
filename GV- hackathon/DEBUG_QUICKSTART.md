# 🎯 Debug Console & Monitoring - Quick Start

## ✅ What Was Added

Your AI Tutor backend now has **comprehensive debug logging and monitoring**!

### 🔍 Debug Features

1. **Request/Response Logging** - Every API call is tracked with:
   - 🔵 Request start (method, path, headers, params)
   - ✅ Request completion (status, processing time)
   - ❌ Errors with full tracebacks

2. **Emoji-Coded Logs** - Easy visual scanning:
   - 📥 Incoming data
   - 📤 Outgoing data  
   - 💡 Processing steps
   - ⚠️ Warnings
   - 🔥 Critical errors

3. **Interactive Debug Console** - Test tool with menu-driven interface

## 🚀 Quick Start

### Backend is Already Running!

Your backend is currently running at: **http://127.0.0.1:8000**

You can see debug logs in the terminal showing:
```
🚀 AI Tutor Backend API starting up...
   Environment: development
   CORS Origins: ['http://localhost:3000']
   Routers registered: ingest, pages, qa, study_aids, media
```

### View API Documentation

Open in browser (or click the Simple Browser tab):
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Test with Debug Console

Open a **new terminal** and run:
```bash
python debug_console.py
```

This gives you an interactive menu to:
- ✅ Test health endpoints
- 📤 Upload PDFs
- 💡 Test explanations
- ❓ Ask questions
- 🎴 Generate flashcards
- 🔊 Test TTS

**While testing, watch your backend terminal for detailed logs!**

## 📊 Example Debug Output

When you make a request, you'll see:

```
2025-10-16 16:35:12,345 - backend - INFO - 🔵 REQUEST START: POST /ingest/upload
2025-10-16 16:35:12,346 - backend.ingest - INFO - 📥 Upload request: filename=test.pdf
2025-10-16 16:35:12,347 - backend.ingest - INFO - ✅ File saved to: ./uploads/abc123.pdf
2025-10-16 16:35:14,123 - backend.ingest - INFO - ✅ PDF ingested: 5 pages
2025-10-16 16:35:16,789 - backend.ingest - INFO - ✅ Vector index built
2025-10-16 16:35:16,790 - backend.ingest - INFO - ✅ Document stored: doc_id=xyz789
2025-10-16 16:35:16,791 - backend - INFO - ✅ REQUEST END: POST /ingest/upload - Status: 200 - Time: 4.446s
```

## 🛠️ What's Being Tracked

### Upload Endpoint
- File receipt and validation
- PDF ingestion progress (pages extracted)
- Vector index building
- Document storage confirmation
- Total processing time

### Pages Endpoint
- Document retrieval
- Page listing
- Explanation generation timing

### Q&A Endpoint
- Question received
- Vector search results (k chunks found)
- Answer generation
- Response size

### All Endpoints
- HTTP method and path
- Request headers and params
- Response status codes
- Processing time in seconds
- Full error tracebacks on failure

## 📁 Files Created

```
backend/
├── main.py              ← Enhanced with logging middleware
├── routers/
│   ├── ingest.py       ← Debug logs added
│   ├── pages.py        ← Debug logs added
│   ├── qa.py           ← Debug logs added
│   └── ...
└── test_api.py         ← Basic API tests

debug_console.py        ← Interactive test console
DEBUG_GUIDE.md          ← Full debug guide
DEBUG_QUICKSTART.md     ← This file
```

## 🔥 Common Issues & Debug Info

### Issue: "Cannot connect to backend"
**Look for**: Startup logs showing `🚀 AI Tutor Backend API starting up...`

### Issue: "Upload fails"
**Look for**: 
- `📥 Upload request` - confirms file received
- `❌` emojis showing where it failed
- Full traceback with error details

### Issue: "LLM timeout"
**Look for**:
- API call attempts in logs
- Response time tracking
- Error messages from Gemini/Ollama

### Issue: "No results from search"
**Look for**:
- `🔍` Vector search logs
- Number of chunks retrieved
- Query text being used

## 🎮 Try It Now!

### Option 1: Use Browser
Open: http://127.0.0.1:8000/docs
Click "Try it out" on any endpoint

### Option 2: Use Debug Console
```bash
python debug_console.py
```
Select options from menu

### Option 3: Use curl/Postman
Make requests and watch backend logs

## 📚 More Details

See `DEBUG_GUIDE.md` for:
- Complete logging reference
- Advanced debugging techniques
- Log file configuration
- Custom logger setup

---

**Backend Status**: ✅ Running with debug logging at http://127.0.0.1:8000

**API Docs**: http://127.0.0.1:8000/docs

**Debug Console**: Run `python debug_console.py` in new terminal

Happy debugging! 🐛🔍✨
