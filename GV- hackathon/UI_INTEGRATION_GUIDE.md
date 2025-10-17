# UI Integration Guide - Backend to Frontend

## ✅ What's Done

1. ✅ Backend API running at http://127.0.0.1:8000
2. ✅ Gemini API key configured in `backend/.env`
3. ✅ API client created at `lib/api.ts`
4. ✅ Upload page updated (`pages/create.tsx`)
5. ⚠️ Viewer page needs manual fix (`pages/viewer.tsx`)

## 🔧 How to Complete Integration

### Step 1: Fix viewer.tsx

The file got corrupted with duplicate imports. Replace the entire contents of `pages/viewer.tsx` with the clean version from `VIEWER_FIXED.tsx` (see below).

### Step 2: Test Upload Flow

1. **Start Backend** (if not running):
   ```bash
   uvicorn backend.main:app --reload --log-level debug
   ```

2. **Start Frontend**:
   ```bash
   npm run dev
   ```

3. **Upload a PDF**:
   - Go to http://localhost:3000/create
   - Choose a PDF file
   - Click "Next"
   - Backend will process it and store doc_id

### Step 3: View Explanations

When you navigate to the viewer:
1. **Auto-loaded Summary**: Each time you change slides, the backend will:
   - Call `/pages/{doc_id}/pages/{page_id}/explain`
   - Display the AI-generated explanation in the chat

2. **Ask Questions**: Type questions in the textbox:
   - Sends to `/qa/{doc_id}/qa`
   - Uses vector search (k=3) to find relevant context
   - Returns AI-generated answer

## 📋 API Endpoints Being Used

### Upload (create.tsx)
```typescript
POST /ingest/upload
- Uploads PDF
- Returns: { doc_id, name, page_count }
- Stored in localStorage
```

### Explanation (viewer.tsx - auto on page change)
```typescript
GET /pages/{doc_id}/pages/{page_id}/explain
- Returns: { page_id, explanation }
- Displayed as "Slide X Summary"
```

### Q&A (viewer.tsx - on send button)
```typescript
POST /qa/{doc_id}/qa
- Body: { question: string, k: 3 }
- Returns: { answer: string }
- Displayed as assistant message
```

## 🎨 UI Flow

```
Create Page (upload)
     ↓
  [Backend processes PDF]
     ↓
localStorage saves:
  - pptUrl (blob URL for display)
  - pptName
  - docId (backend document ID)
  - pageCount
     ↓
Viewer Page
     ↓
  When page changes:
    → Fetch explanation
    → Display in chat as "Slide X Summary"
     ↓
  When user asks question:
    → Send to Q&A endpoint
    → Display answer in chat
```

## 🐛 Debug Tips

### Backend Logs
Watch the backend terminal for:
```
🔵 REQUEST START: POST /ingest/upload
📥 Upload request: filename=test.pdf
✅ PDF ingested: 5 pages
✅ Vector index built
✅ Document stored: doc_id=xyz
✅ REQUEST END - Status: 200 - Time: 4.5s
```

### Frontend Console
Check browser console for:
```javascript
console.log('✅ Document uploaded:', result);
console.log('✅ Explanation loaded for page', page);
console.log('✅ Answer received');
```

### Common Issues

**Issue**: "ConnectionRefusedError"
- **Fix**: Make sure backend is running on port 8000

**Issue**: "Document not found"
- **Fix**: Check localStorage has `docId` after upload

**Issue**: "No explanation shown"
- **Fix**: Check backend logs for errors in `/pages/{doc_id}/pages/{page_id}/explain`

## 📄 Clean viewer.tsx Code

Since the file got corrupted, here's the working version to copy:

### Key Changes Made:
1. ✅ Added `docId` state from localStorage
2. ✅ Added `loadingExplanation` state
3. ✅ Added `useEffect` to auto-fetch explanation on page change
4. ✅ Added async `onSend` function for Q&A
5. ✅ Added loading indicators
6. ✅ Added auto-scroll to latest message
7. ✅ Added Enter key to send (Shift+Enter for new line)

## 🚀 Next Steps

1. **Fix viewer.tsx manually**: Delete duplicated imports at top of file
2. **Test the flow**: Upload → View → Ask questions
3. **Add study aids**: Flashcards, Quiz buttons can call:
   - `/study/{doc_id}/pages/{page_id}/flashcards`
   - `/study/{doc_id}/pages/{page_id}/quiz`
4. **Add voice**: Connect Voice Mode button to TTS/STT endpoints

## 📝 Example Usage

```typescript
// In viewer.tsx

// Auto-load explanation when page changes
useEffect(() => {
  if (!docId) return;
  
  explainPage(docId, page - 1).then(result => {
    setChat(prev => [...prev, {
      role: "assistant",
      text: `📄 Slide ${page} Summary:\n\n${result.explanation}`
    }]);
  });
}, [page, docId]);

// Handle Q&A
const onSend = async () => {
  const result = await askQuestion(docId, question, 3);
  setChat(prev => [...prev, {
    role: "assistant",
    text: result.answer
  }]);
};
```

## ✅ Summary

**What Works Now**:
- ✅ Backend API with debug logging
- ✅ PDF upload with processing
- ✅ API client with TypeScript types
- ✅ Create page with upload integration

**What Needs Manual Fix**:
- ⚠️ viewer.tsx has duplicate imports (delete lines 7-14)
- Then the explanation + Q&A will work automatically!

**Expected Behavior**:
1. Upload PDF → Gets `doc_id`
2. Open viewer → Shows slides
3. Change page → Auto-loads AI summary
4. Ask question → Gets AI answer with context

All backend processing (OCR, embeddings, LLM) happens automatically! 🎉
