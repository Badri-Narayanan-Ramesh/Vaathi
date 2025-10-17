"""
Interactive Debug Console for AI Tutor Backend API
Run this to test endpoints and see debug logs in real-time
"""
import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def print_response(response):
    """Pretty print response"""
    print(f"  Status: {response.status_code}")
    try:
        data = response.json()
        print(f"  Response: {json.dumps(data, indent=2)}")
    except:
        print(f"  Response: {response.text[:200]}")

def test_health():
    """Test health check"""
    print("\n" + "="*60)
    print("🏥 HEALTH CHECK")
    print("="*60)
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)

def test_root():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("🏠 ROOT ENDPOINT")
    print("="*60)
    response = requests.get(f"{BASE_URL}/")
    print_response(response)

def test_upload(pdf_path):
    """Test document upload"""
    print("\n" + "="*60)
    print(f"📤 UPLOAD DOCUMENT: {pdf_path}")
    print("="*60)
    
    if not Path(pdf_path).exists():
        print(f"  ❌ File not found: {pdf_path}")
        return None
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
        data = {'name': Path(pdf_path).stem}
        response = requests.post(f"{BASE_URL}/ingest/upload", files=files, data=data)
    
    print_response(response)
    
    if response.status_code == 200:
        return response.json().get('doc_id')
    return None

def test_list_pages(doc_id):
    """Test list pages"""
    print("\n" + "="*60)
    print(f"📄 LIST PAGES: {doc_id}")
    print("="*60)
    response = requests.get(f"{BASE_URL}/pages/{doc_id}/pages")
    print_response(response)

def test_explain_page(doc_id, page_id=0):
    """Test explain page"""
    print("\n" + "="*60)
    print(f"💡 EXPLAIN PAGE: doc_id={doc_id}, page={page_id}")
    print("="*60)
    response = requests.get(f"{BASE_URL}/pages/{doc_id}/pages/{page_id}/explain")
    print_response(response)

def test_qa(doc_id, question="What is this document about?"):
    """Test Q&A"""
    print("\n" + "="*60)
    print(f"❓ Q&A: {question}")
    print("="*60)
    data = {"question": question, "k": 3}
    response = requests.post(f"{BASE_URL}/qa/{doc_id}/qa", json=data)
    print_response(response)

def test_flashcards(doc_id, page_id=0):
    """Test flashcards generation"""
    print("\n" + "="*60)
    print(f"🎴 FLASHCARDS: doc_id={doc_id}, page={page_id}")
    print("="*60)
    response = requests.get(f"{BASE_URL}/study/{doc_id}/pages/{page_id}/flashcards")
    print_response(response)

def test_tts(text="Hello from AI Tutor"):
    """Test text-to-speech"""
    print("\n" + "="*60)
    print(f"🔊 TTS: {text}")
    print("="*60)
    response = requests.post(f"{BASE_URL}/media/tts", params={"text": text})
    print(f"  Status: {response.status_code}")
    print(f"  Content-Type: {response.headers.get('content-type')}")
    print(f"  Content-Length: {len(response.content)} bytes")

def interactive_menu():
    """Interactive menu"""
    doc_id = None
    
    while True:
        print("\n" + "="*60)
        print("🎮 AI TUTOR BACKEND DEBUG CONSOLE")
        print("="*60)
        print("1. Health Check")
        print("2. Root Endpoint")
        print("3. Upload PDF")
        print("4. List Pages")
        print("5. Explain Page")
        print("6. Ask Question")
        print("7. Generate Flashcards")
        print("8. Text-to-Speech")
        print("9. Run All Basic Tests")
        print("0. Exit")
        print("="*60)
        if doc_id:
            print(f"Current doc_id: {doc_id}")
        print("="*60)
        
        choice = input("\nSelect option: ").strip()
        
        try:
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                test_health()
            elif choice == "2":
                test_root()
            elif choice == "3":
                pdf_path = input("Enter PDF path (or press Enter for test_pdf.pdf): ").strip()
                if not pdf_path:
                    pdf_path = "test_pdf.pdf"
                doc_id = test_upload(pdf_path)
            elif choice == "4":
                if not doc_id:
                    print("❌ No document uploaded. Upload first (option 3)")
                else:
                    test_list_pages(doc_id)
            elif choice == "5":
                if not doc_id:
                    print("❌ No document uploaded. Upload first (option 3)")
                else:
                    page_id = input("Enter page ID (default 0): ").strip()
                    page_id = int(page_id) if page_id else 0
                    test_explain_page(doc_id, page_id)
            elif choice == "6":
                if not doc_id:
                    print("❌ No document uploaded. Upload first (option 3)")
                else:
                    question = input("Enter your question: ").strip()
                    if question:
                        test_qa(doc_id, question)
            elif choice == "7":
                if not doc_id:
                    print("❌ No document uploaded. Upload first (option 3)")
                else:
                    page_id = input("Enter page ID (default 0): ").strip()
                    page_id = int(page_id) if page_id else 0
                    test_flashcards(doc_id, page_id)
            elif choice == "8":
                text = input("Enter text to speak: ").strip()
                if text:
                    test_tts(text)
            elif choice == "9":
                test_health()
                test_root()
            else:
                print("❌ Invalid option")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🚀 AI Tutor Backend Debug Console")
    print(f"📡 Connecting to: {BASE_URL}")
    print("\nThis console will help you test the API and see debug logs")
    print("Make sure the backend is running with: uvicorn backend.main:app --reload --log-level debug")
    
    input("\nPress Enter to continue...")
    interactive_menu()
