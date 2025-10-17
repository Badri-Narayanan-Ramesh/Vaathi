"""Quick test to verify PDF upload is working"""
import requests

print("Testing PDF upload...")
print("=" * 60)

try:
    # Check if test PDF exists
    import os
    if not os.path.exists("test_pdf.pdf"):
        print("❌ test_pdf.pdf not found")
        print("   Please create or specify a PDF file")
        exit(1)
    
    print("✅ Found test_pdf.pdf")
    
    # Upload
    with open("test_pdf.pdf", "rb") as f:
        files = {"file": ("test_pdf.pdf", f, "application/pdf")}
        data = {"name": "Test Document"}
        response = requests.post("http://127.0.0.1:8000/ingest/upload", files=files, data=data)
    
    print(f"\nStatus: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        print("\n✅ SUCCESS! PDF upload working")
    else:
        print(f"\n❌ FAILED with status {response.status_code}")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
