"""
Test script to verify backend API with debug logging
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_root():
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_docs():
    print("\n=== Testing Docs Endpoint ===")
    response = requests.get(f"{BASE_URL}/docs")
    print(f"Status: {response.status_code}")
    print(f"Response type: {response.headers.get('content-type')}")

if __name__ == "__main__":
    print("🧪 Testing AI Tutor Backend API")
    print(f"Base URL: {BASE_URL}")
    
    try:
        test_health()
        test_root()
        test_docs()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
