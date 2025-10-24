#!/usr/bin/env python3
"""
Simple test script to verify the system is working
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all modules can be imported"""
    try:
        print("Testing imports...")
        
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from library_vectorizer import LibraryVectorizer
        print("✅ LibraryVectorizer imported")
        
        from rag_pipeline import SimpleRAG
        print("✅ SimpleRAG imported")
        
        from config import get_config
        print("✅ Config imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_qdrant_connection():
    """Test Qdrant connection"""
    try:
        print("\nTesting Qdrant connection...")
        
        # Set environment variables
        os.environ["QDRANT_HOST"] = "192.168.1.13"
        os.environ["QDRANT_PORT"] = "6333"
        os.environ["QDRANT_API_KEY"] = "4c77aa13-30ff-41e0-ac87-ab1e0c791d2e"
        
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            host="192.168.1.13",
            port=6333,
            api_key="4c77aa13-30ff-41e0-ac87-ab1e0c791d2e",
            https=False
        )
        
        # Test connection
        collections = client.get_collections()
        print(f"✅ Connected to Qdrant. Found {len(collections.collections)} collections")
        
        return True
    except Exception as e:
        print(f"❌ Qdrant connection error: {e}")
        return False

def test_documents():
    """Test if documents exist"""
    try:
        print("\nTesting documents...")
        
        docs_path = Path("documents")
        if not docs_path.exists():
            print("❌ Documents directory not found")
            return False
        
        doc_files = list(docs_path.glob("*"))
        if not doc_files:
            print("❌ No documents found")
            return False
        
        print(f"✅ Found {len(doc_files)} documents:")
        for doc in doc_files:
            print(f"   - {doc.name}")
        
        return True
    except Exception as e:
        print(f"❌ Documents test error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Library Vectorization RAG System - System Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Qdrant Connection", test_qdrant_connection),
        ("Documents", test_documents)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Add more documents to ./documents/")
        print("2. Run: python -m src.cli vectorize ./documents --vector-store qdrant --collection-name market_documents")
        print("3. Run: python -m src.cli query 'Your question' --vector-store qdrant --collection-name market_documents")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
