"""
Quick Start Script for Qdrant Configuration

This script sets up the environment variables for Qdrant and provides
quick commands for vectorizing documents and using the RAG system.
"""

import os
import subprocess
import sys
from pathlib import Path

def set_qdrant_env():
    """Set Qdrant environment variables"""
    os.environ["QDRANT_HOST"] = "192.168.1.13"
    os.environ["QDRANT_PORT"] = "6333"
    os.environ["QDRANT_API_KEY"] = "4c77aa13-30ff-41e0-ac87-ab1e0c791d2e"
    os.environ["QDRANT_COLLECTION"] = "market_documents"
    os.environ["VECTOR_STORE_TYPE"] = "qdrant"
    
    print("✅ Qdrant environment variables set:")
    print(f"   Host: {os.environ['QDRANT_HOST']}")
    print(f"   Port: {os.environ['QDRANT_PORT']}")
    print(f"   Collection: {os.environ['QDRANT_COLLECTION']}")
    print(f"   API Key: {os.environ['QDRANT_API_KEY'][:8]}...")

def vectorize_documents(documents_path="./documents"):
    """Vectorize documents in the specified path"""
    print(f"\n📚 Vectorizing documents from: {documents_path}")
    
    cmd = [
        sys.executable, "-m", "src.cli", "vectorize", documents_path,
        "--vector-store", "qdrant",
        "--collection-name", "market_documents"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Documents vectorized successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error vectorizing documents: {e}")
        return False

def test_query(question="What is machine learning?"):
    """Test a single query"""
    print(f"\n❓ Testing query: {question}")
    
    cmd = [
        sys.executable, "-m", "src.cli", "query", question,
        "--vector-store", "qdrant",
        "--collection-name", "market_documents"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Query test completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error testing query: {e}")
        return False

def show_stats():
    """Show collection statistics"""
    print("\n📊 Collection Statistics:")
    
    cmd = [
        sys.executable, "-m", "src.cli", "stats",
        "--vector-store", "qdrant",
        "--collection-name", "market_documents"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting stats: {e}")
        return False

def interactive_qa():
    """Start interactive Q&A session"""
    print("\n💬 Starting interactive Q&A session...")
    print("Type 'quit' or 'exit' to end the session.")
    
    cmd = [
        sys.executable, "-m", "src.cli", "ask",
        "--vector-store", "qdrant",
        "--collection-name", "market_documents"
    ]
    
    try:
        result = subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\n👋 Session ended by user.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting interactive session: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Library Vectorization RAG System - Qdrant Quick Start")
    print("=" * 60)
    
    # Set environment variables
    set_qdrant_env()
    
    # Check if documents directory exists
    documents_path = Path("./documents")
    if not documents_path.exists():
        print(f"\n⚠️  Documents directory not found: {documents_path}")
        print("Please create the directory and add your documents.")
        return
    
    # Check if documents exist
    doc_files = list(documents_path.glob("*"))
    if not doc_files:
        print(f"\n⚠️  No documents found in: {documents_path}")
        print("Please add your documents (PDF, DOCX, TXT, MD) to the directory.")
        return
    
    print(f"\n📁 Found {len(doc_files)} documents in {documents_path}")
    
    # Show menu
    while True:
        print("\n" + "=" * 60)
        print("Choose an option:")
        print("1. Vectorize documents")
        print("2. Test single query")
        print("3. Show collection stats")
        print("4. Start interactive Q&A")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            vectorize_documents()
        elif choice == "2":
            question = input("Enter your question: ").strip()
            if question:
                test_query(question)
            else:
                test_query()
        elif choice == "3":
            show_stats()
        elif choice == "4":
            interactive_qa()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
