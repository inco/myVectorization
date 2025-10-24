"""
Library Vectorization RAG System - Usage Instructions

This file provides detailed instructions for using the system.
"""

# USAGE INSTRUCTIONS

"""
1. QUICK START (Recommended for first-time users)
   Run: python quickstart.py
   This will guide you through the entire setup process.

2. MANUAL SETUP
   
   Step 1: Install dependencies
   pip install -r requirements.txt
   
   Step 2: Start a vector store (choose one)
   # Milvus (recommended)
   docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest
   
   # Qdrant (alternative)
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
   
   # ChromaDB (easiest, no Docker needed)
   # No setup required
   
   Step 3: Add documents
   Place your documents (PDF, DOCX, TXT, MD) in the 'documents' directory
   
   Step 4: Vectorize documents
   python -m src.cli vectorize ./documents --vector-store milvus
   
   Step 5: Start using the system
   python -m src.cli ask

3. CLI COMMANDS

   Vectorize documents:
   python -m src.cli vectorize <path> [options]
   
   Interactive Q&A:
   python -m src.cli ask [options]
   
   Search documents:
   python -m src.cli search [options]
   
   Single query:
   python -m src.cli query "your question" [options]
   
   Show statistics:
   python -m src.cli stats [options]
   
   Show system info:
   python -m src.cli info

4. PYTHON API USAGE

   # Vectorize documents
   from src.library_vectorizer import LibraryVectorizer
   
   vectorizer = LibraryVectorizer(
       vector_store_type="milvus",
       embedding_model="BAAI/bge-small-en-v1.5",
       chunking_strategy="hierarchical"
   )
   
   result = vectorizer.vectorize_library("./documents", "my_library")
   
   # Use RAG system
   from src.rag_pipeline import SimpleRAG
   
   rag = SimpleRAG(vector_store_type="milvus", collection_name="my_library")
   answer = rag.ask("What is machine learning?")
   
   # Search documents
   docs = rag.search("machine learning", top_k=5)

5. CONFIGURATION

   Environment variables:
   export VECTOR_STORE_TYPE=milvus
   export EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
   export CHUNKING_STRATEGY=hierarchical
   
   Or edit the .env file (copy from env.example)

6. DOCKER USAGE

   # Start everything with Docker Compose
   docker-compose up -d
   
   # Or build and run manually
   docker build -t rag-system .
   docker run -p 8000:8000 rag-system

7. SUPPORTED DOCUMENT FORMATS

   - PDF files (.pdf)
   - Microsoft Word (.docx, .doc)
   - Plain text (.txt)
   - Markdown (.md)

8. VECTOR STORES

   Milvus:
   - High performance
   - Good for production
   - Requires Docker
   
   Qdrant:
   - RESTful API
   - Good alternative to Milvus
   - Requires Docker
   
   ChromaDB:
   - Easy to use
   - No external setup
   - Good for development

9. EMBEDDING MODELS

   Recommended models:
   - BAAI/bge-small-en-v1.5 (384 dim, fast)
   - BAAI/bge-base-en-v1.5 (768 dim, balanced)
   - BAAI/bge-large-en-v1.5 (1024 dim, high quality)

10. CHUNKING STRATEGIES

    Hierarchical:
    - Preserves document structure
    - Good for structured documents
    - Uses headings and sections
    
    Hybrid:
    - Combines semantic and structural chunking
    - Good for mixed content
    - Uses token-based splitting

11. TROUBLESHOOTING

    Common issues:
    - Vector store not running: Check Docker containers
    - Import errors: Run pip install -r requirements.txt
    - Memory issues: Use smaller embedding model
    - Slow processing: Use GPU if available
    
    Enable verbose logging:
    python -m src.cli vectorize ./documents --verbose

12. EXAMPLES

    See examples/basic_examples.py for detailed usage examples.

13. TESTING

    Run tests:
    python -m pytest tests/ -v
    
    Or run individual test files:
    python tests/test_system.py

14. CONTRIBUTING

    - Fork the repository
    - Create a feature branch
    - Make your changes
    - Add tests if applicable
    - Submit a pull request

15. SUPPORT

    - Check README.md for detailed documentation
    - Look at examples/ for usage patterns
    - Check issues on GitHub for known problems
    - Create an issue for bugs or feature requests
"""

if __name__ == "__main__":
    print(__doc__)

