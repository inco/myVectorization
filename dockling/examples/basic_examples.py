"""
Example usage of the Library Vectorization RAG System

This module demonstrates how to use the system for different scenarios.
"""

import os
import logging
from pathlib import Path
from library_vectorizer import LibraryVectorizer
from rag_pipeline import SimpleRAG, RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_vectorization():
    """Basic example of vectorizing documents"""
    print("=== Basic Vectorization Example ===")
    
    # Initialize vectorizer
    vectorizer = LibraryVectorizer(
        vector_store_type="milvus",
        embedding_model="BAAI/bge-small-en-v1.5",
        chunking_strategy="hierarchical"
    )
    
    # Vectorize documents (replace with your document path)
    documents_path = "./example_documents"
    
    if not Path(documents_path).exists():
        print(f"Creating example documents directory: {documents_path}")
        Path(documents_path).mkdir(exist_ok=True)
        
        # Create a sample document
        sample_doc = """
# Machine Learning Overview

Machine learning is a subset of artificial intelligence that focuses on algorithms 
that can learn from data without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping from inputs to outputs.
Common algorithms include linear regression, decision trees, and neural networks.

### Unsupervised Learning
Unsupervised learning finds patterns in data without labeled examples.
Examples include clustering, dimensionality reduction, and association rules.

### Reinforcement Learning
Reinforcement learning learns through interaction with an environment.
The agent receives rewards or penalties for its actions.

## Applications
Machine learning is used in many fields including:
- Computer vision
- Natural language processing
- Recommendation systems
- Fraud detection
- Medical diagnosis
"""
        
        with open(f"{documents_path}/ml_overview.md", "w", encoding="utf-8") as f:
            f.write(sample_doc)
        
        print("Sample document created. Please add your own documents to the directory.")
        return
    
    try:
        # Vectorize the library
        result = vectorizer.vectorize_library(documents_path, "example_library")
        print(f"Vectorization completed: {result}")
        
    except Exception as e:
        print(f"Error during vectorization: {e}")


def example_rag_qa():
    """Example of using RAG for question answering"""
    print("\n=== RAG Q&A Example ===")
    
    # Initialize RAG system
    rag = SimpleRAG(
        vector_store_type="milvus",
        collection_name="example_library"
    )
    
    # Example questions
    questions = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "What are some applications of machine learning?",
        "How does supervised learning work?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            answer = rag.ask(question, top_k=3)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error answering question: {e}")


def example_document_search():
    """Example of searching for relevant documents"""
    print("\n=== Document Search Example ===")
    
    # Initialize RAG system
    rag = SimpleRAG(
        vector_store_type="milvus",
        collection_name="example_library"
    )
    
    # Example search queries
    queries = [
        "supervised learning algorithms",
        "neural networks",
        "clustering methods",
        "reinforcement learning"
    ]
    
    for query in queries:
        print(f"\nSearch Query: {query}")
        try:
            docs = rag.search(query, top_k=3)
            print(f"Found {len(docs)} relevant documents:")
            
            for i, doc in enumerate(docs, 1):
                source = doc['metadata'].get('source_file', 'Unknown')
                score = doc['score']
                preview = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
                
                print(f"  {i}. {source} (score: {score:.3f})")
                print(f"     {preview}")
                
        except Exception as e:
            print(f"Error searching documents: {e}")


def example_advanced_rag():
    """Example of advanced RAG usage with custom configuration"""
    print("\n=== Advanced RAG Example ===")
    
    # Initialize advanced RAG pipeline
    pipeline = RAGPipeline(
        vector_store_type="milvus",
        embedding_model="BAAI/bge-small-en-v1.5",
        collection_name="example_library",
        llm_model="gpt-3.5-turbo"  # Optional LLM integration
    )
    
    # Ask a complex question
    question = "Compare supervised and unsupervised learning approaches"
    
    try:
        response = pipeline.ask_question(question, top_k=5)
        
        print(f"Question: {response['query']}")
        print(f"Answer: {response['answer']}")
        print(f"Retrieved {response['num_documents']} documents")
        
        # Show retrieved documents
        print("\nRetrieved Documents:")
        for i, doc in enumerate(response['retrieved_documents'], 1):
            source = doc['metadata'].get('source_file', 'Unknown')
            score = doc['score']
            print(f"  {i}. {source} (relevance: {score:.3f})")
            
    except Exception as e:
        print(f"Error in advanced RAG: {e}")


def example_collection_stats():
    """Example of getting collection statistics"""
    print("\n=== Collection Statistics Example ===")
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(
        vector_store_type="milvus",
        collection_name="example_library"
    )
    
    try:
        stats = pipeline.get_collection_stats()
        print("Collection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error getting stats: {e}")


def example_different_vector_stores():
    """Example using different vector stores"""
    print("\n=== Different Vector Stores Example ===")
    
    vector_stores = ["milvus", "qdrant", "chroma"]
    
    for vs_type in vector_stores:
        print(f"\nTesting {vs_type} vector store:")
        
        try:
            # Initialize vectorizer
            vectorizer = LibraryVectorizer(
                vector_store_type=vs_type,
                embedding_model="BAAI/bge-small-en-v1.5",
                chunking_strategy="hierarchical"
            )
            
            # Initialize RAG
            rag = SimpleRAG(
                vector_store_type=vs_type,
                collection_name=f"test_{vs_type}"
            )
            
            print(f"  ✓ {vs_type} initialized successfully")
            
            # Test a simple query
            answer = rag.ask("What is machine learning?", top_k=2)
            print(f"  ✓ Query answered successfully")
            
        except Exception as e:
            print(f"  ✗ Error with {vs_type}: {e}")


def example_chunking_strategies():
    """Example using different chunking strategies"""
    print("\n=== Chunking Strategies Example ===")
    
    strategies = ["hierarchical", "hybrid"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy} chunking:")
        
        try:
            # Initialize vectorizer
            vectorizer = LibraryVectorizer(
                vector_store_type="milvus",
                embedding_model="BAAI/bge-small-en-v1.5",
                chunking_strategy=strategy
            )
            
            # Process documents
            documents_path = "./example_documents"
            if Path(documents_path).exists():
                chunks = vectorizer.process_documents(documents_path)
                print(f"  ✓ Processed {len(chunks)} chunks with {strategy} strategy")
                
                # Show sample chunk
                if chunks:
                    sample_chunk = chunks[0]
                    print(f"  Sample chunk: {sample_chunk['text'][:100]}...")
            
        except Exception as e:
            print(f"  ✗ Error with {strategy}: {e}")


def main():
    """Run all examples"""
    print("Library Vectorization RAG System - Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_vectorization()
    example_rag_qa()
    example_document_search()
    example_advanced_rag()
    example_collection_stats()
    example_different_vector_stores()
    example_chunking_strategies()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()

