#!/usr/bin/env python3
"""
Simple test script for Ollama integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ollama_integration import OllamaEmbeddings, OllamaLLM, OllamaRAGPipeline

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("Testing Ollama connection...")
    
    try:
        # Test embeddings
        print("Testing embeddings...")
        embeddings = OllamaEmbeddings()
        
        test_text = "Hello, this is a test document about machine learning."
        embedding = embeddings.embed_text(test_text)
        
        if embedding:
            print(f"Embedding generated successfully! Dimension: {len(embedding)}")
        else:
            print("Failed to generate embedding")
            return False
        
        # Test LLM
        print("Testing LLM...")
        llm = OllamaLLM()
        
        response = llm.generate("What is machine learning?", max_tokens=100)
        
        if response:
            print(f"LLM response generated: {response[:100]}...")
        else:
            print("Failed to generate LLM response")
            return False
        
        print("All Ollama tests passed!")
        return True
        
    except Exception as e:
        print(f"Error testing Ollama: {e}")
        return False

def test_ollama_rag():
    """Test complete RAG pipeline with Ollama"""
    print("\nTesting Ollama RAG pipeline...")
    
    try:
        # Initialize RAG pipeline
        rag = OllamaRAGPipeline(
            vector_store_type="qdrant",
            collection_name="test_ollama_rag"
        )
        
        # Test question
        query = "What is artificial intelligence?"
        print(f"Asking: {query}")
        
        response = rag.ask_question(query, top_k=3)
        
        if response['answer']:
            print(f"Answer generated: {response['answer'][:200]}...")
            print(f"Retrieved {response['num_documents']} documents")
        else:
            print("No answer generated")
            return False
        
        print("Ollama RAG pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"Error testing Ollama RAG: {e}")
        return False

if __name__ == "__main__":
    print("Starting Ollama Integration Tests")
    print("=" * 50)
    
    # Test basic connection
    if test_ollama_connection():
        print("\n" + "=" * 50)
        # Test RAG pipeline
        test_ollama_rag()
    
    print("\nTests completed!")
