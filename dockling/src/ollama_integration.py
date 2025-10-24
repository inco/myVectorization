"""
Ollama integration for embeddings and text generation

This module provides integration with Ollama for local embeddings and LLM capabilities.
"""

import logging
import requests
import json
from typing import List, Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """Ollama embeddings client"""
    
    def __init__(self, base_url: str = "http://192.168.1.89:11434", model: str = "nomic-embed-text:latest"):
        """
        Initialize Ollama embeddings client
        
        Args:
            base_url: Ollama server URL
            model: Embedding model name
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Ollama at {self.base_url}")
                
                # Check if model is available
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                    if model_names:
                        self.model = model_names[0]  # Use first available model
                        logger.info(f"Using model: {self.model}")
                else:
                    logger.info(f"Using embedding model: {self.model}")
            else:
                raise Exception(f"Ollama server returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = self.session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('embedding', [])
            else:
                logger.error(f"Ollama embeddings API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            logger.info(f"Generating embedding {i+1}/{len(texts)}")
            embedding = self.embed_text(text)
            embeddings.append(embedding)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
        
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    if model['name'] == self.model:
                        return model
            return {}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}


class OllamaLLM:
    """Ollama LLM client for text generation"""
    
    def __init__(self, base_url: str = "http://192.168.1.89:11434", model: str = "llama3:latest"):
        """
        Initialize Ollama LLM client
        
        Args:
            base_url: Ollama server URL
            model: LLM model name
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Ollama LLM at {self.base_url}")
                
                # Check if model is available
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                    if model_names:
                        self.model = model_names[0]  # Use first available model
                        logger.info(f"Using LLM model: {self.model}")
                else:
                    logger.info(f"Using LLM model: {self.model}")
            else:
                raise Exception(f"Ollama server returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama LLM at {self.base_url}: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text response
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Ollama generate API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Assistant response
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False
            }
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '')
            else:
                logger.error(f"Ollama chat API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return ""


class OllamaRAGPipeline:
    """Complete RAG pipeline using Ollama"""
    
    def __init__(
        self, 
        ollama_url: str = "http://192.168.1.89:11434",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        vector_store_type: str = "qdrant",
        collection_name: str = "docling_rag"
    ):
        """
        Initialize Ollama RAG pipeline
        
        Args:
            ollama_url: Ollama server URL
            embedding_model: Embedding model name
            llm_model: LLM model name
            vector_store_type: Vector store type
            collection_name: Collection name
        """
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_store_type = vector_store_type
        self.collection_name = collection_name
        
        # Initialize components
        self.embeddings = OllamaEmbeddings(ollama_url, embedding_model)
        self.llm = OllamaLLM(ollama_url, llm_model)
        
        # Initialize vector store connection
        self._setup_vector_store()
    
    def _setup_vector_store(self):
        """Setup connection to vector store"""
        if self.vector_store_type == "qdrant":
            self._setup_qdrant()
        elif self.vector_store_type == "milvus":
            self._setup_milvus()
        elif self.vector_store_type == "chroma":
            self._setup_chroma()
        else:
            raise ValueError(f"Unsupported vector store: {self.vector_store_type}")
    
    def _setup_qdrant(self):
        """Setup Qdrant connection"""
        try:
            from qdrant_client import QdrantClient
            import os
            
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            api_key = os.getenv("QDRANT_API_KEY")
            
            if api_key:
                self.client = QdrantClient(host=host, port=port, api_key=api_key, https=False)
            else:
                self.client = QdrantClient(host=host, port=port, https=False)
            
            logger.info("Connected to Qdrant vector store")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _setup_milvus(self):
        """Setup Milvus connection"""
        try:
            from pymilvus import connections, Collection
            
            connections.connect("default", host="localhost", port=19530)
            self.collection = Collection(self.collection_name)
            self.collection.load()
            
            logger.info("Connected to Milvus vector store")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_chroma(self):
        """Setup ChromaDB connection"""
        try:
            import chromadb
            
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_collection(self.collection_name)
            
            logger.info("Connected to ChromaDB vector store")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        # Generate query embedding using Ollama
        query_embedding = self.embeddings.embed_text(query)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search in vector store
        if self.vector_store_type == "qdrant":
            results = self._search_qdrant(query_embedding, top_k)
        elif self.vector_store_type == "milvus":
            results = self._search_milvus(query_embedding, top_k)
        elif self.vector_store_type == "chroma":
            results = self._search_chroma(query, top_k)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def _search_qdrant(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search in Qdrant vector store"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            documents = []
            for hit in results:
                doc = {
                    'text': hit.payload.get('text'),
                    'score': hit.score,
                    'metadata': {
                        'source_file': hit.payload.get('source_file'),
                        'chunk_id': hit.payload.get('chunk_id'),
                        **hit.payload
                    }
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []
    
    def _search_milvus(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search in Milvus vector store"""
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "source_file", "chunk_id"]
            )
            
            documents = []
            for hit in results[0]:
                doc = {
                    'text': hit.entity.get('text'),
                    'score': hit.score,
                    'metadata': {
                        'source_file': hit.entity.get('source_file'),
                        'chunk_id': hit.entity.get('chunk_id')
                    }
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching Milvus: {e}")
            return []
    
    def _search_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search in ChromaDB vector store"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            documents = []
            for i, (text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                doc = {
                    'text': text,
                    'score': 1 - distance,
                    'metadata': metadata
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate answer using retrieved documents and Ollama LLM
        
        Args:
            query: User query
            retrieved_docs: Retrieved document chunks
            
        Returns:
            Generated answer
        """
        if not retrieved_docs:
            return "No relevant documents found for your query."
        
        # Create context from retrieved documents
        context = self._create_context(retrieved_docs)
        
        # Create prompt for Ollama
        prompt = f"""Based on the following context, please answer the question: {query}

Context:
{context}

Answer:"""
        
        # Generate answer using Ollama LLM
        answer = self.llm.generate(prompt, max_tokens=1000, temperature=0.7)
        
        return answer
    
    def _create_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Create context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source_file', 'Unknown')
            chunk_id = doc['metadata'].get('chunk_id', f'chunk_{i}')
            
            context_part = f"""
Document {i} (Source: {source}, Chunk: {chunk_id}):
{doc['text']}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def ask_question(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate answer
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing question: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_docs)
        
        # Prepare response
        response = {
            'query': query,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'num_documents': len(retrieved_docs),
            'metadata': {
                'vector_store': self.vector_store_type,
                'embedding_model': self.embedding_model,
                'llm_model': self.llm_model,
                'ollama_url': self.ollama_url,
                'collection': self.collection_name
            }
        }
        
        logger.info(f"Generated answer for query: {query}")
        return response


if __name__ == "__main__":
    # Test Ollama integration
    try:
        # Test embeddings
        embeddings = OllamaEmbeddings()
        test_embedding = embeddings.embed_text("Hello, world!")
        print(f"Embedding dimension: {len(test_embedding)}")
        
        # Test LLM
        llm = OllamaLLM()
        response = llm.generate("What is machine learning?")
        print(f"LLM response: {response}")
        
    except Exception as e:
        print(f"Error testing Ollama: {e}")
