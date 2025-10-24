"""
RAG Pipeline for Document Retrieval and Answer Generation

This module provides a complete RAG pipeline that integrates with the vectorized
document library to answer questions using retrieved context.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for question answering using vectorized documents
    """
    
    def __init__(
        self,
        vector_store_type: str = "milvus",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        collection_name: str = "docling_rag",
        llm_model: Optional[str] = None
    ):
        """
        Initialize RAG Pipeline
        
        Args:
            vector_store_type: Type of vector store ("milvus", "qdrant", "chroma")
            embedding_model: HuggingFace model for embeddings
            collection_name: Name of the collection in vector store
            llm_model: Optional LLM model for answer generation
        """
        self.vector_store_type = vector_store_type
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.llm_model = llm_model
        
        # Initialize components
        self._setup_components()
    
    def _setup_components(self):
        """Setup embedding model and vector store connection"""
        try:
            # Initialize embedding model
            self.embedder = SentenceTransformer(self.embedding_model)
            
            # Initialize vector store connection
            self._setup_vector_store()
            
            logger.info(f"RAG Pipeline initialized with {self.vector_store_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise
    
    def _setup_vector_store(self):
        """Setup connection to vector store"""
        if self.vector_store_type == "milvus":
            self._setup_milvus()
        elif self.vector_store_type == "qdrant":
            self._setup_qdrant()
        elif self.vector_store_type == "chroma":
            self._setup_chroma()
        else:
            raise ValueError(f"Unsupported vector store: {self.vector_store_type}")
    
    def _setup_milvus(self):
        """Setup Milvus connection"""
        try:
            from pymilvus import connections, Collection
            
            # Connect to Milvus
            connections.connect("default", host="localhost", port="19530")
            
            # Get collection
            self.collection = Collection(self.collection_name)
            self.collection.load()
            
            logger.info("Connected to Milvus vector store")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_qdrant(self):
        """Setup Qdrant connection"""
        try:
            from qdrant_client import QdrantClient
            import os
            
            # Get Qdrant configuration
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            api_key = os.getenv("QDRANT_API_KEY")
            
            # Initialize Qdrant client with API key if provided
            if api_key:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    https=False  # Use HTTP instead of HTTPS
                )
            else:
                self.client = QdrantClient(host=host, port=port, https=False)
            
            logger.info("Connected to Qdrant vector store")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _setup_chroma(self):
        """Setup ChromaDB connection"""
        try:
            import chromadb
            
            # Initialize ChromaDB client with new configuration
            self.client = chromadb.PersistentClient(
                path="./chroma_db"
            )
            
            # Get collection
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
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0]
        
        # Search in vector store
        if self.vector_store_type == "milvus":
            results = self._search_milvus(query_embedding, top_k)
        elif self.vector_store_type == "qdrant":
            results = self._search_qdrant(query_embedding, top_k)
        elif self.vector_store_type == "chroma":
            results = self._search_chroma(query, top_k)
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def _search_milvus(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search in Milvus vector store"""
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_embedding.tolist()],
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
    
    def _search_qdrant(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search in Qdrant vector store"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
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
                    'score': 1 - distance,  # Convert distance to similarity score
                    'metadata': metadata
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate answer using retrieved documents
        
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
        
        # Generate answer using LLM or simple template
        if self.llm_model:
            answer = self._generate_with_llm(query, context)
        else:
            answer = self._generate_template_answer(query, context, retrieved_docs)
        
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
    
    def _generate_with_llm(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        try:
            # This is a placeholder for LLM integration
            # You can integrate with OpenAI, HuggingFace, or local models
            
            prompt = f"""
Based on the following context, please answer the question: {query}

Context:
{context}

Answer:
"""
            
            # Placeholder response - replace with actual LLM call
            return f"Based on the retrieved documents, here's what I found regarding '{query}':\n\n[LLM integration needed - this is a placeholder response]"
            
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            return "Error generating answer with LLM."
    
    def _generate_template_answer(self, query: str, context: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using template (fallback when no LLM)"""
        answer_parts = [
            f"Query: {query}\n",
            f"Found {len(retrieved_docs)} relevant document chunks:\n"
        ]
        
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source_file', 'Unknown')
            score = doc.get('score', 0)
            
            answer_parts.append(f"{i}. From {source} (relevance: {score:.3f}):")
            answer_parts.append(f"   {doc['text'][:200]}...")
            answer_parts.append("")
        
        answer_parts.append("Note: For detailed answers, integrate with an LLM model.")
        
        return "\n".join(answer_parts)
    
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
                'collection': self.collection_name
            }
        }
        
        logger.info(f"Generated answer for query: {query}")
        return response
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if self.vector_store_type == "milvus":
                stats = self.collection.num_entities
                return {
                    'total_documents': stats,
                    'vector_store': self.vector_store_type,
                    'collection_name': self.collection_name
                }
            elif self.vector_store_type == "qdrant":
                info = self.client.get_collection(self.collection_name)
                return {
                    'total_documents': info.points_count,
                    'vector_store': self.vector_store_type,
                    'collection_name': self.collection_name
                }
            elif self.vector_store_type == "chroma":
                count = self.collection.count()
                return {
                    'total_documents': count,
                    'vector_store': self.vector_store_type,
                    'collection_name': self.collection_name
                }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


class SimpleRAG:
    """
    Simplified RAG interface for easy usage
    """
    
    def __init__(self, vector_store_type: str = "milvus", collection_name: str = "docling_rag"):
        """
        Initialize Simple RAG
        
        Args:
            vector_store_type: Type of vector store
            collection_name: Name of the collection
        """
        self.pipeline = RAGPipeline(
            vector_store_type=vector_store_type,
            collection_name=collection_name
        )
    
    def ask(self, question: str, top_k: int = 5) -> str:
        """
        Ask a question and get an answer
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Answer string
        """
        response = self.pipeline.ask_question(question, top_k)
        return response['answer']
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        return self.pipeline.retrieve_documents(query, top_k)


if __name__ == "__main__":
    # Example usage
    rag = SimpleRAG(vector_store_type="milvus", collection_name="my_library")
    
    # Ask a question
    answer = rag.ask("What is the main topic of the documents?")
    print(f"Answer: {answer}")
    
    # Search for documents
    docs = rag.search("machine learning", top_k=3)
    print(f"Found {len(docs)} relevant documents")

