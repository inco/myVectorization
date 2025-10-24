"""
Library Vectorization RAG System with Docling

A comprehensive system for converting document libraries into vectorized RAG systems
using Docling for document processing and various vector stores for retrieval.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from dotenv import load_dotenv
from .document_processors import ExtendedDocumentProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LibraryVectorizer:
    """
    Main class for vectorizing document libraries using Docling
    """
    
    def __init__(
        self,
        vector_store_type: str = "milvus",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunking_strategy: str = "hierarchical",
        use_ollama: bool = False,
        ollama_url: str = "http://192.168.1.89:11434",
        ollama_embedding_model: str = "nomic-embed-text"
    ):
        """
        Initialize the Library Vectorizer
        
        Args:
            vector_store_type: Type of vector store ("milvus", "qdrant", "chroma")
            embedding_model: HuggingFace model for embeddings
            chunking_strategy: Chunking strategy ("hierarchical", "hybrid")
            use_ollama: Whether to use Ollama for embeddings
            ollama_url: Ollama server URL
            ollama_embedding_model: Ollama embedding model name
        """
        self.vector_store_type = vector_store_type
        self.embedding_model = embedding_model
        self.chunking_strategy = chunking_strategy
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url
        self.ollama_embedding_model = ollama_embedding_model
        
        # Initialize components
        self._setup_components()
        
        # Initialize extended document processor
        self.extended_processor = ExtendedDocumentProcessor(
            vector_store_type, 
            "docling_rag"  # Default collection name
        )
    
    def _setup_components(self):
        """Setup Docling and vector store components"""
        try:
            from docling.document_converter import DocumentConverter
            from docling.chunking import HierarchicalChunker, HybridChunker
            
            # Initialize Docling components
            self.converter = DocumentConverter()
            
            if self.chunking_strategy == "hierarchical":
                self.chunker = HierarchicalChunker()
            elif self.chunking_strategy == "hybrid":
                self.chunker = HybridChunker(tokenizer=self.embedding_model)
            else:
                raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")
            
            # Initialize embedding model
            if self.use_ollama:
                from .ollama_integration import OllamaEmbeddings
                self.embedder = OllamaEmbeddings(self.ollama_url, self.ollama_embedding_model)
                logger.info(f"Using Ollama embeddings: {self.ollama_embedding_model}")
            else:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(self.embedding_model)
                logger.info(f"Using HuggingFace embeddings: {self.embedding_model}")
            
            logger.info(f"Initialized Library Vectorizer with {self.vector_store_type} vector store")
            
        except ImportError as e:
            logger.error(f"Failed to import required dependencies: {e}")
            raise
    
    def process_documents(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Process documents from input path and return processed chunks
        
        Args:
            input_path: Path to documents (file or directory)
            
        Returns:
            List of processed document chunks with metadata
        """
        input_path = Path(input_path)
        processed_chunks = []
        
        if input_path.is_file():
            files = [input_path]
        elif input_path.is_dir():
            # Support common document formats
            extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
            files = []
            for ext in extensions:
                files.extend(input_path.glob(f"**/*{ext}"))
        else:
            raise ValueError(f"Invalid input path: {input_path}")
        
        logger.info(f"Processing {len(files)} documents")
        
        for file_path in files:
            try:
                logger.info(f"Processing: {file_path}")
                
                # Convert document using Docling
                result = self.converter.convert(str(file_path))
                doc = result.document
                
                # Chunk the document
                chunks = list(self.chunker.chunk(doc))
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        'text': chunk.text,
                        'metadata': {
                            'source_file': str(file_path),
                            'chunk_id': f"{file_path.stem}_{i}",
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        }
                    }
                    
                    # Add Docling-specific metadata if available
                    if hasattr(chunk, 'meta') and chunk.meta:
                        chunk_data['metadata'].update({
                            'headings': getattr(chunk.meta, 'headings', []),
                            'page_number': getattr(chunk.meta, 'page_number', None),
                            'bbox': getattr(chunk.meta, 'bbox', None)
                        })
                    
                    processed_chunks.append(chunk_data)
                
                logger.info(f"Processed {len(chunks)} chunks from {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Total processed chunks: {len(processed_chunks)}")
        return processed_chunks
    
    def vectorize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of processed document chunks
            
        Returns:
            List of chunks with embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts for batch embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        if self.use_ollama:
            embeddings = self.embedder.embed_documents(texts)
        else:
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            if self.use_ollama:
                chunk['embedding'] = embeddings[i]
            else:
                chunk['embedding'] = embeddings[i].tolist()
        
        logger.info("Embeddings generated successfully")
        return chunks
    
    def store_vectors(self, vectorized_chunks: List[Dict[str, Any]], collection_name: str = "docling_rag"):
        """
        Store vectorized chunks in vector store
        
        Args:
            vectorized_chunks: List of chunks with embeddings
            collection_name: Name of the collection in vector store
        """
        logger.info(f"Storing {len(vectorized_chunks)} vectors in {self.vector_store_type}")
        
        if self.vector_store_type == "milvus":
            self._store_in_milvus(vectorized_chunks, collection_name)
        elif self.vector_store_type == "qdrant":
            self._store_in_qdrant(vectorized_chunks, collection_name)
        elif self.vector_store_type == "chroma":
            self._store_in_chroma(vectorized_chunks, collection_name)
        else:
            raise ValueError(f"Unsupported vector store: {self.vector_store_type}")
    
    def _store_in_milvus(self, chunks: List[Dict[str, Any]], collection_name: str):
        """Store vectors in Milvus"""
        try:
            from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
            
            # Connect to Milvus
            connections.connect("default", host="localhost", port="19530")
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(chunks[0]['embedding'])),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=255),
            ]
            
            schema = CollectionSchema(fields, f"RAG collection for {collection_name}")
            
            # Create or get collection
            if connections.has_connection("default"):
                try:
                    collection = Collection(name=collection_name, schema=schema)
                except:
                    collection = Collection(name=collection_name, schema=schema)
            
            # Prepare data for insertion
            data = [
                [chunk['text'] for chunk in chunks],
                [chunk['embedding'] for chunk in chunks],
                [chunk['metadata']['source_file'] for chunk in chunks],
                [chunk['metadata']['chunk_id'] for chunk in chunks],
            ]
            
            # Insert data
            collection.insert(data)
            collection.flush()
            
            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("embedding", index_params)
            
            logger.info(f"Successfully stored vectors in Milvus collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error storing in Milvus: {e}")
            raise
    
    def _store_in_qdrant(self, chunks: List[Dict[str, Any]], collection_name: str):
        """Store vectors in Qdrant"""
        try:
            from qdrant_client import QdrantClient, models
            import os
            
            # Get Qdrant configuration
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            api_key = os.getenv("QDRANT_API_KEY")
            
            # Initialize Qdrant client with API key if provided
            if api_key:
                client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    https=False  # Use HTTP instead of HTTPS
                )
            else:
                client = QdrantClient(host=host, port=port, https=False)
            
            # Create collection
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=len(chunks[0]['embedding']), 
                    distance=models.Distance.COSINE
                )
            )
            
            # Prepare points
            points = []
            for i, chunk in enumerate(chunks):
                point = models.PointStruct(
                    id=i,
                    vector=chunk['embedding'],
                    payload={
                        'text': chunk['text'],
                        'source_file': chunk['metadata']['source_file'],
                        'chunk_id': chunk['metadata']['chunk_id'],
                        **chunk['metadata']
                    }
                )
                points.append(point)
            
            # Upload points
            client.upsert(collection_name=collection_name, points=points)
            
            logger.info(f"Successfully stored vectors in Qdrant collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error storing in Qdrant: {e}")
            raise
    
    def _store_in_chroma(self, chunks: List[Dict[str, Any]], collection_name: str):
        """Store vectors in ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize ChromaDB client with new configuration
            client = chromadb.PersistentClient(
                path="./chroma_db"
            )
            
            # Create or get collection
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Prepare data
            documents = [chunk['text'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            ids = [chunk['metadata']['chunk_id'] for chunk in chunks]
            
            # Add documents (ChromaDB will generate embeddings automatically)
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully stored vectors in ChromaDB collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Error storing in ChromaDB: {e}")
            raise
    
    def vectorize_library(self, input_path: str, collection_name: str = "docling_rag") -> Dict[str, Any]:
        """
        Complete pipeline: process documents, generate embeddings, and store vectors
        
        Args:
            input_path: Path to documents
            collection_name: Name of the collection
            
        Returns:
            Summary of the vectorization process
        """
        logger.info(f"Starting library vectorization for: {input_path}")
        
        # Process documents
        chunks = self.process_documents(input_path)
        
        if not chunks:
            raise ValueError("No documents were processed successfully")
        
        # Generate embeddings
        vectorized_chunks = self.vectorize_chunks(chunks)
        
        # Store in vector database
        self.store_vectors(vectorized_chunks, collection_name)
        
        summary = {
            'total_chunks': len(vectorized_chunks),
            'collection_name': collection_name,
            'vector_store': self.vector_store_type,
            'embedding_model': self.embedding_model,
            'chunking_strategy': self.chunking_strategy
        }
        
        logger.info(f"Library vectorization completed: {summary}")
        return summary
    
    def scan_and_vectorize_folder(
        self, 
        folder_path: str, 
        collection_name: str = "docling_rag",
        recursive: bool = True,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Scan folder for documents and vectorize new ones
        
        Args:
            folder_path: Path to folder to scan
            collection_name: Name of the collection
            recursive: Whether to scan subfolders recursively
            force_reprocess: Whether to reprocess already processed files
            
        Returns:
            Summary of the vectorization process
        """
        logger.info(f"Scanning and vectorizing folder: {folder_path}")
        
        # Update collection name in processor
        self.extended_processor.collection_name = collection_name
        self.extended_processor.file_tracker.collection_name = collection_name
        
        if force_reprocess:
            # Clear processed files list to force reprocessing
            self.extended_processor.file_tracker.processed_files.clear()
            logger.info("Force reprocessing enabled - all files will be reprocessed")
        
        # Scan folder for documents
        processed_docs = self.extended_processor.scan_folder(folder_path, recursive)
        
        if not processed_docs:
            logger.info("No new documents found to process")
            return {
                'total_chunks': 0,
                'collection_name': collection_name,
                'vector_store': self.vector_store_type,
                'embedding_model': self.embedding_model,
                'chunking_strategy': self.chunking_strategy,
                'processed_files': 0,
                'skipped_files': len(self.extended_processor.file_tracker.processed_files)
            }
        
        logger.info(f"Processing {len(processed_docs)} documents")
        
        # Process documents and create chunks
        all_chunks = []
        for doc in processed_docs:
            try:
                # Convert document content to Docling format for chunking
                chunks = self._process_document_content(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing document {doc['file_path']}: {e}")
                continue
        
        if not all_chunks:
            logger.warning("No chunks were created from documents")
            return {
                'total_chunks': 0,
                'collection_name': collection_name,
                'vector_store': self.vector_store_type,
                'embedding_model': self.embedding_model,
                'chunking_strategy': self.chunking_strategy,
                'processed_files': len(processed_docs),
                'skipped_files': len(self.extended_processor.file_tracker.processed_files)
            }
        
        # Generate embeddings
        vectorized_chunks = self.vectorize_chunks(all_chunks)
        
        # Store in vector database
        self.store_vectors(vectorized_chunks, collection_name)
        
        summary = {
            'total_chunks': len(vectorized_chunks),
            'collection_name': collection_name,
            'vector_store': self.vector_store_type,
            'embedding_model': self.embedding_model,
            'chunking_strategy': self.chunking_strategy,
            'processed_files': len(processed_docs),
            'skipped_files': len(self.extended_processor.file_tracker.processed_files)
        }
        
        logger.info(f"Folder scan and vectorization completed: {summary}")
        return summary
    
    def _process_document_content(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process document content and create chunks
        
        Args:
            doc: Document data from extended processor
            
        Returns:
            List of document chunks
        """
        try:
            # Create a temporary file for Docling processing
            import tempfile
            import os
            
            # Determine file extension based on content
            if doc['file_type'] in ['.epub', '.fb2']:
                # For EPUB/FB2, create a temporary markdown file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
                    temp_file.write(doc['content'])
                    temp_path = temp_file.name
            else:
                # For other formats, use original file
                temp_path = doc['file_path']
            
            # Convert using Docling
            result = self.converter.convert(temp_path)
            docling_doc = result.document
            
            # Chunk the document
            chunks = list(self.chunker.chunk(docling_doc))
            
            # Process each chunk
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'text': chunk.text,
                    'metadata': {
                        'source_file': doc['file_path'],
                        'chunk_id': f"{Path(doc['file_path']).stem}_{i}",
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'file_type': doc['file_type'],
                        'file_size': doc['file_size'],
                        'file_hash': doc['file_hash']
                    }
                }
                
                # Add Docling-specific metadata if available
                if hasattr(chunk, 'meta') and chunk.meta:
                    chunk_data['metadata'].update({
                        'headings': getattr(chunk.meta, 'headings', []),
                        'page_number': getattr(chunk.meta, 'page_number', None),
                        'bbox': getattr(chunk.meta, 'bbox', None)
                    })
                
                processed_chunks.append(chunk_data)
            
            # Clean up temporary file if created
            if doc['file_type'] in ['.epub', '.fb2'] and temp_path != doc['file_path']:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document content: {e}")
            return []
    
    def get_processed_files(self) -> List[str]:
        """Get list of already processed files"""
        return list(self.extended_processor.file_tracker.processed_files)
    
    def clear_processed_files(self):
        """Clear the list of processed files (for force reprocessing)"""
        self.extended_processor.file_tracker.processed_files.clear()
        logger.info("Cleared processed files list")


if __name__ == "__main__":
    # Example usage
    vectorizer = LibraryVectorizer(
        vector_store_type="milvus",
        embedding_model="BAAI/bge-small-en-v1.5",
        chunking_strategy="hierarchical"
    )
    
    # Vectorize documents
    result = vectorizer.vectorize_library("./documents", "my_library")
    print(f"Vectorization completed: {result}")

