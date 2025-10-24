"""
Test suite for the Library Vectorization RAG System

This module contains tests for the core functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Import modules to test
from src.library_vectorizer import LibraryVectorizer
from src.rag_pipeline import RAGPipeline, SimpleRAG
from src.config import Config, get_config


class TestLibraryVectorizer:
    """Test cases for LibraryVectorizer"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_doc_path = Path(self.temp_dir) / "test.md"
        
        # Create test document
        test_content = """
# Test Document

This is a test document for unit testing.

## Section 1
This is the first section with some content.

## Section 2
This is the second section with more content.
"""
        self.test_doc_path.write_text(test_content)
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.library_vectorizer.DocumentConverter')
    @patch('src.library_vectorizer.HierarchicalChunker')
    @patch('src.library_vectorizer.SentenceTransformer')
    def test_initialization(self, mock_transformer, mock_chunker, mock_converter):
        """Test vectorizer initialization"""
        vectorizer = LibraryVectorizer(
            vector_store_type="milvus",
            embedding_model="BAAI/bge-small-en-v1.5",
            chunking_strategy="hierarchical"
        )
        
        assert vectorizer.vector_store_type == "milvus"
        assert vectorizer.embedding_model == "BAAI/bge-small-en-v1.5"
        assert vectorizer.chunking_strategy == "hierarchical"
    
    @patch('src.library_vectorizer.DocumentConverter')
    @patch('src.library_vectorizer.HierarchicalChunker')
    @patch('src.library_vectorizer.SentenceTransformer')
    def test_process_documents(self, mock_transformer, mock_chunker, mock_converter):
        """Test document processing"""
        # Mock the converter and chunker
        mock_converter.return_value.convert.return_value.document = Mock()
        mock_chunker.return_value.chunk.return_value = [
            Mock(text="Test chunk 1", meta=Mock(headings=["Section 1"])),
            Mock(text="Test chunk 2", meta=Mock(headings=["Section 2"]))
        ]
        
        vectorizer = LibraryVectorizer()
        chunks = vectorizer.process_documents(str(self.test_doc_path))
        
        assert len(chunks) == 2
        assert chunks[0]['text'] == "Test chunk 1"
        assert chunks[1]['text'] == "Test chunk 2"
    
    @patch('src.library_vectorizer.SentenceTransformer')
    def test_vectorize_chunks(self, mock_transformer):
        """Test chunk vectorization"""
        # Mock the transformer
        mock_transformer.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        vectorizer = LibraryVectorizer()
        chunks = [
            {'text': 'Test chunk 1', 'metadata': {'chunk_id': '1'}},
            {'text': 'Test chunk 2', 'metadata': {'chunk_id': '2'}}
        ]
        
        vectorized_chunks = vectorizer.vectorize_chunks(chunks)
        
        assert len(vectorized_chunks) == 2
        assert 'embedding' in vectorized_chunks[0]
        assert len(vectorized_chunks[0]['embedding']) == 3


class TestRAGPipeline:
    """Test cases for RAGPipeline"""
    
    @patch('src.rag_pipeline.SentenceTransformer')
    @patch('src.rag_pipeline.Collection')
    def test_initialization(self, mock_collection, mock_transformer):
        """Test RAG pipeline initialization"""
        # Mock Milvus connection
        with patch('src.rag_pipeline.connections') as mock_connections:
            mock_connections.connect.return_value = None
            mock_connections.has_connection.return_value = True
            
            pipeline = RAGPipeline(
                vector_store_type="milvus",
                embedding_model="BAAI/bge-small-en-v1.5",
                collection_name="test_collection"
            )
            
            assert pipeline.vector_store_type == "milvus"
            assert pipeline.embedding_model == "BAAI/bge-small-en-v1.5"
            assert pipeline.collection_name == "test_collection"
    
    @patch('src.rag_pipeline.SentenceTransformer')
    @patch('src.rag_pipeline.Collection')
    def test_retrieve_documents(self, mock_collection, mock_transformer):
        """Test document retrieval"""
        # Mock the transformer
        mock_transformer.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Mock Milvus search results
        mock_hit = Mock()
        mock_hit.score = 0.95
        mock_hit.entity.get.side_effect = lambda x: {
            'text': 'Test document content',
            'source_file': 'test.pdf',
            'chunk_id': 'chunk_1'
        }.get(x)
        
        mock_collection.return_value.search.return_value = [[mock_hit]]
        
        with patch('src.rag_pipeline.connections') as mock_connections:
            mock_connections.connect.return_value = None
            mock_connections.has_connection.return_value = True
            
            pipeline = RAGPipeline(vector_store_type="milvus")
            results = pipeline.retrieve_documents("test query", top_k=1)
            
            assert len(results) == 1
            assert results[0]['text'] == 'Test document content'
            assert results[0]['score'] == 0.95
    
    @patch('src.rag_pipeline.SentenceTransformer')
    @patch('src.rag_pipeline.Collection')
    def test_generate_answer(self, mock_collection, mock_transformer):
        """Test answer generation"""
        with patch('src.rag_pipeline.connections') as mock_connections:
            mock_connections.connect.return_value = None
            mock_connections.has_connection.return_value = True
            
            pipeline = RAGPipeline(vector_store_type="milvus")
            
            retrieved_docs = [
                {
                    'text': 'Machine learning is a subset of AI',
                    'metadata': {'source_file': 'ml.pdf', 'chunk_id': '1'}
                }
            ]
            
            answer = pipeline.generate_answer("What is machine learning?", retrieved_docs)
            
            assert "Machine learning" in answer
            assert "ml.pdf" in answer


class TestSimpleRAG:
    """Test cases for SimpleRAG"""
    
    @patch('src.rag_pipeline.RAGPipeline')
    def test_ask(self, mock_pipeline):
        """Test SimpleRAG ask method"""
        mock_pipeline.return_value.ask_question.return_value = {
            'answer': 'Test answer',
            'query': 'test question'
        }
        
        rag = SimpleRAG()
        answer = rag.ask("test question")
        
        assert answer == 'Test answer'
    
    @patch('src.rag_pipeline.RAGPipeline')
    def test_search(self, mock_pipeline):
        """Test SimpleRAG search method"""
        mock_pipeline.return_value.retrieve_documents.return_value = [
            {'text': 'Test document', 'score': 0.9, 'metadata': {}}
        ]
        
        rag = SimpleRAG()
        docs = rag.search("test query")
        
        assert len(docs) == 1
        assert docs[0]['text'] == 'Test document'


class TestConfig:
    """Test cases for Config class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = Config()
        
        assert config.get("vector_store.type") == "milvus"
        assert config.get("embeddings.model") == "BAAI/bge-small-en-v1.5"
        assert config.get("chunking.strategy") == "hierarchical"
    
    def test_env_override(self):
        """Test environment variable override"""
        with patch.dict(os.environ, {'VECTOR_STORE_TYPE': 'qdrant'}):
            config = Config()
            assert config.get("vector_store.type") == "qdrant"
    
    def test_get_vector_store_config(self):
        """Test getting vector store configuration"""
        config = Config()
        vs_config = config.get_vector_store_config()
        
        assert "host" in vs_config
        assert "port" in vs_config
    
    def test_get_embedding_config(self):
        """Test getting embedding configuration"""
        config = Config()
        embed_config = config.get_embedding_config()
        
        assert "model" in embed_config
        assert "dimension" in embed_config


class TestIntegration:
    """Integration tests"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_doc_path = Path(self.temp_dir) / "integration_test.md"
        
        test_content = """
# Integration Test Document

This document tests the integration between components.

## Machine Learning
Machine learning is a method of data analysis.

## Deep Learning
Deep learning is a subset of machine learning.
"""
        self.test_doc_path.write_text(test_content)
    
    def teardown_method(self):
        """Cleanup integration test environment"""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.library_vectorizer.DocumentConverter')
    @patch('src.library_vectorizer.HierarchicalChunker')
    @patch('src.library_vectorizer.SentenceTransformer')
    @patch('src.rag_pipeline.Collection')
    def test_end_to_end_workflow(self, mock_collection, mock_transformer, mock_chunker, mock_converter):
        """Test end-to-end workflow"""
        # Mock document processing
        mock_converter.return_value.convert.return_value.document = Mock()
        mock_chunker.return_value.chunk.return_value = [
            Mock(text="Machine learning is a method of data analysis.", meta=Mock(headings=["Machine Learning"]))
        ]
        
        # Mock embedding
        mock_transformer.return_value.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Mock vector store
        with patch('src.library_vectorizer.connections') as mock_connections:
            mock_connections.connect.return_value = None
            mock_connections.has_connection.return_value = True
            
            # Test vectorization
            vectorizer = LibraryVectorizer()
            chunks = vectorizer.process_documents(str(self.test_doc_path))
            vectorized_chunks = vectorizer.vectorize_chunks(chunks)
            
            assert len(vectorized_chunks) == 1
            assert 'embedding' in vectorized_chunks[0]
        
        # Test RAG
        with patch('src.rag_pipeline.connections') as mock_connections:
            mock_connections.connect.return_value = None
            mock_connections.has_connection.return_value = True
            
            mock_hit = Mock()
            mock_hit.score = 0.95
            mock_hit.entity.get.side_effect = lambda x: {
                'text': 'Machine learning is a method of data analysis.',
                'source_file': 'integration_test.md',
                'chunk_id': 'chunk_1'
            }.get(x)
            
            mock_collection.return_value.search.return_value = [[mock_hit]]
            
            rag = SimpleRAG()
            answer = rag.ask("What is machine learning?")
            
            assert "Machine learning" in answer


# Utility functions for testing
def create_test_document(content: str, filename: str = "test.md") -> Path:
    """Create a test document for testing"""
    temp_dir = tempfile.mkdtemp()
    test_file = Path(temp_dir) / filename
    test_file.write_text(content)
    return test_file

def cleanup_test_document(file_path: Path):
    """Cleanup test document"""
    shutil.rmtree(file_path.parent)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

