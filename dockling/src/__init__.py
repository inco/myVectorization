"""
Library Vectorization RAG System

A comprehensive system for converting document libraries into vectorized RAG systems
using Docling for document processing and various vector stores for retrieval.
"""

__version__ = "1.0.0"
__author__ = "Library Vectorization RAG System"
__description__ = "Document library vectorization and RAG system using Docling"

from .library_vectorizer import LibraryVectorizer
from .rag_pipeline import SimpleRAG, RAGPipeline
from .config import Config, get_config

__all__ = [
    "LibraryVectorizer",
    "SimpleRAG", 
    "RAGPipeline",
    "Config",
    "get_config"
]
