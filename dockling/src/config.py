"""
Configuration file for the Library Vectorization RAG System

This file contains default configurations and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Default configurations
DEFAULT_CONFIG = {
    # Vector store settings
    "vector_store": {
        "type": "milvus",  # Options: milvus, qdrant, chroma
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "docling_rag"
        },
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "docling_rag"
        },
        "chroma": {
            "persist_directory": "./chroma_db",
            "collection_name": "docling_rag"
        }
    },
    
    # Embedding settings
    "embeddings": {
        "model": "BAAI/bge-small-en-v1.5",  # HuggingFace model
        "dimension": 384,  # Dimension for bge-small-en-v1.5
        "batch_size": 32,
        "device": "auto"  # auto, cpu, cuda
    },
    
    # Ollama settings
    "ollama": {
        "base_url": "http://192.168.1.89:11434",
        "embedding_model": "nomic-embed-text:latest",
        "llm_model": "llama3:latest",
        "timeout": 30
    },
    
    # Chunking settings
    "chunking": {
        "strategy": "hierarchical",  # Options: hierarchical, hybrid
        "hierarchical": {
            "max_chunk_size": 1000,
            "overlap": 200
        },
        "hybrid": {
            "tokenizer": "BAAI/bge-small-en-v1.5",
            "max_tokens": 512,
            "overlap": 50
        }
    },
    
    # RAG settings
    "rag": {
        "top_k": 5,
        "similarity_threshold": 0.7,
        "max_context_length": 4000,
        "llm_model": None  # Optional LLM integration
    },
    
    # Document processing
    "documents": {
        "supported_formats": [".pdf", ".docx", ".doc", ".txt", ".md", ".epub", ".fb2"],
        "max_file_size": 50 * 1024 * 1024,  # 50MB
        "temp_directory": "./temp_docs"
    },
    
    # Logging
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": None  # Optional log file
    }
}

# Environment variable mappings
ENV_MAPPINGS = {
    "VECTOR_STORE_TYPE": "vector_store.type",
    "MILVUS_HOST": "vector_store.milvus.host",
    "MILVUS_PORT": "vector_store.milvus.port",
    "QDRANT_HOST": "vector_store.qdrant.host",
    "QDRANT_PORT": "vector_store.qdrant.port",
    "QDRANT_API_KEY": "vector_store.qdrant.api_key",
    "QDRANT_COLLECTION": "vector_store.qdrant.collection_name",
    "EMBEDDING_MODEL": "embeddings.model",
    "CHUNKING_STRATEGY": "chunking.strategy",
    "RAG_TOP_K": "rag.top_k",
    "LOG_LEVEL": "logging.level",
    "OLLAMA_BASE_URL": "ollama.base_url",
    "OLLAMA_EMBEDDING_MODEL": "ollama.embedding_model",
    "OLLAMA_LLM_MODEL": "ollama.llm_model"
}


class Config:
    """Configuration manager for the RAG system"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Optional path to custom config file
        """
        self.config = DEFAULT_CONFIG.copy()
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file if provided
        if config_file:
            self._load_from_file(config_file)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        for env_var, config_path in ENV_MAPPINGS.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, value)
    
    def _load_from_file(self, config_file: str):
        """Load configuration from file"""
        try:
            import json
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self._merge_config(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _set_nested_value(self, path: str, value: Any):
        """Set a nested configuration value"""
        keys = path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').replace('-', '').isdigit() and '.' in value:
                # Only convert to float if it's a valid number (not IP addresses)
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
        
        current[keys[-1]] = value
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing"""
        def merge_dict(d1, d2):
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    merge_dict(d1[key], value)
                else:
                    d1[key] = value
        
        merge_dict(self.config, new_config)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path
        
        Args:
            path: Dot-separated path to configuration value
            default: Default value if path not found
            
        Returns:
            Configuration value
        """
        keys = path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration"""
        vs_type = self.get("vector_store.type", "milvus")
        return self.get(f"vector_store.{vs_type}", {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.get("embeddings", {})
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking configuration"""
        strategy = self.get("chunking.strategy", "hierarchical")
        return self.get(f"chunking.{strategy}", {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration"""
        return self.get("rag", {})
    
    def get_document_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        return self.get("documents", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get("logging", {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config.copy()


# Global configuration instance
config = Config()

# Convenience functions
def get_config() -> Config:
    """Get global configuration instance"""
    return config

def get_vector_store_type() -> str:
    """Get vector store type"""
    return config.get("vector_store.type", "milvus")

def get_embedding_model() -> str:
    """Get embedding model name"""
    return config.get("embeddings.model", "BAAI/bge-small-en-v1.5")

def get_chunking_strategy() -> str:
    """Get chunking strategy"""
    return config.get("chunking.strategy", "hierarchical")

def get_collection_name() -> str:
    """Get default collection name"""
    vs_type = get_vector_store_type()
    return config.get(f"vector_store.{vs_type}.collection_name", "docling_rag")

def get_rag_top_k() -> int:
    """Get RAG top-k value"""
    return config.get("rag.top_k", 5)

def get_supported_formats() -> list:
    """Get supported document formats"""
    return config.get("documents.supported_formats", [".pdf", ".docx", ".doc", ".txt", ".md"])

def get_max_file_size() -> int:
    """Get maximum file size"""
    return config.get("documents.max_file_size", 50 * 1024 * 1024)

def get_temp_directory() -> str:
    """Get temporary directory"""
    return config.get("documents.temp_directory", "./temp_docs")

def get_log_level() -> str:
    """Get logging level"""
    return config.get("logging.level", "INFO")

def get_ollama_config() -> Dict[str, Any]:
    """Get Ollama configuration"""
    return config.get("ollama", {})

def get_ollama_base_url() -> str:
    """Get Ollama base URL"""
    return config.get("ollama.base_url", "http://192.168.1.89:11434")

def get_ollama_embedding_model() -> str:
    """Get Ollama embedding model"""
    return config.get("ollama.embedding_model", "nomic-embed-text:latest")

def get_ollama_llm_model() -> str:
    """Get Ollama LLM model"""
    return config.get("ollama.llm_model", "llama3:latest")


# Environment setup helpers
def setup_environment():
    """Setup environment for the RAG system"""
    # Create necessary directories
    temp_dir = Path(get_temp_directory())
    temp_dir.mkdir(exist_ok=True)
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=getattr(logging, get_log_level().upper()),
        format=config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Create ChromaDB directory if using ChromaDB
    if get_vector_store_type() == "chroma":
        chroma_dir = Path(config.get("vector_store.chroma.persist_directory", "./chroma_db"))
        chroma_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    # Print current configuration
    print("Current Configuration:")
    print("=" * 50)
    
    config_dict = config.to_dict()
    import json
    print(json.dumps(config_dict, indent=2))
    
    print("\nEnvironment Variables:")
    print("=" * 50)
    for env_var in ENV_MAPPINGS.keys():
        value = os.getenv(env_var)
        if value:
            print(f"{env_var}: {value}")
    
    print("\nConvenience Functions:")
    print("=" * 50)
    print(f"Vector Store Type: {get_vector_store_type()}")
    print(f"Embedding Model: {get_embedding_model()}")
    print(f"Chunking Strategy: {get_chunking_strategy()}")
    print(f"Collection Name: {get_collection_name()}")
    print(f"RAG Top-K: {get_rag_top_k()}")
    print(f"Supported Formats: {get_supported_formats()}")
    print(f"Log Level: {get_log_level()}")

