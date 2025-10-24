"""
Setup script for the Library Vectorization RAG System

This script helps set up the environment and dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n=== Installing Python Dependencies ===")
    
    # Install basic requirements
    if not run_command("pip install -r requirements.txt", "Installing basic requirements"):
        return False
    
    # Install Docling
    if not run_command("pip install docling docling-core docling-langchain", "Installing Docling"):
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("\n=== Setting up directories ===")
    
    directories = [
        "documents",
        "data", 
        "temp_docs",
        "chroma_db",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def setup_environment():
    """Setup environment configuration"""
    print("\n=== Setting up environment ===")
    
    # Copy example env file if .env doesn't exist
    if not Path(".env").exists() and Path("env.example").exists():
        import shutil
        shutil.copy("env.example", ".env")
        print("✓ Created .env file from template")
    
    return True

def check_vector_stores():
    """Check if vector stores are available"""
    print("\n=== Checking Vector Store Availability ===")
    
    # Check Milvus
    try:
        import pymilvus
        print("✓ Milvus client available")
    except ImportError:
        print("⚠ Milvus client not installed (optional)")
    
    # Check Qdrant
    try:
        import qdrant_client
        print("✓ Qdrant client available")
    except ImportError:
        print("⚠ Qdrant client not installed (optional)")
    
    # Check ChromaDB
    try:
        import chromadb
        print("✓ ChromaDB available")
    except ImportError:
        print("⚠ ChromaDB not installed (optional)")
    
    return True

def test_installation():
    """Test the installation"""
    print("\n=== Testing Installation ===")
    
    try:
        # Test imports
        from src.library_vectorizer import LibraryVectorizer
        from src.rag_pipeline import SimpleRAG
        from src.config import get_config
        print("✓ Core modules imported successfully")
        
        # Test configuration
        config = get_config()
        print(f"✓ Configuration loaded: {config.get('vector_store.type', 'milvus')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("SETUP COMPLETED!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Start a vector store (choose one):")
    print("   • Milvus: docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest")
    print("   • Qdrant: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest")
    print("   • ChromaDB: No setup needed (runs locally)")
    
    print("\n2. Add documents to the 'documents' directory")
    
    print("\n3. Vectorize your documents:")
    print("   python -m src.cli vectorize ./documents")
    
    print("\n4. Start interactive Q&A:")
    print("   python -m src.cli ask")
    
    print("\n5. Or use Docker Compose:")
    print("   docker-compose up -d")
    
    print("\nFor more information, see README.md")

def main():
    """Main setup function"""
    print("Library Vectorization RAG System - Setup Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Dependency installation failed")
        sys.exit(1)
    
    # Setup directories
    if not setup_directories():
        print("\n✗ Directory setup failed")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("\n✗ Environment setup failed")
        sys.exit(1)
    
    # Check vector stores
    check_vector_stores()
    
    # Test installation
    if not test_installation():
        print("\n✗ Installation test failed")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()

