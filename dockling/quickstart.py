"""
Quick Start Guide for Library Vectorization RAG System

This script provides a quick way to get started with the system.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("Library Vectorization RAG System")
    print("Quick Start Guide")
    print("=" * 60)

def check_requirements():
    """Check if requirements are met"""
    print("\n1. Checking Requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print("✅ Python version OK")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    print("✅ requirements.txt found")
    
    return True

def install_dependencies():
    """Install dependencies"""
    print("\n2. Installing Dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Setup environment"""
    print("\n3. Setting up Environment...")
    
    # Create directories
    directories = ["documents", "data", "temp_docs", "chroma_db"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Copy env file if needed
    if not Path(".env").exists() and Path("env.example").exists():
        import shutil
        shutil.copy("env.example", ".env")
        print("✅ Created .env file")
    
    return True

def start_vector_store():
    """Start vector store"""
    print("\n4. Starting Vector Store...")
    
    print("Choose a vector store:")
    print("1. Milvus (recommended for production)")
    print("2. Qdrant (good alternative)")
    print("3. ChromaDB (easiest to start)")
    print("4. Skip (use existing)")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        print("Starting Milvus with Docker...")
        try:
            subprocess.run([
                "docker", "run", "-d", "--name", "milvus-standalone",
                "-p", "19530:19530", "-p", "9091:9091",
                "milvusdb/milvus:latest"
            ], check=True, capture_output=True)
            print("✅ Milvus started")
            return "milvus"
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Failed to start Milvus. Make sure Docker is installed.")
            return None
    
    elif choice == "2":
        print("Starting Qdrant with Docker...")
        try:
            subprocess.run([
                "docker", "run", "-d", "--name", "qdrant",
                "-p", "6333:6333", "qdrant/qdrant:latest"
            ], check=True, capture_output=True)
            print("✅ Qdrant started")
            return "qdrant"
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Failed to start Qdrant. Make sure Docker is installed.")
            return None
    
    elif choice == "3":
        print("✅ ChromaDB will be used (no setup needed)")
        return "chroma"
    
    else:
        print("✅ Skipping vector store setup")
        return None

def add_sample_documents():
    """Add sample documents"""
    print("\n5. Adding Sample Documents...")
    
    documents_dir = Path("documents")
    
    # Create sample document
    sample_content = """
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

## Key Concepts
- **Training Data**: The data used to train the model
- **Features**: Input variables used by the model
- **Labels**: Output variables the model tries to predict
- **Model**: The algorithm that makes predictions
- **Accuracy**: How well the model performs on test data
"""
    
    sample_file = documents_dir / "ml_overview.md"
    sample_file.write_text(sample_content)
    print(f"✅ Created sample document: {sample_file}")
    
    print(f"\n📁 Add your own documents to the '{documents_dir}' directory")
    print("Supported formats: PDF, DOCX, TXT, MD")

def vectorize_documents(vector_store_type):
    """Vectorize documents"""
    print("\n6. Vectorizing Documents...")
    
    if not vector_store_type:
        vector_store_type = "chroma"  # Default to ChromaDB
    
    try:
        cmd = [
            sys.executable, "-m", "src.cli", "vectorize", "./documents",
            "--vector-store", vector_store_type,
            "--collection-name", "quickstart_library"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Documents vectorized successfully")
            return True
        else:
            print(f"❌ Vectorization failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during vectorization: {e}")
        return False

def test_rag_system():
    """Test RAG system"""
    print("\n7. Testing RAG System...")
    
    vector_store_type = "chroma"  # Default
    
    try:
        # Test a simple query
        cmd = [
            sys.executable, "-m", "src.cli", "query",
            "What is machine learning?",
            "--vector-store", vector_store_type,
            "--collection-name", "quickstart_library"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ RAG system working correctly")
            print("Sample answer:")
            print(result.stdout)
            return True
        else:
            print(f"❌ RAG test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        return False

def print_next_steps():
    """Print next steps"""
    print("\n" + "=" * 60)
    print("🎉 QUICK START COMPLETED!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. 📚 Add more documents to the 'documents' directory")
    print("2. 🔄 Re-vectorize: python -m src.cli vectorize ./documents")
    print("3. 💬 Interactive Q&A: python -m src.cli ask")
    print("4. 🔍 Search documents: python -m src.cli search")
    print("5. 📊 View stats: python -m src.cli stats")
    
    print("\nAdvanced usage:")
    print("• Use different vector stores: --vector-store milvus|qdrant|chroma")
    print("• Custom embedding models: --embedding-model 'BAAI/bge-large-en-v1.5'")
    print("• Different chunking: --chunking-strategy hybrid")
    
    print("\nDocumentation:")
    print("• README.md - Full documentation")
    print("• examples/ - Usage examples")
    print("• src/config.py - Configuration options")

def main():
    """Main quick start function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        return
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Failed to setup environment.")
        return
    
    # Start vector store
    vector_store_type = start_vector_store()
    
    # Add sample documents
    add_sample_documents()
    
    # Vectorize documents
    if not vectorize_documents(vector_store_type):
        print("\n❌ Failed to vectorize documents.")
        return
    
    # Test RAG system
    if not test_rag_system():
        print("\n❌ RAG system test failed.")
        return
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()

