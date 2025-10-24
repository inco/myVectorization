# Library Vectorization RAG System

A comprehensive system for converting document libraries into vectorized RAG (Retrieval-Augmented Generation) systems using Docling for document processing.

## Features

- **Document Processing**: Support for PDF, DOCX, TXT, and Markdown files using Docling
- **Advanced Chunking**: Hierarchical and hybrid chunking strategies to preserve document structure
- **Multiple Vector Stores**: Integration with Milvus, Qdrant, and ChromaDB
- **HuggingFace Embeddings**: Support for various embedding models
- **RAG Pipeline**: Complete question-answering system with document retrieval
- **CLI Interface**: Easy-to-use command-line interface
- **Interactive Q&A**: Real-time question answering sessions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd myLibraryScan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Docling (if not already installed):
```bash
pip install docling docling-core docling-langchain
```

## Quick Start

### 1. Vectorize Documents

```bash
# Vectorize documents in a directory
python -m src.cli vectorize ./documents --vector-store milvus --collection-name my_library

# With custom embedding model
python -m src.cli vectorize ./documents --embedding-model "BAAI/bge-large-en-v1.5" --chunking-strategy hybrid
```

### 2. Interactive Q&A

```bash
# Start interactive Q&A session
python -m src.cli ask --vector-store milvus --collection-name my_library
```

### 3. Search Documents

```bash
# Search for relevant documents
python -m src.cli search --vector-store milvus --collection-name my_library
```

### 4. Single Query

```bash
# Ask a single question
python -m src.cli query "What is machine learning?" --vector-store milvus --collection-name my_library
```

## Configuration

The system can be configured through environment variables or configuration files:

### Environment Variables

```bash
export VECTOR_STORE_TYPE=milvus
export EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
export CHUNKING_STRATEGY=hierarchical
export RAG_TOP_K=5
export LOG_LEVEL=INFO
```

### Vector Store Setup

#### Milvus
```bash
# Start Milvus using Docker
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

#### Qdrant
```bash
# Start Qdrant using Docker
docker run -d --name qdrant \
  -p 6333:6333 \
  qdrant/qdrant:latest
```

#### ChromaDB
ChromaDB runs locally and doesn't require external setup.

## Usage Examples

### Python API

```python
from library_vectorizer import LibraryVectorizer
from rag_pipeline import SimpleRAG

# Vectorize documents
vectorizer = LibraryVectorizer(
    vector_store_type="milvus",
    embedding_model="BAAI/bge-small-en-v1.5",
    chunking_strategy="hierarchical"
)

result = vectorizer.vectorize_library("./documents", "my_library")
print(f"Vectorized {result['total_chunks']} chunks")

# Use RAG system
rag = SimpleRAG(vector_store_type="milvus", collection_name="my_library")

# Ask questions
answer = rag.ask("What is the main topic of the documents?")
print(answer)

# Search documents
docs = rag.search("machine learning", top_k=5)
for doc in docs:
    print(f"Source: {doc['metadata']['source_file']}")
    print(f"Score: {doc['score']}")
    print(f"Content: {doc['text'][:100]}...")
```

### Advanced Usage

```python
from rag_pipeline import RAGPipeline

# Advanced RAG with custom configuration
pipeline = RAGPipeline(
    vector_store_type="milvus",
    embedding_model="BAAI/bge-large-en-v1.5",
    collection_name="my_library",
    llm_model="gpt-3.5-turbo"  # Optional LLM integration
)

# Get detailed response
response = pipeline.ask_question("Compare different ML approaches", top_k=10)
print(f"Answer: {response['answer']}")
print(f"Retrieved {response['num_documents']} documents")

# Show retrieved documents
for doc in response['retrieved_documents']:
    print(f"Source: {doc['metadata']['source_file']}")
    print(f"Relevance: {doc['score']}")
```

## Supported Document Formats

- **PDF**: Advanced PDF parsing with layout understanding
- **Microsoft Word**: .docx and .doc files
- **Plain Text**: .txt files
- **Markdown**: .md files

## Chunking Strategies

### Hierarchical Chunking
Preserves document structure by creating chunks based on headings and sections.

### Hybrid Chunking
Combines semantic and structural chunking for optimal retrieval performance.

## Vector Stores

### Milvus
- High-performance vector database
- Supports various index types (IVF_FLAT, HNSW, etc.)
- Scalable for large datasets

### Qdrant
- Vector similarity search engine
- RESTful API
- Good for production deployments

### ChromaDB
- Open-source embedding database
- Easy to use and deploy
- Good for development and small datasets

## Embedding Models

The system supports various HuggingFace embedding models:

- `BAAI/bge-small-en-v1.5` (384 dimensions) - Default, fast
- `BAAI/bge-base-en-v1.5` (768 dimensions) - Balanced
- `BAAI/bge-large-en-v1.5` (1024 dimensions) - High quality
- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) - Lightweight

## CLI Commands

### Vectorize Documents
```bash
python -m src.cli vectorize <input_path> [options]
```

Options:
- `--vector-store`: Vector store type (milvus, qdrant, chroma)
- `--embedding-model`: HuggingFace embedding model
- `--chunking-strategy`: Chunking strategy (hierarchical, hybrid)
- `--collection-name`: Collection name in vector store
- `--force`: Force recreation of collection

### Interactive Q&A
```bash
python -m src.cli ask [options]
```

### Search Documents
```bash
python -m src.cli search [options]
```

### Single Query
```bash
python -m src.cli query <question> [options]
```

### Collection Statistics
```bash
python -m src.cli stats [options]
```

### System Information
```bash
python -m src.cli info
```

## Examples

See the `examples/` directory for detailed usage examples:

- `basic_examples.py`: Basic usage examples
- `advanced_examples.py`: Advanced configuration examples

## Troubleshooting

### Common Issues

1. **Vector Store Connection Error**
   - Ensure the vector store is running
   - Check connection parameters (host, port)
   - Verify collection exists

2. **Document Processing Error**
   - Check file format support
   - Verify file permissions
   - Ensure sufficient disk space

3. **Embedding Generation Error**
   - Check internet connection for model download
   - Verify sufficient memory
   - Try a different embedding model

### Logging

Enable verbose logging:
```bash
python -m src.cli vectorize ./documents --verbose
```

Or set log level:
```bash
export LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Docling](https://github.com/docling-project/docling) for document processing
- [HuggingFace](https://huggingface.co/) for embedding models
- [Milvus](https://milvus.io/), [Qdrant](https://qdrant.tech/), and [ChromaDB](https://www.trychroma.com/) for vector storage

