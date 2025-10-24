# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Docling
RUN pip install docling docling-core docling-langchain

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY README.md .

# Create necessary directories
RUN mkdir -p documents data temp_docs chroma_db

# Set environment variables
ENV PYTHONPATH=/app
ENV VECTOR_STORE_TYPE=milvus
ENV EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV CHUNKING_STRATEGY=hierarchical
ENV LOG_LEVEL=INFO

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.cli", "info"]

