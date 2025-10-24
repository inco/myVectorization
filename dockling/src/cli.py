"""
Command Line Interface for Library Vectorization RAG System

Provides easy-to-use CLI commands for vectorizing document libraries
and interacting with the RAG system.
"""

import click
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .library_vectorizer import LibraryVectorizer
from .rag_pipeline import SimpleRAG, RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console = Console()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Library Vectorization RAG System CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--vector-store', '-vs', 
              type=click.Choice(['milvus', 'qdrant', 'chroma']), 
              default='milvus',
              help='Vector store type')
@click.option('--embedding-model', '-em', 
              default='BAAI/bge-small-en-v1.5',
              help='HuggingFace embedding model')
@click.option('--chunking-strategy', '-cs',
              type=click.Choice(['hierarchical', 'hybrid']),
              default='hierarchical',
              help='Document chunking strategy')
@click.option('--use-ollama', is_flag=True,
              help='Use Ollama for embeddings instead of HuggingFace')
@click.option('--ollama-url', default='http://192.168.1.89:11434',
              help='Ollama server URL')
@click.option('--ollama-embedding-model', default='nomic-embed-text:latest',
              help='Ollama embedding model name')
@click.option('--collection-name', '-cn',
              default='docling_rag',
              help='Collection name in vector store')
@click.option('--force', '-f', is_flag=True,
              help='Force recreation of collection')
@click.option('--recursive', '-r', is_flag=True, default=True,
              help='Scan subfolders recursively')
@click.option('--force-reprocess', is_flag=True,
              help='Force reprocessing of already processed files')
@click.option('--scan-mode', is_flag=True,
              help='Use folder scanning mode (checks for already processed files)')
def vectorize(input_path, vector_store, embedding_model, chunking_strategy, use_ollama, ollama_url, ollama_embedding_model, collection_name, force, recursive, force_reprocess, scan_mode):
    """Vectorize documents from input path"""
    
    mode_text = "Folder Scan" if scan_mode else "Standard"
    embedding_source = "Ollama" if use_ollama else "HuggingFace"
    embedding_info = f"{ollama_embedding_model} ({ollama_url})" if use_ollama else embedding_model
    
    console.print(Panel.fit(
        f"[bold blue]Vectorizing Documents[/bold blue]\n"
        f"Input: {input_path}\n"
        f"Mode: {mode_text}\n"
        f"Vector Store: {vector_store}\n"
        f"Embedding Source: {embedding_source}\n"
        f"Embedding Model: {embedding_info}\n"
        f"Chunking: {chunking_strategy}\n"
        f"Collection: {collection_name}\n"
        f"Recursive: {recursive}\n"
        f"Force Reprocess: {force_reprocess}",
        title="Configuration"
    ))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialize vectorizer
            task = progress.add_task("Initializing vectorizer...", total=None)
            vectorizer = LibraryVectorizer(
                vector_store_type=vector_store,
                embedding_model=embedding_model,
                chunking_strategy=chunking_strategy,
                use_ollama=use_ollama,
                ollama_url=ollama_url,
                ollama_embedding_model=ollama_embedding_model
            )
            progress.update(task, description="Vectorizer initialized")
            
            # Vectorize library
            progress.update(task, description="Processing documents...")
            
            if scan_mode:
                # Use folder scanning mode
                result = vectorizer.scan_and_vectorize_folder(
                    input_path, 
                    collection_name, 
                    recursive=recursive,
                    force_reprocess=force_reprocess
                )
            else:
                # Use standard mode
                result = vectorizer.vectorize_library(input_path, collection_name)
            
            progress.update(task, description="Vectorization completed!")
        
        # Display results
        table = Table(title="Vectorization Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Chunks", str(result['total_chunks']))
        table.add_row("Collection Name", result['collection_name'])
        table.add_row("Vector Store", result['vector_store'])
        table.add_row("Embedding Model", result['embedding_model'])
        table.add_row("Chunking Strategy", result['chunking_strategy'])
        
        # Add additional info for scan mode
        if scan_mode and 'processed_files' in result:
            table.add_row("Processed Files", str(result['processed_files']))
            table.add_row("Skipped Files", str(result['skipped_files']))
        
        console.print(table)
        console.print(Panel.fit("[bold green]Vectorization completed successfully![/bold green]"))
        
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Error: {str(e)}[/bold red]", title="Error"))
        raise click.Abort()


@cli.command()
@click.option('--vector-store', '-vs',
              type=click.Choice(['milvus', 'qdrant', 'chroma']),
              default='milvus',
              help='Vector store type')
@click.option('--collection-name', '-cn',
              default='docling_rag',
              help='Collection name in vector store')
@click.option('--top-k', '-k',
              default=5,
              help='Number of documents to retrieve')
def ask(vector_store, collection_name, top_k):
    """Interactive Q&A session with the vectorized library"""
    
    console.print(Panel.fit(
        f"[bold blue]RAG Q&A Session[/bold blue]\n"
        f"Vector Store: {vector_store}\n"
        f"Collection: {collection_name}\n"
        f"Top-K: {top_k}",
        title="Configuration"
    ))
    
    try:
        # Initialize RAG pipeline
        rag = SimpleRAG(vector_store_type=vector_store, collection_name=collection_name)
        
        console.print(Panel.fit(
            "[bold green]RAG system initialized![/bold green]\n"
            "Type 'quit' or 'exit' to end the session.",
            title="Ready"
        ))
        
        while True:
            try:
                # Get user question
                question = click.prompt('\nYour question', type=str)
                
                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("[bold yellow]Goodbye![/bold yellow]")
                    break
                
                if not question.strip():
                    continue
                
                # Get answer
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Processing question...", total=None)
                    answer = rag.ask(question, top_k)
                    progress.update(task, description="Answer generated!")
                
                # Display answer
                console.print(Panel(answer, title="Answer", border_style="green"))
                
                # Show retrieved documents
                docs = rag.search(question, top_k)
                if docs:
                    table = Table(title="Retrieved Documents")
                    table.add_column("Source", style="cyan")
                    table.add_column("Score", style="yellow")
                    table.add_column("Preview", style="white")
                    
                    for doc in docs[:3]:  # Show top 3
                        source = doc['metadata'].get('source_file', 'Unknown')
                        score = f"{doc['score']:.3f}"
                        preview = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
                        
                        table.add_row(source, score, preview)
                    
                    console.print(table)
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break
            except Exception as e:
                console.print(Panel.fit(f"[bold red]Error: {str(e)}[/bold red]", title="Error"))
                
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Failed to initialize RAG system: {str(e)}[/bold red]", title="Error"))
        raise click.Abort()


@cli.command()
@click.option('--vector-store', '-vs',
              type=click.Choice(['milvus', 'qdrant', 'chroma']),
              default='milvus',
              help='Vector store type')
@click.option('--collection-name', '-cn',
              default='docling_rag',
              help='Collection name in vector store')
@click.option('--top-k', '-k',
              default=5,
              help='Number of documents to retrieve')
def search(vector_store, collection_name, top_k):
    """Search for documents in the vectorized library"""
    
    console.print(Panel.fit(
        f"[bold blue]Document Search[/bold blue]\n"
        f"Vector Store: {vector_store}\n"
        f"Collection: {collection_name}\n"
        f"Top-K: {top_k}",
        title="Configuration"
    ))
    
    try:
        # Initialize RAG pipeline
        rag = SimpleRAG(vector_store_type=vector_store, collection_name=collection_name)
        
        while True:
            try:
                # Get search query
                query = click.prompt('\nSearch query', type=str)
                
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("[bold yellow]Goodbye![/bold yellow]")
                    break
                
                if not query.strip():
                    continue
                
                # Search documents
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Searching documents...", total=None)
                    docs = rag.search(query, top_k)
                    progress.update(task, description="Search completed!")
                
                # Display results
                if docs:
                    table = Table(title=f"Search Results for: '{query}'")
                    table.add_column("Rank", style="cyan")
                    table.add_column("Source", style="green")
                    table.add_column("Score", style="yellow")
                    table.add_column("Content", style="white")
                    
                    for i, doc in enumerate(docs, 1):
                        source = doc['metadata'].get('source_file', 'Unknown')
                        score = f"{doc['score']:.3f}"
                        content = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                        
                        table.add_row(str(i), source, score, content)
                    
                    console.print(table)
                else:
                    console.print(Panel.fit("[bold yellow]No documents found for your query.[/bold yellow]"))
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break
            except Exception as e:
                console.print(Panel.fit(f"[bold red]Error: {str(e)}[/bold red]", title="Error"))
                
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Failed to initialize search system: {str(e)}[/bold red]", title="Error"))
        raise click.Abort()


@cli.command()
@click.option('--vector-store', '-vs',
              type=click.Choice(['milvus', 'qdrant', 'chroma']),
              default='milvus',
              help='Vector store type')
@click.option('--collection-name', '-cn',
              default='docling_rag',
              help='Collection name in vector store')
def stats(vector_store, collection_name):
    """Show statistics about the vectorized collection"""
    
    console.print(Panel.fit(
        f"[bold blue]Collection Statistics[/bold blue]\n"
        f"Vector Store: {vector_store}\n"
        f"Collection: {collection_name}",
        title="Configuration"
    ))
    
    try:
        # Initialize RAG pipeline
        pipeline = RAGPipeline(vector_store_type=vector_store, collection_name=collection_name)
        
        # Get stats
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Retrieving statistics...", total=None)
            stats = pipeline.get_collection_stats()
            progress.update(task, description="Statistics retrieved!")
        
        # Display stats
        table = Table(title="Collection Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Error retrieving statistics: {str(e)}[/bold red]", title="Error"))
        raise click.Abort()


@cli.command()
@click.argument('question')
@click.option('--vector-store', '-vs',
              type=click.Choice(['milvus', 'qdrant', 'chroma']),
              default='milvus',
              help='Vector store type')
@click.option('--collection-name', '-cn',
              default='docling_rag',
              help='Collection name in vector store')
@click.option('--top-k', '-k',
              default=5,
              help='Number of documents to retrieve')
def query(question, vector_store, collection_name, top_k):
    """Ask a single question and get an answer"""
    
    console.print(Panel.fit(
        f"[bold blue]Single Query[/bold blue]\n"
        f"Question: {question}\n"
        f"Vector Store: {vector_store}\n"
        f"Collection: {collection_name}\n"
        f"Top-K: {top_k}",
        title="Configuration"
    ))
    
    try:
        # Initialize RAG pipeline
        rag = SimpleRAG(vector_store_type=vector_store, collection_name=collection_name)
        
        # Get answer
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing question...", total=None)
            answer = rag.ask(question, top_k)
            progress.update(task, description="Answer generated!")
        
        # Display answer
        console.print(Panel(answer, title="Answer", border_style="green"))
        
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Error: {str(e)}[/bold red]", title="Error"))
        raise click.Abort()


@cli.command()
@click.option('--vector-store', '-vs',
              type=click.Choice(['milvus', 'qdrant', 'chroma']),
              default='milvus',
              help='Vector store type')
@click.option('--collection-name', '-cn',
              default='docling_rag',
              help='Collection name in vector store')
def processed_files(vector_store, collection_name):
    """Show list of already processed files"""
    
    console.print(Panel.fit(
        f"[bold blue]Processed Files[/bold blue]\n"
        f"Vector Store: {vector_store}\n"
        f"Collection: {collection_name}",
        title="Configuration"
    ))
    
    try:
        # Initialize vectorizer
        vectorizer = LibraryVectorizer(
            vector_store_type=vector_store,
            embedding_model="BAAI/bge-small-en-v1.5",
            chunking_strategy="hierarchical",
            use_ollama=False
        )
        
        # Get processed files
        processed_files = vectorizer.get_processed_files()
        
        if processed_files:
            table = Table(title="Processed Files")
            table.add_column("File Path", style="cyan")
            table.add_column("File Name", style="green")
            
            for file_path in sorted(processed_files):
                file_name = Path(file_path).name
                table.add_row(file_path, file_name)
            
            console.print(table)
            console.print(f"\nTotal processed files: {len(processed_files)}")
        else:
            console.print(Panel.fit("[bold yellow]No processed files found.[/bold yellow]"))
        
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Error: {str(e)}[/bold red]", title="Error"))
        raise click.Abort()


@cli.command()
def info():
    """Show information about the Library Vectorization RAG System"""
    
    info_text = """
[bold blue]Library Vectorization RAG System[/bold blue]

This system allows you to vectorize document libraries and create a RAG (Retrieval-Augmented Generation) system using Docling for document processing.

[bold green]Features:[/bold green]
• Document processing with Docling (PDF, DOCX, TXT, MD)
• Hierarchical and hybrid chunking strategies
• Multiple vector store support (Milvus, Qdrant, ChromaDB)
• HuggingFace embedding models
• Interactive Q&A sessions
• Document search capabilities

[bold green]Supported Formats:[/bold green]
• PDF documents
• Microsoft Word documents (.docx, .doc)
• Plain text files (.txt)
• Markdown files (.md)
• EPUB e-books (.epub)
• FictionBook files (.fb2)

[bold green]Vector Stores:[/bold green]
• Milvus - High-performance vector database
• Qdrant - Vector similarity search engine
• ChromaDB - Open-source embedding database

[bold green]Usage Examples:[/bold green]
• Vectorize documents: [cyan]python -m src.cli vectorize ./documents[/cyan]
• Scan folder (skip processed): [cyan]python -m src.cli vectorize ./documents --scan-mode[/cyan]
• Force reprocess: [cyan]python -m src.cli vectorize ./documents --scan-mode --force-reprocess[/cyan]
• Interactive Q&A: [cyan]python -m src.cli ask[/cyan]
• Search documents: [cyan]python -m src.cli search[/cyan]
• Single query: [cyan]python -m src.cli query "What is machine learning?"[/cyan]
• Show stats: [cyan]python -m src.cli stats[/cyan]
• Show processed files: [cyan]python -m src.cli processed-files[/cyan]

[bold yellow]Note:[/bold yellow] Make sure to have the appropriate vector store running before using the system.
"""
    
    console.print(Panel.fit(info_text, title="Library Vectorization RAG System"))


if __name__ == '__main__':
    cli()

