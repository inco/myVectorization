"""
Additional document format processors for EPUB and FB2 files

This module provides processors for formats not directly supported by Docling.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

logger = logging.getLogger(__name__)


class EPUBProcessor:
    """Processor for EPUB files"""
    
    def __init__(self):
        try:
            import ebooklib
            from ebooklib import epub
            self.epub = epub
            self.ebooklib = ebooklib
        except ImportError:
            logger.error("ebooklib not installed. Install with: pip install ebooklib")
            raise
    
    def extract_text(self, epub_path: str) -> str:
        """
        Extract text content from EPUB file
        
        Args:
            epub_path: Path to EPUB file
            
        Returns:
            Extracted text content
        """
        try:
            book = self.epub.read_epub(epub_path)
            
            # Extract text from all chapters
            text_content = []
            
            # Get book metadata
            title = book.get_metadata('DC', 'title')
            if title:
                text_content.append(f"# {title[0][0]}")
            
            author = book.get_metadata('DC', 'creator')
            if author:
                text_content.append(f"**Author:** {author[0][0]}")
            
            text_content.append("")  # Empty line
            
            # Extract content from all items
            for item in book.get_items():
                if item.get_type() == self.ebooklib.ITEM_DOCUMENT:
                    # Convert HTML to text (simple approach)
                    content = self._html_to_text(item.get_content().decode('utf-8'))
                    if content.strip():
                        text_content.append(content)
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error processing EPUB file {epub_path}: {e}")
            return ""
    
    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML content to plain text"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean up
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            logger.warning("BeautifulSoup not available, using basic HTML stripping")
            # Basic HTML tag removal
            import re
            clean = re.compile('<.*?>')
            return re.sub(clean, '', html_content)
        except Exception as e:
            logger.error(f"Error converting HTML to text: {e}")
            return html_content


class FB2Processor:
    """Processor for FB2 (FictionBook) files"""
    
    def extract_text(self, fb2_path: str) -> str:
        """
        Extract text content from FB2 file
        
        Args:
            fb2_path: Path to FB2 file
            
        Returns:
            Extracted text content
        """
        try:
            import xml.etree.ElementTree as ET
            
            # Parse FB2 XML
            tree = ET.parse(fb2_path)
            root = tree.getroot()
            
            # Define namespace
            ns = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
            
            text_content = []
            
            # Extract title
            title_elem = root.find('.//fb:book-title', ns)
            if title_elem is not None:
                text_content.append(f"# {title_elem.text}")
            
            # Extract author
            author_elem = root.find('.//fb:author/fb:first-name', ns)
            if author_elem is not None:
                author_name = author_elem.text
                last_name_elem = root.find('.//fb:author/fb:last-name', ns)
                if last_name_elem is not None:
                    author_name += f" {last_name_elem.text}"
                text_content.append(f"**Author:** {author_name}")
            
            text_content.append("")  # Empty line
            
            # Extract body content
            body_elem = root.find('.//fb:body', ns)
            if body_elem is not None:
                # Extract all text from body
                body_text = self._extract_text_from_element(body_elem)
                if body_text.strip():
                    text_content.append(body_text)
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error processing FB2 file {fb2_path}: {e}")
            return ""
    
    def _extract_text_from_element(self, element) -> str:
        """Recursively extract text from XML element"""
        text_parts = []
        
        if element.text:
            text_parts.append(element.text.strip())
        
        for child in element:
            child_text = self._extract_text_from_element(child)
            if child_text.strip():
                text_parts.append(child_text)
        
        if element.tail:
            text_parts.append(element.tail.strip())
        
        return " ".join(text_parts)


class FileTracker:
    """Track processed files to avoid reprocessing"""
    
    def __init__(self, vector_store_type: str, collection_name: str):
        self.vector_store_type = vector_store_type
        self.collection_name = collection_name
        self.processed_files = set()
        self._load_processed_files()
    
    def _load_processed_files(self):
        """Load list of already processed files from vector store"""
        try:
            if self.vector_store_type == "qdrant":
                self._load_from_qdrant()
            elif self.vector_store_type == "milvus":
                self._load_from_milvus()
            elif self.vector_store_type == "chroma":
                self._load_from_chroma()
            
            logger.info(f"Loaded {len(self.processed_files)} processed files")
            
        except Exception as e:
            logger.warning(f"Could not load processed files list: {e}")
            self.processed_files = set()
    
    def _load_from_qdrant(self):
        """Load processed files from Qdrant"""
        try:
            from qdrant_client import QdrantClient
            import os
            
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            api_key = os.getenv("QDRANT_API_KEY")
            
            if api_key:
                client = QdrantClient(host=host, port=port, api_key=api_key, https=False)
            else:
                client = QdrantClient(host=host, port=port, https=False)
            
            # Get all points to extract file information
            points = client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your needs
                with_payload=True
            )[0]
            
            for point in points:
                if 'source_file' in point.payload:
                    self.processed_files.add(point.payload['source_file'])
                    
        except Exception as e:
            logger.error(f"Error loading from Qdrant: {e}")
    
    def _load_from_milvus(self):
        """Load processed files from Milvus"""
        try:
            from pymilvus import connections, Collection
            
            connections.connect("default", host="localhost", port=19530)
            collection = Collection(self.collection_name)
            collection.load()
            
            # Query all documents to get file information
            results = collection.query(
                expr="",  # Empty expression to get all
                output_fields=["source_file"]
            )
            
            for result in results:
                if 'source_file' in result:
                    self.processed_files.add(result['source_file'])
                    
        except Exception as e:
            logger.error(f"Error loading from Milvus: {e}")
    
    def _load_from_chroma(self):
        """Load processed files from ChromaDB"""
        try:
            import chromadb
            
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection(self.collection_name)
            
            # Get all documents
            results = collection.get(include=["metadatas"])
            
            for metadata in results['metadatas']:
                if 'source_file' in metadata:
                    self.processed_files.add(metadata['source_file'])
                    
        except Exception as e:
            logger.error(f"Error loading from ChromaDB: {e}")
    
    def is_processed(self, file_path: str) -> bool:
        """Check if file has already been processed"""
        return file_path in self.processed_files
    
    def mark_processed(self, file_path: str):
        """Mark file as processed"""
        self.processed_files.add(file_path)
    
    def get_file_hash(self, file_path: str) -> str:
        """Get MD5 hash of file for change detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""


class ExtendedDocumentProcessor:
    """Extended document processor with EPUB/FB2 support and file tracking"""
    
    def __init__(self, vector_store_type: str, collection_name: str):
        self.vector_store_type = vector_store_type
        self.collection_name = collection_name
        self.file_tracker = FileTracker(vector_store_type, collection_name)
        self.epub_processor = EPUBProcessor()
        self.fb2_processor = FB2Processor()
        
        # Supported formats
        self.supported_formats = {
            '.pdf': self._process_with_docling,
            '.docx': self._process_with_docling,
            '.doc': self._process_with_docling,
            '.txt': self._process_with_docling,
            '.md': self._process_with_docling,
            '.epub': self._process_epub,
            '.fb2': self._process_fb2
        }
    
    def scan_folder(self, folder_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Scan folder for documents and process them
        
        Args:
            folder_path: Path to folder to scan
            recursive: Whether to scan subfolders recursively
            
        Returns:
            List of processed documents
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            logger.error(f"Folder does not exist: {folder_path}")
            return []
        
        logger.info(f"Scanning folder: {folder_path}")
        
        # Find all supported files
        files_to_process = []
        
        if recursive:
            for ext in self.supported_formats.keys():
                files_to_process.extend(folder_path.rglob(f"*{ext}"))
        else:
            for ext in self.supported_formats.keys():
                files_to_process.extend(folder_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Filter out already processed files
        new_files = []
        for file_path in files_to_process:
            if not self.file_tracker.is_processed(str(file_path)):
                new_files.append(file_path)
            else:
                logger.info(f"Skipping already processed file: {file_path}")
        
        logger.info(f"Found {len(new_files)} new files to process")
        
        # Process new files
        processed_docs = []
        for file_path in new_files:
            try:
                doc = self.process_file(str(file_path))
                if doc:
                    processed_docs.append(doc)
                    self.file_tracker.mark_processed(str(file_path))
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        return processed_docs
    
    def process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single file
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Processed document data or None if failed
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext not in self.supported_formats:
            logger.warning(f"Unsupported file format: {ext}")
            return None
        
        logger.info(f"Processing file: {file_path}")
        
        try:
            processor = self.supported_formats[ext]
            content = processor(str(file_path))
            
            if not content.strip():
                logger.warning(f"No content extracted from {file_path}")
                return None
            
            return {
                'file_path': str(file_path),
                'content': content,
                'file_type': ext,
                'file_size': file_path.stat().st_size,
                'file_hash': self.file_tracker.get_file_hash(str(file_path))
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    def _process_with_docling(self, file_path: str) -> str:
        """Process file using Docling"""
        try:
            from docling.document_converter import DocumentConverter
            
            converter = DocumentConverter()
            result = converter.convert(file_path)
            
            # Export to markdown
            return result.document.export_to_markdown()
            
        except Exception as e:
            logger.error(f"Error processing with Docling: {e}")
            return ""
    
    def _process_epub(self, file_path: str) -> str:
        """Process EPUB file"""
        return self.epub_processor.extract_text(file_path)
    
    def _process_fb2(self, file_path: str) -> str:
        """Process FB2 file"""
        return self.fb2_processor.extract_text(file_path)


if __name__ == "__main__":
    # Test the processors
    processor = ExtendedDocumentProcessor("qdrant", "test_collection")
    
    # Test EPUB processing
    test_epub = "test.epub"
    if Path(test_epub).exists():
        content = processor._process_epub(test_epub)
        print(f"EPUB content length: {len(content)}")
    
    # Test FB2 processing
    test_fb2 = "test.fb2"
    if Path(test_fb2).exists():
        content = processor._process_fb2(test_fb2)
        print(f"FB2 content length: {len(content)}")

