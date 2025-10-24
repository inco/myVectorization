#!/usr/bin/env python3
"""

Book Vectorization Utility with LangChain and PostgreSQL

This utility vectorizes documents using LangChain and stores embeddings in PostgreSQL
for semantic search and retrieval.

Libraries Used:
- langchain: Document processing and embeddings
- langchain-community: Community integrations
- langchain-core: Core LangChain functionality
- pypdf: PDF text extraction
- sentence-transformers: Embedding models
- psycopg2: PostgreSQL connection

Features:
- Document processing with LangChain
- Text chunking with overlap
- Embedding generation via SentenceTransformers or Ollama
- Metadata and vector storage in PostgreSQL
- Duplicate detection
- Recursive directory scanning
- Multiple file format support
"""
# python -u src\utility\myBookVectorizer.py --from-env --directory "O:\Finance" --workers 5

import argparse
import concurrent.futures
import hashlib
import logging
import os
import socket
import sys
import time
import uuid
from typing import List, Dict, Any
import threading
_wol_lock = threading.Lock()
# Host-level locks and recent-success cache to prevent duplicate wake attempts
_ollama_host_locks: Dict[str, threading.Lock] = {}
_ollama_last_ok: Dict[str, float] = {}
_ollama_ok_cooldown = 60  # seconds - treat server as OK for this many seconds after a successful check
# Third-party imports
import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv
# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
	PyPDFLoader,
	TextLoader
)
from langchain_core.documents import Document
from pathlib import Path
from psycopg2 import pool as pg_pool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('book_vectorizer')


def send_wake_on_lan(mac_address: str, broadcast_ips: List[str] | None = None) -> bool:
    """Send Wake-on-LAN packet to wake up a sleeping machine.

    Try a list of broadcast IPs (default includes 255.255.255.255). Return True on first
    successful send. This is more robust across different network setups.
    """
    # Ensure only one thread sends WOL packets at a time to avoid network/flooding issues
    try:
        with _wol_lock:
            # Remove any separators from MAC address
            mac = mac_address.replace(':', '').replace('-', '').replace('.', '')

            if len(mac) != 12:
                logger.error(f"Invalid MAC address format: {mac_address}")
                return False

            # Convert MAC to binary
            mac_bytes = bytes.fromhex(mac)

            # Create magic packet: 6 bytes of 0xFF followed by 16 repetitions of MAC
            magic_packet = b'\xFF' * 6 + mac_bytes * 16

            # Default broadcast targets
            targets = broadcast_ips[:] if broadcast_ips else ['255.255.255.255']

            # Ensure 255.255.255.255 is present as last resort
            if '255.255.255.255' not in targets:
                targets.append('255.255.255.255')

            last_exc = None
            for bcast in targets:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    # Allow reuse (helpful on some Windows setups)
                    try:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    except Exception:
                        pass

                    sock.sendto(magic_packet, (bcast, 9))
                    sock.close()
                    logger.info(f"Wake-on-LAN magic packet sent to {mac_address} via {bcast}")
                    return True
                except Exception as e:
                    last_exc = e
                    logger.debug(f"WOL send via {bcast} failed: {e}")

            logger.error(f"Failed to send Wake-on-LAN packet to {mac_address} on targets {targets}: {last_exc}")
            return False
    except Exception as e:
        logger.error(f"Failed to prepare/send Wake-on-LAN packet: {e}")
        return False


def ping_host(host: str, timeout: int = 3) -> bool:
    """Ping a host to check if it's reachable"""
    try:
        import subprocess
        import platform

        # Use appropriate ping command for Windows/Unix
        if platform.system().lower() == "windows":
            cmd = ["ping", "-n", "1", "-w", str(timeout * 1000), host]
        else:
            cmd = ["ping", "-c", "1", "-W", str(timeout), host]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 1)
        return result.returncode == 0

    except Exception as e:
        logger.error(f"Failed to ping {host}: {e}")
        return False


def ensure_ollama_server_available(host: str, port: int, mac_address: str = None, max_retries: int = 3,
                                    wol_wait: int = 30, service_wait: int = 15, max_total_wait: int = 120) -> bool:
    """Ensure Ollama server is available, wake it up if needed with retries.

    This function uses a total timeout (max_total_wait) to avoid long blocking when the
    remote machine is flaky. wol_wait and service_wait are configurable short waits used
    after sending a WOL packet and after the host becomes reachable respectively.

    This implementation serializes wake attempts per-host so that multiple threads do
    not simultaneously send Wake-on-LAN packets or repeatedly poll the service.
    """
    start_time = time.time()

    # Quick success-cache: if we recently verified the host, skip heavy checks
    last_ok = _ollama_last_ok.get(host)
    if last_ok and (time.time() - last_ok) < _ollama_ok_cooldown:
        logger.debug(f"Using recent success cache for {host} (age={int(time.time()-last_ok)}s)")
        return True

    # Obtain or create host lock
    host_lock = _ollama_host_locks.get(host)
    if host_lock is None:
        # Use a dict set/get under global lock to avoid races creating locks
        with _wol_lock:
            host_lock = _ollama_host_locks.get(host)
            if host_lock is None:
                host_lock = threading.Lock()
                _ollama_host_locks[host] = host_lock

    acquired = host_lock.acquire(blocking=False)
    # Try to acquire the host lock; wait up to the remaining allowed time so only one
    # thread performs the wake/retry work for this host at a time.
    we_acquired = False
    if not acquired:
        remaining = max(1, int(max_total_wait - (time.time() - start_time)))
        logger.info(f"Another thread may be handling Ollama for {host}; waiting up to {remaining}s to acquire host lock")
        try:
            got = host_lock.acquire(timeout=remaining)
            if not got:
                logger.warning(f"Timed out waiting to acquire host lock for {host}")
                return False
            else:
                we_acquired = True
        except Exception:
            logger.debug("Interrupted while waiting for host lock")
            return False
    else:
        we_acquired = True

    # At this point this thread holds the host_lock (we_acquired == True)
    try:
        # Helper: poll the Ollama HTTP /api/tags until it responds or timeout elapses
        def _poll_ollama_http(timeout_seconds: int, poll_interval: float = 1.0) -> bool:
            end_time = time.time() + timeout_seconds
            while time.time() < end_time:
                try:
                    response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
                    if response.status_code == 200:
                        return True
                except Exception:
                    # ignore and retry until timeout
                    pass
                time.sleep(poll_interval)
            return False

        for attempt in range(max_retries):
            elapsed = time.time() - start_time
            if elapsed > max_total_wait:
                logger.error(f"Timeout waiting for server {host}:{port} after {int(elapsed)}s")
                return False

            # First check if server responds to ping
            if ping_host(host):
                logger.info(f"Host {host} is reachable on attempt {attempt + 1}")

                # Poll the Ollama HTTP endpoint for a short time (bounded by remaining wait)
                remaining_for_poll = max(1, int(min(service_wait, max_total_wait - (time.time() - start_time))))
                logger.debug(f"Polling Ollama HTTP for up to {remaining_for_poll}s to allow services to come up")
                if _poll_ollama_http(remaining_for_poll, poll_interval=1.0):
                    logger.info(f"Ollama service is running on {host}:{port}")
                    _ollama_last_ok[host] = time.time()
                    return True

                logger.warning(f"Host {host} is reachable but Ollama service on port {port} is not responding (after polling)")
            else:
                logger.warning(f"Host {host} is not reachable on attempt {attempt + 1}")

            # If we're not on the last attempt, try to wake up the server
            if attempt < max_retries - 1:
                if mac_address:
                    logger.info(f"Attempting to wake up host {host} using MAC {mac_address}")
                    # Build candidate broadcast IPs: try a subnet broadcast near the host and global broadcast
                    candidate_bcasts = ['255.255.255.255']
                    try:
                        # If host is an IP address, create a simple /24 broadcast (best-effort)
                        host_ip = socket.gethostbyname(host)
                        if host_ip not in ('127.0.0.1', 'localhost'):
                            parts = host_ip.split('.')
                            if len(parts) == 4:
                                subnet_bcast = '.'.join(parts[:3] + ['255'])
                                if subnet_bcast not in candidate_bcasts:
                                    candidate_bcasts.insert(0, subnet_bcast)
                    except Exception:
                        pass

                    if send_wake_on_lan(mac_address, broadcast_ips=candidate_bcasts):
                        logger.info(f"WOL packet sent, waiting {wol_wait}s for system to wake up...")
                        time.sleep(wol_wait)

                        # Check host reachability and poll HTTP endpoint until service responds
                        if ping_host(host):
                            logger.info(f"Host {host} is now reachable after wake-up")

                            # Instead of one-off request, poll the HTTP endpoint for up to service_wait
                            logger.info(f"Polling Ollama HTTP for up to {service_wait}s for services to start...")
                            if _poll_ollama_http(service_wait, poll_interval=1.0):
                                logger.info(f"Ollama service is now running on {host}:{port}")
                                _ollama_last_ok[host] = time.time()
                                return True

                            logger.warning(f"Host {host} is awake but Ollama service is still not responding after polling")
                        else:
                            logger.warning(f"Host {host} did not wake up after WOL packet")
                    else:
                        logger.warning("Failed to send Wake-on-LAN packet")

                    # Exponential backoff before next retry, but bounded by remaining total wait
                    wait_time = min(5 * (attempt + 1), max(1, int(max_total_wait - (time.time() - start_time))))
                    if wait_time > 0:
                        logger.info(f"Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)
                else:
                    logger.warning("No MAC address provided for Wake-on-LAN")
                    wait_time = min(10, max(1, int(max_total_wait - (time.time() - start_time))))
                    if wait_time > 0:
                        logger.info(f"Waiting {wait_time} seconds before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait_time)

        # All retries exhausted
        err_msg = f"Failed to connect to Ollama server at {host}:{port} after {max_retries} attempts"
        logger.error(err_msg)
        try:
            print(err_msg)
        except Exception:
            pass
        return False
    finally:
        # Release host lock only if this thread acquired it earlier
        try:
            if we_acquired:
                host_lock.release()
        except RuntimeError:
            # Already released or not owned by us
            pass


class OpenAIEmbeddings:
    """Custom OpenAI embeddings implementation"""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API"""
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": self.model
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            raise


class OllamaEmbeddings:
    """Custom LangChain-compatible Ollama embeddings with retry and WOL support"""

    def __init__(self, model: str = "nomic-embed-text:latest", host: str = "localhost", port: int = 11434):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.max_retries = 3
        self.mac_address = os.getenv('OLLAMA_MAC_ADDRESS', 'a4:1f:72:fe:bf:fa')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama API with retry logic and WOL support"""
        # Try a direct request first; on connection issues, delegate to centralized ensure_ollama_server_available
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectTimeout) as e:
            logger.warning(f"Connection error when posting to Ollama: {e}")
            print(f"Ollama connection error: {e}")
            # Try to ensure the server is available (this will perform WOL and bounded retries)
            if ensure_ollama_server_available(self.host, self.port, self.mac_address, max_retries=self.max_retries,
                                              wol_wait=10, service_wait=5, max_total_wait=90):
                 # After ensure succeeded, attempt request again once
                 try:
                     response = requests.post(
                         f"{self.base_url}/api/embeddings",
                         json={"model": self.model, "prompt": text},
                         timeout=30
                     )
                     response.raise_for_status()
                     result = response.json()
                     return result["embedding"]
                 except Exception as e2:
                    logger.error(f"Failed to get embedding after waking Ollama: {e2}")
                    try:
                        print(f"Failed to get embedding after waking Ollama: {e2}")
                    except Exception:
                        pass
                    raise
            else:
                err = "Ollama server could not be brought online"
                logger.error(err)
                try:
                    print(err)
                except Exception:
                    pass
                raise
        except Exception as e:
            logger.error(f"Error getting Ollama embedding: {e}")
            raise


class PostgreSQLHandler:
    """Handle PostgreSQL operations for document metadata"""

    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.pool: pg_pool.ThreadedConnectionPool | None = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Initialize a threaded connection pool"""
        try:
            max_conn = max(8, (os.cpu_count() or 4) * 2)
            self.pool = pg_pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=max_conn,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            logger.info(f"PostgreSQL pool ready at {self.host}:{self.port} (maxconn={max_conn})")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise

    def _get_conn(self):
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized")
        return self.pool.getconn()

    def _put_conn(self, conn):
        if self.pool and conn:
            self.pool.putconn(conn)

    def _create_tables(self):
        """Create necessary tables"""
        # Install pgvector extension if not exists
        install_extension = """
        CREATE EXTENSION IF NOT EXISTS vector;
        """

        create_documents_table = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            document_id UUID UNIQUE NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL,
            file_stem VARCHAR(255) NOT NULL,
            file_size BIGINT,
            file_type VARCHAR(50),
            total_chunks INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            embedding_model VARCHAR(100),
            content_hash VARCHAR(64),
            metadata JSONB
        );
        
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            document_id UUID REFERENCES documents(document_id),
            chunk_index INTEGER NOT NULL,
            chunk_id UUID UNIQUE NOT NULL,
            content TEXT NOT NULL,
            content_preview TEXT,
            chunk_size INTEGER,
            embedding_vector vector(768) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_documents_file_stem ON documents(file_stem);
        CREATE UNIQUE INDEX IF NOT EXISTS uidx_documents_file_stem ON documents(file_stem);
        CREATE INDEX IF NOT EXISTS idx_documents_processed_at ON documents(processed_at);
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON document_chunks(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_vector ON document_chunks USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);
        """

        try:
            conn = self._get_conn()
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(install_extension)
                    cursor.execute(create_documents_table)
            logger.info("PostgreSQL tables created/verified with vector extension")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
        finally:
            self._put_conn(locals().get('conn'))

    def document_exists(self, file_stem: str) -> bool:
        """Check if document already exists by file stem"""
        try:
            conn = self._get_conn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM documents WHERE file_stem = %s LIMIT 1", (file_stem,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False
        finally:
            self._put_conn(locals().get('conn'))

    def insert_document(self, document_data: Dict[str, Any]) -> str:
        """Insert document metadata and return document_id"""
        try:
            conn = self._get_conn()
            document_id = str(uuid.uuid4())
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO documents 
                        (document_id, file_name, file_path, file_stem, file_size, file_type, 
                         total_chunks, embedding_model, content_hash, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        document_id,
                        document_data['file_name'],
                        document_data['file_path'],
                        document_data['file_stem'],
                        document_data['file_size'],
                        document_data['file_type'],
                        document_data['total_chunks'],
                        document_data['embedding_model'],
                        document_data['content_hash'],
                        psycopg2.extras.Json(document_data.get('metadata', {}))
                    ))
            logger.info(f"Inserted document: {document_data['file_name']}")
            return document_id
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            raise
        finally:
            self._put_conn(locals().get('conn'))

    def insert_chunk(self, chunk_data: Dict[str, Any]):
        """Insert document chunk"""
        try:
            conn = self._get_conn()
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO document_chunks 
                        (document_id, chunk_index, chunk_id, content, content_preview, 
                         chunk_size, embedding_vector)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        chunk_data['document_id'],
                        chunk_data['chunk_index'],
                        chunk_data['chunk_id'],
                        chunk_data['content'],
                        chunk_data['content_preview'],
                        chunk_data['chunk_size'],
                        chunk_data['embedding_vector']
                    ))
        except Exception as e:
            logger.error(f"Error inserting chunk: {e}")
            raise
        finally:
            self._put_conn(locals().get('conn'))

    def insert_document_with_chunks(self, document_data: Dict[str, Any], chunks_data: List[Dict[str, Any]]) -> str:
        """Insert document and all its chunks atomically using a transaction"""
        conn = None
        try:
            conn = self._get_conn()
            with conn:
                with conn.cursor() as cursor:
                    # Try to insert the document; skip if already exists (by file_stem)
                    cursor.execute("""
                        INSERT INTO documents 
                        (document_id, file_name, file_path, file_stem, file_size, file_type, 
                         total_chunks, embedding_model, content_hash, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (file_stem) DO NOTHING
                        RETURNING document_id
                    """, (
                        str(uuid.uuid4()),
                        document_data['file_name'],
                        document_data['file_path'],
                        document_data['file_stem'],
                        document_data['file_size'],
                        document_data['file_type'],
                        document_data['total_chunks'],
                        document_data['embedding_model'],
                        document_data['content_hash'],
                        psycopg2.extras.Json(document_data.get('metadata', {}))
                    ))
                    row = cursor.fetchone()
                    if row is None:
                        # Already exists, fetch its id and return without inserting chunks
                        cursor.execute("SELECT document_id FROM documents WHERE file_stem = %s", (document_data['file_stem'],))
                        existing_id = cursor.fetchone()[0]
                        logger.info(f"Document already exists, skipping chunks: {document_data['file_name']}")
                        return existing_id
                    document_id = row[0]

                    # Bulk insert all chunks in a single operation for better performance
                    if chunks_data:
                        chunk_values = [
                            (
                                document_id,
                                chunk_data['chunk_index'],
                                chunk_data['chunk_id'],
                                chunk_data['content'],
                                chunk_data['content_preview'],
                                chunk_data['chunk_size'],
                                chunk_data['embedding_vector']
                            )
                            for chunk_data in chunks_data
                        ]

                        # Use executemany for bulk insert - much faster than loop
                        psycopg2.extras.execute_batch(cursor, """
                            INSERT INTO document_chunks 
                            (document_id, chunk_index, chunk_id, content, content_preview, 
                             chunk_size, embedding_vector)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, chunk_values, page_size=100)

                logger.info(f"Successfully inserted document with {len(chunks_data)} chunks: {document_data['file_name']}")
                return document_id
        except Exception as e:
            logger.error(f"Error in atomic document insertion: {e}")
            raise
        finally:
            self._put_conn(conn)

    def cleanup_and_recreate_schema(self):
        """Drop all data and recreate the database schema"""
        conn = None
        try:
            conn = self._get_conn()
            with conn:
                with conn.cursor() as cursor:
                    logger.info("Dropping existing tables and data...")
                    cursor.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")
                    cursor.execute("DROP TABLE IF EXISTS documents CASCADE;")
            logger.info("Recreating schema...")
            self._create_tables()
            logger.info("Schema cleanup and recreation completed successfully")
        except Exception as e:
            logger.error(f"Failed to cleanup and recreate schema: {e}")
            raise
        finally:
            self._put_conn(conn)

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        conn = None
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        SUM(total_chunks) as total_chunks,
                        AVG(total_chunks) as avg_chunks_per_doc,
                        COUNT(DISTINCT embedding_model) as unique_models,
                        MAX(processed_at) as last_processed
                    FROM documents
                """)
                stats = cursor.fetchone()
                return dict(stats) if stats else {}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
        finally:
            self._put_conn(conn)

    def close(self):
        """Close database connection"""
        if self.pool:
            self.pool.closeall()



class BookVectorizer:
    """Main book vectorization class using LangChain and PostgreSQL"""

    def __init__(self,
                 pg_host: str,
                 pg_port: int,
                 pg_user: str,
                 pg_password: str,
                 pg_database: str,
                 embedding_mode: str = "ollama",
                 embedding_model: str = "nomic-embed-text:latest",
                 ollama_host: str = "localhost",
                 ollama_port: int = 11434,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):

        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.pg_database = pg_database
        self.embedding_mode = embedding_mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize PostgreSQL handler
        self.pg_handler = PostgreSQLHandler(pg_host, pg_port, pg_user, pg_password, pg_database)

        # Initialize embeddings
        if embedding_mode.lower() == "ollama":
            # Check if Ollama server is available, wake it up if needed
            mac_address = os.getenv('OLLAMA_MAC_ADDRESS', 'a4:1f:72:fe:bf:fa')
            if not ensure_ollama_server_available(ollama_host, ollama_port, mac_address):
                err_init = f"Cannot connect to Ollama server at {ollama_host}:{ollama_port} during initialization"
                logger.error(err_init)
                try:
                    print(err_init)
                except Exception:
                    pass
                raise ConnectionError(f"Cannot connect to Ollama server at {ollama_host}:{ollama_port}")

            self.embeddings = OllamaEmbeddings(embedding_model, ollama_host, ollama_port)
            self.embedding_model_name = embedding_model
        elif embedding_mode.lower() == "openai":
            self.embeddings = OpenAIEmbeddings(embedding_model)
            self.embedding_model_name = embedding_model
        else:
            logger.error(f"Unsupported embedding mode: {embedding_mode}")
            raise ValueError(f"Unsupported embedding mode: {embedding_mode}")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info(f"BookVectorizer initialized with {embedding_mode} {embedding_model}")

    def _load_document(self, file_path: str) -> List[Document]:
        """Load document using appropriate LangChain loader"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        suffixes = [s.lower() for s in file_path.suffixes]
        fb2_zip = len(suffixes) >= 2 and suffixes[-2] == '.fb2' and suffixes[-1] == '.zip'
        treat_as_fb2 = (extension == '.fb2') or fb2_zip

        try:
            if extension == '.pdf':
                # Enhanced PDF extraction with multiple fallback methods
                documents = None
                extraction_method = None
                pypdf_documents = None
                pypdf_total_len = 0

                # Method 1: PyPDFLoader (fast, no heavy deps)
                try:
                    logger.info(f"Attempting PyPDFLoader for {file_path.name}...")
                    loader = PyPDFLoader(str(file_path))
                    pypdf_documents = loader.load()
                    pypdf_total_len = sum(len((d.page_content or '')) for d in pypdf_documents)

                    if pypdf_total_len >= 100:  # Lowered from 500 to 100
                        extraction_method = "PyPDFLoader"
                        documents = pypdf_documents
                        logger.info(f"✓ PyPDFLoader: Loaded {len(documents)} pages, {pypdf_total_len} chars from {file_path.name}")
                    else:
                        logger.warning(f"PyPDFLoader extracted only {pypdf_total_len} chars, trying alternative methods")

                except Exception as pypdf_error:
                    error_type = type(pypdf_error).__name__
                    logger.warning(f"PyPDFLoader failed ({error_type}: {str(pypdf_error)[:100]})")

                # Method 2: PyMuPDF (fitz) - handles more PDF types including encrypted/image PDFs
                if documents is None:
                    try:
                        logger.info(f"Attempting PyMuPDF (fitz) extraction for {file_path.name}...")
                        import fitz  # PyMuPDF

                        pdf_doc = fitz.open(str(file_path))
                        fitz_content = []
                        total_fitz_len = 0

                        for page_num in range(pdf_doc.page_count):
                            page = pdf_doc[page_num]
                            text = page.get_text()
                            if text.strip():  # Only add non-empty pages
                                fitz_content.append(Document(
                                    page_content=text,
                                    metadata={
                                        "source": str(file_path),
                                        "page": page_num + 1,
                                        "total_pages": pdf_doc.page_count
                                    }
                                ))
                                total_fitz_len += len(text)

                        pdf_doc.close()

                        if total_fitz_len >= 50:
                            documents = fitz_content
                            extraction_method = "PyMuPDF (fitz)"
                            logger.info(f"✓ PyMuPDF: Extracted {total_fitz_len} chars from {len(fitz_content)} pages in {file_path.name}")
                        else:
                            logger.warning(f"PyMuPDF extracted only {total_fitz_len} chars, trying next method")

                    except ImportError:
                        logger.warning("PyMuPDF (fitz) not available. Install with: pip install pymupdf")
                    except Exception as fitz_error:
                        error_type = type(fitz_error).__name__
                        logger.warning(f"PyMuPDF failed ({error_type}: {str(fitz_error)[:100]})")

                # Method 3: pdfplumber - excellent for tables and complex layouts
                if documents is None:
                    try:
                        logger.info(f"Attempting pdfplumber extraction for {file_path.name}...")
                        import pdfplumber

                        plumber_content = []
                        total_plumber_len = 0

                        with pdfplumber.open(str(file_path)) as pdf:
                            for page_num, page in enumerate(pdf.pages):
                                text = page.extract_text()
                                if text and text.strip():
                                    plumber_content.append(Document(
                                        page_content=text,
                                        metadata={
                                            "source": str(file_path),
                                            "page": page_num + 1,
                                            "total_pages": len(pdf.pages)
                                        }
                                    ))
                                    total_plumber_len += len(text)

                        if total_plumber_len >= 50:
                            documents = plumber_content
                            extraction_method = "pdfplumber"
                            logger.info(f"✓ pdfplumber: Extracted {total_plumber_len} chars from {len(plumber_content)} pages in {file_path.name}")
                        else:
                            logger.warning(f"pdfplumber extracted only {total_plumber_len} chars, trying next method")

                    except ImportError:
                        logger.warning("pdfplumber not available. Install with: pip install pdfplumber")
                    except Exception as plumber_error:
                        error_type = type(plumber_error).__name__
                        logger.warning(f"pdfplumber failed ({error_type}: {str(plumber_error)[:100]})")

                # Method 4: UnstructuredPDFLoader if previous methods failed
                if documents is None:
                    try:
                        logger.info(f"Attempting UnstructuredPDFLoader (hi_res) for {file_path.name}...")
                        from langchain_community.document_loaders import UnstructuredPDFLoader
                        uloader = UnstructuredPDFLoader(str(file_path), mode='single', strategy='hi_res')
                        documents = uloader.load()
                        utotal_len = sum(len((d.page_content or '')) for d in documents)

                        if utotal_len > 50:
                            extraction_method = "UnstructuredPDFLoader (hi_res)"
                            logger.info(f"✓ UnstructuredPDFLoader: Extracted {utotal_len} chars from {file_path.name}")
                        else:
                            logger.warning(f"UnstructuredPDFLoader extracted only {utotal_len} chars, trying OCR...")
                            documents = None

                    except ModuleNotFoundError as module_error:
                        logger.warning(f"UnstructuredPDFLoader requires additional dependencies: {module_error}")
                        logger.debug("To enable advanced PDF processing, install: pip install unstructured[pdf]")
                        documents = None
                    except Exception as unstructured_error:
                        error_type = type(unstructured_error).__name__
                        logger.warning(f"UnstructuredPDFLoader (hi_res) failed ({error_type}: {str(unstructured_error)[:100]})")
                        documents = None

                # Method 5: OCR as last resort for image-based PDFs
                if documents is None:
                    try:
                        logger.info(f"Attempting OCR extraction for {file_path.name}...")

                        # Try PyMuPDF OCR first (if available)
                        try:
                            import fitz
                            import io
                            from PIL import Image

                            pdf_doc = fitz.open(str(file_path))
                            ocr_content = []
                            total_ocr_len = 0

                            for page_num in range(pdf_doc.page_count):
                                page = pdf_doc[page_num]

                                # First try text extraction
                                text = page.get_text()
                                if len(text.strip()) < 10:  # If minimal text, try OCR
                                    try:
                                        # Convert page to image
                                        pix = page.get_pixmap()
                                        img_data = pix.tobytes("ppm")
                                        image = Image.open(io.BytesIO(img_data))

                                        # Use Tesseract OCR if available
                                        try:
                                            import pytesseract
                                            ocr_text = pytesseract.image_to_string(image)
                                            if ocr_text.strip():
                                                text = ocr_text
                                        except ImportError:
                                            logger.debug("pytesseract not available for OCR")

                                    except Exception as page_ocr_error:
                                        logger.debug(f"OCR failed for page {page_num + 1}: {page_ocr_error}")

                                if text.strip():
                                    ocr_content.append(Document(
                                        page_content=text,
                                        metadata={
                                            "source": str(file_path),
                                            "page": page_num + 1,
                                            "total_pages": pdf_doc.page_count,
                                            "extraction_method": "OCR" if len(text.strip()) > 10 else "text"
                                        }
                                    ))
                                    total_ocr_len += len(text)

                            pdf_doc.close()

                            if total_ocr_len > 50:
                                documents = ocr_content
                                extraction_method = "PyMuPDF + OCR"
                                logger.info(f"✓ OCR: Extracted {total_ocr_len} chars from {len(ocr_content)} pages in {file_path.name}")
                            else:
                                logger.warning(f"OCR extracted only {total_ocr_len} chars")

                        except ImportError:
                            logger.debug("PyMuPDF not available for OCR fallback")

                        # Fallback to UnstructuredPDFLoader OCR
                        if documents is None:
                            try:
                                from langchain_community.document_loaders import UnstructuredPDFLoader
                                uloader_ocr = UnstructuredPDFLoader(str(file_path), mode='single', strategy='ocr_only')
                                documents = uloader_ocr.load()
                                utotal_ocr = sum(len((d.page_content or '')) for d in documents)

                                if utotal_ocr > 50:
                                    extraction_method = "UnstructuredPDFLoader (OCR)"
                                    logger.info(f"✓ OCR: Extracted {utotal_ocr} chars from {file_path.name}")
                                else:
                                    logger.warning(f"OCR extracted only {utotal_ocr} chars - insufficient text")
                                    documents = None

                            except Exception as ocr_error:
                                error_type = type(ocr_error).__name__
                                logger.warning(f"OCR extraction failed ({error_type}: {str(ocr_error)[:100]})")
                                documents = None

                    except Exception as general_ocr_error:
                        logger.warning(f"All OCR methods failed: {general_ocr_error}")

                # Enhanced fallback with better metadata
                if documents is None and pypdf_documents is not None:
                    if pypdf_total_len > 0:
                        logger.warning(f"⚠ All advanced methods failed. Using PyPDFLoader results with minimal text ({pypdf_total_len} chars)")
                        documents = pypdf_documents
                        extraction_method = "PyPDFLoader (minimal text fallback)"
                    else:
                        # Create enhanced placeholder with file analysis
                        logger.warning(f"⚠ All extraction methods failed for {file_path.name}")
                        logger.warning(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")

                        # Try to determine PDF properties
                        pdf_info = self._analyze_pdf_file(str(file_path))

                        placeholder_content = f"[PDF Extraction Failed - {file_path.name}]\n\n"
                        placeholder_content += f"File Analysis:\n"
                        placeholder_content += f"- Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB\n"
                        placeholder_content += f"- {pdf_info}\n\n"
                        placeholder_content += "This PDF may require:\n"
                        placeholder_content += "- OCR tools (install: pip install pytesseract)\n"
                        placeholder_content += "- Poppler utilities for image conversion\n"
                        placeholder_content += "- Password if encrypted\n"
                        placeholder_content += "- Manual text extraction if scanned images only"

                        documents = [Document(
                            page_content=placeholder_content,
                            metadata={
                                "source": str(file_path),
                                "extraction_failed": True,
                                "file_name": file_path.name,
                                "file_size_mb": round(file_path.stat().st_size / 1024 / 1024, 2),
                                "analysis": pdf_info,
                                "note": "All extraction methods failed - may need OCR/Poppler or be encrypted"
                            }
                        )]
                        extraction_method = "Enhanced Placeholder (extraction failed)"

                if documents and extraction_method:
                    logger.info(f"✓ Successfully extracted PDF using {extraction_method}")
                    return documents
                else:
                    logger.error(f"❌ All extraction methods failed for {file_path.name}")
                    logger.error("This PDF may be: encrypted, image-only without OCR capabilities, or severely corrupted")
                    raise ValueError(f"Cannot extract text from PDF - all methods failed")

            elif extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif extension == '.epub':
                # Use lightweight ebooklib instead of pandoc-dependent UnstructuredEPubLoader
                try:
                    content = self._extract_epub_content(str(file_path))
                    # Create a Document object directly
                    doc = Document(page_content=content, metadata={"source": str(file_path)})
                    return [doc]
                except Exception as epub_error:
                    logger.warning(f"Failed to load EPUB with ebooklib: {epub_error}")
                    try:
                        # Fallback to generic unstructured loader
                        from langchain_community.document_loaders import UnstructuredFileLoader
                        loader = UnstructuredFileLoader(str(file_path))
                        logger.info(f"Using UnstructuredFileLoader as fallback for EPUB: {file_path.name}")
                    except Exception as fallback_error:
                        # Final fallback - try to read as text (won't work well but better than failing)
                        logger.warning(f"All EPUB loaders failed, trying text loader: {fallback_error}")
                        loader = TextLoader(str(file_path), encoding='utf-8', autodetect_encoding=True)
            elif treat_as_fb2:
                # FB2 files are XML-based format - need proper XML parsing
                logger.info(f"Attempting to load FB2 file: {file_path.name}")
                try:
                    content = self._extract_fb2_content(str(file_path))
                    doc = Document(page_content=content, metadata={"source": str(file_path)})
                    return [doc]
                except Exception as fb2_error:
                    logger.error(f"Failed to load FB2 file: {fb2_error}")
                    raise
            elif extension == '.mobi':
                # MOBI files are proprietary Amazon format - try multiple approaches
                logger.info(f"Attempting to load MOBI file: {file_path.name}")
                
                # Try 1: Use mobi library if available
                try:
                    content = self._extract_mobi_content(str(file_path))
                    doc = Document(page_content=content, metadata={"source": str(file_path)})
                    return [doc]
                except ImportError:
                    logger.warning("mobi library not available, trying alternative methods...")
                except Exception as mobi_error:
                    logger.warning(f"Failed to load MOBI with mobi library: {mobi_error}")
                
                # Try 2: Use UnstructuredFileLoader with mode='elements'
                try:
                    from langchain_community.document_loaders import UnstructuredFileLoader
                    loader = UnstructuredFileLoader(str(file_path), mode='single')
                    documents = loader.load()
                    logger.info(f"Successfully loaded MOBI using UnstructuredFileLoader")
                    return documents
                except Exception as unstructured_error:
                    logger.warning(f"UnstructuredFileLoader failed for MOBI: {unstructured_error}")
                
                # Try 3: Read as binary and try to extract text (very basic)
                try:
                    logger.warning(f"Attempting basic text extraction from MOBI file: {file_path.name}")
                    # MOBI files contain text that can be partially extracted
                    with open(file_path, 'rb') as f:
                        content = f.read()

                        # MOBI files have a specific structure - try to find the text content
                        # Look for text between record boundaries
                        import re

                        # Try multiple encodings
                        text = None
                        for encoding in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                decoded = content.decode(encoding, errors='ignore')
                                # MOBI text often starts after certain markers
                                # Remove binary headers and metadata
                                if 'BOOKMOBI' in decoded or 'MOBI' in decoded:
                                    # Find actual text content (usually after metadata)
                                    # Look for consecutive readable text
                                    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', decoded)
                                    # Keep only printable ASCII and common Unicode
                                    cleaned = re.sub(r'[^\x20-\x7E\u0080-\uFFFF\n\r\t]+', ' ', cleaned)

                                    # Extract sentences (text with spaces and punctuation)
                                    sentences = re.findall(r'[A-Z][^.!?]*[.!?]', cleaned)
                                    if sentences:
                                        text = ' '.join(sentences)
                                        break

                                    # Fallback: just clean up and use what we have
                                    if len(cleaned.strip()) > 1000:
                                        text = cleaned
                                        break
                            except Exception:
                                continue

                        if not text:
                            # Final attempt: just decode and clean
                            text = content.decode('utf-8', errors='ignore')
                            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
                            text = re.sub(r'[^\x20-\x7E\n\r\t]+', ' ', text)

                        # Clean up extra whitespace
                        text = re.sub(r'\s+', ' ', text).strip()
                        # Remove very short fragments and binary junk
                        text = re.sub(r'\b\w{1,2}\b', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()

                        if len(text) > 500:  # Need at least 500 chars of meaningful text
                            logger.info(f"Extracted {len(text)} characters from MOBI file using binary extraction")
                            doc = Document(page_content=text, metadata={"source": str(file_path), "extraction_method": "binary"})
                            return [doc]
                        else:
                            logger.error(f"Could not extract sufficient text from MOBI file: {file_path.name} (only {len(text)} chars)")
                            raise ValueError(f"Failed to extract meaningful text from MOBI file: {file_path.name}")

                except Exception as binary_error:
                    logger.error(f"All MOBI extraction methods failed: {binary_error}")
                    raise ValueError(f"Unsupported MOBI file - please convert to EPUB or PDF: {file_path.name}")
                    
            elif extension in ['.doc', '.docx']:
                try:
                    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                    loader = UnstructuredWordDocumentLoader(str(file_path))
                except ImportError:
                    # Fallback to unstructured file loader
                    try:
                        from langchain_community.document_loaders import UnstructuredFileLoader
                        loader = UnstructuredFileLoader(str(file_path))
                    except ImportError:
                        raise ValueError(f"No suitable loader available for {extension} files")
            else:
                raise ValueError(f"Unsupported file type: {extension}")

            # For files using standard loaders
            if 'loader' in locals():
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} document(s) from {file_path.name}")
                return documents

        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise

    def _extract_epub_content(self, file_path: str) -> str:
        """Extract text content from EPUB file using ebooklib"""
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        try:
            book = epub.read_epub(file_path)
            content_parts = []

            # Extract text from all HTML documents in the EPUB
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Parse HTML content and extract text
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    if text:
                        content_parts.append(text)

            full_content = '\n\n'.join(content_parts)
            logger.info(f"Extracted {len(full_content)} characters from EPUB")
            return full_content

        except Exception as e:
            logger.error(f"Error extracting EPUB content: {e}")
            raise

    def _extract_mobi_content(self, file_path: str) -> str:
        """Extract text content from MOBI file using mobi library"""
        try:
            import mobi

            # Use mobi library to extract text
            content = mobi.extract(file_path)

            if content:
                logger.info(f"Successfully extracted {len(content)} characters from MOBI file")
                return content
            else:
                raise ValueError("No content extracted from MOBI file")

        except Exception as e:
            logger.error(f"Error extracting MOBI content: {e}")
            raise

    def _extract_fb2_content(self, file_path: str) -> str:
        """Extract text content from FB2 file (XML-based or zipped),
        extracting to a temp folder when zipped and deleting it on success,
        but keeping it for debugging if an error occurs.
        """
        import zipfile
        import tempfile
        import os
        import shutil

        temp_dir = None
        raw_content: bytes

        # Check if file is a ZIP archive
        with open(file_path, 'rb') as f:
            header = f.read(4)
        try:
            if header == b'PK\x03\x04':
                logger.info(f"FB2 file is a ZIP-wrapped archive, extracting...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    fb2_files = [f for f in file_list if f.lower().endswith('.fb2') or f.lower().endswith('.xml')]
                    target_member = None
                    if fb2_files:
                        target_member = fb2_files[0]
                        logger.info(f"Extracting {target_member} from ZIP...")
                    else:
                        # Try any file that looks like XML
                        for filename in file_list:
                            if not filename.endswith('/'):
                                candidate = zip_ref.read(filename)
                                if b'<?xml' in candidate[:500] or b'<FictionBook' in candidate[:1000] or b'<fictionbook' in candidate[:1000].lower():
                                    logger.info(f"Found XML content in {filename}")
                                    target_member = filename
                                    break
                        if not target_member:
                            raise ValueError("No XML/FB2 content found in ZIP archive")

                    # Extract to a persistent temp directory so we can keep it on failure
                    temp_dir = tempfile.mkdtemp(prefix="fb2_extract_")
                    zip_ref.extract(member=target_member, path=temp_dir)
                    extracted_path = os.path.join(temp_dir, target_member)
                    # If the member path included directories, ensure we point to the actual file
                    if os.path.isdir(extracted_path):
                        # Find first fb2/xml file inside
                        for root, _dirs, files in os.walk(extracted_path):
                            for f_name in files:
                                if f_name.lower().endswith(('.fb2', '.xml')):
                                    extracted_path = os.path.join(root, f_name)
                                    break
                    with open(extracted_path, 'rb') as ef:
                        raw_content = ef.read()
            else:
                with open(file_path, 'rb') as f:
                    raw_content = f.read()

            # Parse and return content
            content = BookVectorizer._parse_fb2_raw(raw_content)

            # Success: cleanup any temp dir
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

            return content
        except Exception as e:
            # On failure: keep temp_dir for debugging
            if temp_dir and os.path.isdir(temp_dir):
                logger.warning(f"Keeping extracted temp folder for debugging: {temp_dir}")
            raise

    @staticmethod
    def _parse_fb2_raw(raw_content: bytes) -> str:
        """Parse raw FB2/XML bytes into text. Handles BOM removal, multiple encodings,
        XML parsing, and BeautifulSoup fallbacks. Returns text or raises on failure.
        """
        import xml.etree.ElementTree as ET
        from bs4 import BeautifulSoup
        import codecs

        # Remove BOM if present
        if raw_content.startswith(codecs.BOM_UTF8):
            raw_content = raw_content[len(codecs.BOM_UTF8):]
        elif raw_content.startswith(codecs.BOM_UTF16_LE):
            raw_content = raw_content[len(codecs.BOM_UTF16_LE):]
        elif raw_content.startswith(codecs.BOM_UTF16_BE):
            raw_content = raw_content[len(codecs.BOM_UTF16_BE):]
        elif raw_content.startswith(codecs.BOM_UTF32_LE):
            raw_content = raw_content[len(codecs.BOM_UTF32_LE):]
        elif raw_content.startswith(codecs.BOM_UTF32_BE):
            raw_content = raw_content[len(codecs.BOM_UTF32_BE):]

        # Try multiple encodings
        decoded_content = None
        encoding_used = None
        for encoding in ['utf-8', 'windows-1251', 'cp1251', 'koi8-r', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'iso-8859-1']:
            try:
                decoded_content = raw_content.decode(encoding)
                encoding_used = encoding
                logger.info(f"Successfully decoded FB2 with {encoding} encoding")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if decoded_content is None:
            decoded_content = raw_content.decode('utf-8', errors='replace')
            encoding_used = 'utf-8 (with errors replaced)'
            logger.warning(f"Decoded FB2 with utf-8 and replaced errors")

        # Try to parse as XML
        try:
            root = ET.fromstring(decoded_content.encode('utf-8'))
            namespaces = {
                'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0',
                'xlink': 'http://www.w3.org/1999/xlink'
            }
            paragraphs = root.findall('.//fb:p', namespaces)
            if not paragraphs:
                paragraphs = root.findall('.//p')
            content_parts = []
            for p in paragraphs:
                text = ''.join(p.itertext())
                if text and text.strip():
                    content_parts.append(text.strip())
            if content_parts:
                full_content = '\n\n'.join(content_parts)
                logger.info(f"Extracted {len(full_content)} characters from {len(content_parts)} paragraphs using {encoding_used}")
                return full_content
            else:
                all_text = ''.join(root.itertext())
                cleaned_text = ' '.join(all_text.split())
                if len(cleaned_text.strip()) > 50:
                    logger.info(f"Extracted {len(cleaned_text)} characters from FB2 (fallback method)")
                    return cleaned_text.strip()
                else:
                    raise ValueError("Insufficient content from XML parser")
        except Exception as e:
            logger.warning(f"XML parsing failed: {e}, trying BeautifulSoup")
            # Try BeautifulSoup for malformed XML
            for parser in ['lxml-xml', 'lxml', 'html.parser']:
                try:
                    soup = BeautifulSoup(decoded_content, parser)
                    paragraphs = soup.find_all('p')
                    if paragraphs:
                        content_parts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
                        if content_parts:
                            full_content = '\n\n'.join(content_parts)
                            logger.info(f"Extracted {len(full_content)} characters from {len(content_parts)} paragraphs using BeautifulSoup")
                            return full_content
                    for tag_name in ['body', 'FictionBook', 'fictionbook', 'description']:
                        body = soup.find(tag_name)
                        if body:
                            all_text = body.get_text(separator=' ', strip=True)
                            cleaned_text = ' '.join(all_text.split())
                            if len(cleaned_text) > 50:
                                logger.info(f"Extracted {len(cleaned_text)} characters from <{tag_name}> tag using BeautifulSoup")
                                return cleaned_text
                    all_text = soup.get_text(separator=' ', strip=True)
                    cleaned_text = ' '.join(all_text.split())
                    if len(cleaned_text) > 50:
                        logger.info(f"Extracted {len(cleaned_text)} characters from FB2 using BeautifulSoup (final fallback)")
                        return cleaned_text
                except Exception as parser_error:
                    logger.debug(f"Parser {parser} failed: {parser_error}")
                    continue
            raise ValueError(f"Could not extract meaningful text from FB2 file")

    def _analyze_pdf_file(self, file_path: str) -> str:
        """Analyze PDF file to determine its properties and potential issues"""
        try:
            # Try to get basic PDF info using PyMuPDF if available
            try:
                import fitz
                pdf_doc = fitz.open(file_path)
                page_count = pdf_doc.page_count

                # Check if PDF has text layers
                has_text = False
                has_images = False
                total_text_len = 0

                # Sample first few pages to determine content type
                sample_pages = min(3, page_count)
                for i in range(sample_pages):
                    page = pdf_doc[i]
                    text = page.get_text()
                    total_text_len += len(text.strip())
                    if len(text.strip()) > 10:
                        has_text = True

                    # Check for images
                    image_list = page.get_images()
                    if image_list:
                        has_images = True

                pdf_doc.close()

                # Determine PDF type
                if total_text_len > 100:
                    pdf_type = "Text-based PDF"
                elif has_images and total_text_len < 50:
                    pdf_type = "Image-only PDF (may need OCR)"
                elif has_images and total_text_len > 0:
                    pdf_type = "Mixed content PDF (text + images)"
                else:
                    pdf_type = "Unknown content type"

                analysis = f"Pages: {page_count}, Type: {pdf_type}, Text sample length: {total_text_len}"

                # Check if encrypted
                try:
                    # Re-open to check encryption
                    test_doc = fitz.open(file_path)
                    if test_doc.needs_pass:
                        analysis += ", Status: Password protected"
                    else:
                        analysis += ", Status: Not encrypted"
                    test_doc.close()
                except:
                    analysis += ", Status: Cannot determine encryption"

                return analysis

            except ImportError:
                # Fallback without PyMuPDF
                import os
                file_size = os.path.getsize(file_path)
                return f"Size: {file_size / 1024 / 1024:.2f} MB, Analysis: Limited (PyMuPDF not available)"

        except Exception as e:
            return f"Analysis failed: {str(e)[:100]}"

    def _clean_text(self, text: str) -> str:
        """Clean text by removing problematic characters for PostgreSQL"""
        if not text:
            return text

        # Remove NUL characters (0x00) that PostgreSQL doesn't allow
        text = text.replace('\x00', '')

        # Remove other control characters that can cause issues
        import re
        # Remove control characters except newlines, tabs, and carriage returns
        text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content"""
        return hashlib.sha256(content.encode()).hexdigest()

    def process_file(self, file_path: str) -> bool:
        """Process a single file"""
        file_path = Path(file_path).resolve()

        if not file_path.exists() or not file_path.is_file():
            logger.error(f"File not found: {file_path}")
            return False

        # Normalize stem for .fb2.zip to avoid duplicate entries vs .fb2
        suffixes = [s.lower() for s in file_path.suffixes]
        if len(suffixes) >= 2 and suffixes[-2] == '.fb2' and suffixes[-1] == '.zip':
            base_path = file_path.with_suffix('').with_suffix('')
            file_stem = base_path.stem
            normalized_file_type = '.fb2'
        else:
            file_stem = file_path.stem
            normalized_file_type = file_path.suffix

        # Check if already processed
        if self.pg_handler.document_exists(file_stem):
            logger.info(f"📋 Document already processed: {file_stem}")
            return True

        try:
            logger.info(f"Processing file: {file_path.name}")

            # Load document
            documents = self._load_document(str(file_path))

            # Combine all document content and clean it
            full_content = "\n".join([doc.page_content for doc in documents])
            full_content = self._clean_text(full_content)

            if len(full_content.strip()) < 10:
                logger.warning(f"Insufficient content in {file_path}")
                return False

            # Split into chunks
            chunks = self.text_splitter.split_text(full_content)
            logger.info(f"Created {len(chunks)} chunks")

            # Clean each chunk before generating embeddings
            cleaned_chunks = [self._clean_text(chunk) for chunk in chunks]

            # Generate embeddings
            embeddings = self.embeddings.embed_documents(cleaned_chunks)

            # Prepare document metadata
            document_data = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_stem': file_stem,
                'file_size': file_path.stat().st_size,
                'file_type': normalized_file_type,
                'total_chunks': len(chunks),
                'embedding_model': self.embedding_model_name,
                'content_hash': self._generate_content_hash(full_content),
                'metadata': {
                    'original_documents': len(documents),
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                }
            }

            # Prepare all chunks data
            chunks_data = []
            for i, (chunk, embedding) in enumerate(zip(cleaned_chunks, embeddings)):
                chunk_id = str(uuid.uuid4())
                # Clean text for database storage
                clean_chunk = self._clean_text(chunk)
                clean_preview = self._clean_text(chunk[:500])

                chunk_data = {
                    'chunk_index': i,
                    'chunk_id': chunk_id,
                    'content': clean_chunk,
                    'content_preview': clean_preview,
                    'chunk_size': len(clean_chunk),
                    'embedding_vector': embedding
                }
                chunks_data.append(chunk_data)

            # Insert document and all chunks atomically (conflict-safe)
            document_id = self.pg_handler.insert_document_with_chunks(document_data, chunks_data)

            logger.info(f"Successfully processed: {file_path.name} ({len(chunks)} chunks)")
            return True

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False

    def _find_best_document_format(self, directory_path: Path) -> Dict[str, Path]:
        """Find the best format for each document in a directory based on priority order"""
        # Priority order: epub, fb2, mobi, doc, docx, pdf, txt only
        format_priority = ['.epub', '.fb2', '.mobi', '.doc', '.docx', '.pdf', '.txt']

        # Group files by stem name
        documents_by_stem = {}

        # Supported extensions derived from format_priority (no duplication)
        supported_extensions = set(format_priority)

        # Add debug logging
        logger.debug(f"Scanning directory: {directory_path}")
        logger.debug(f"Supported extensions: {supported_extensions}")

        # Collect all supported files
        file_count = 0
        for file_path in directory_path.iterdir():
            file_count += 1
            logger.debug(f"Found item: {file_path.name}, is_file: {file_path.is_file()}, suffix: '{file_path.suffix.lower()}'")

            if not file_path.is_file():
                continue

            # Recognize .fb2.zip as FB2 for discovery
            suffixes = [s.lower() for s in file_path.suffixes]
            is_fb2_zip = len(suffixes) >= 2 and suffixes[-2] == '.fb2' and suffixes[-1] == '.zip'
            is_supported = (file_path.suffix.lower() in supported_extensions) or is_fb2_zip

            if is_supported:
                # Normalize stem grouping: treat *.fb2.zip as the same stem as *.fb2
                if is_fb2_zip:
                    # Remove two suffixes: .zip then .fb2
                    base_path = file_path.with_suffix('')  # remove .zip
                    base_path = base_path.with_suffix('')  # remove .fb2
                    stem = base_path.stem
                else:
                    stem = file_path.stem
                if stem not in documents_by_stem:
                    documents_by_stem[stem] = []
                documents_by_stem[stem].append(file_path)
                logger.debug(f"Added supported file: {file_path.name} with normalized stem: {stem}")

        logger.debug(f"Total items in directory: {file_count}")
        logger.debug(f"Documents found by stem: {list(documents_by_stem.keys())}")

        # Select best format for each document
        selected_documents = {}

        for stem, file_list in documents_by_stem.items():
            if len(file_list) == 1:
                # Only one format available
                selected_documents[stem] = file_list[0]
                logger.info(f"📄 Single format found: {file_list[0].name}")
            else:
                # Multiple formats - choose by priority
                best_file = None
                best_priority = len(format_priority)

                for file_path in file_list:
                    suffixes = [s.lower() for s in file_path.suffixes]
                    ext = file_path.suffix.lower()
                    # Map *.fb2.zip to .fb2 for priority comparison
                    if len(suffixes) >= 2 and suffixes[-2] == '.fb2' and suffixes[-1] == '.zip':
                        ext = '.fb2'
                    try:
                        priority = format_priority.index(ext)
                        if priority < best_priority:
                            best_priority = priority
                            best_file = file_path
                    except ValueError:
                        # Extension not in priority list, use as fallback
                        if best_file is None:
                            best_file = file_path

                if best_file:
                    selected_documents[stem] = best_file
                    other_formats = [f.name for f in file_list if f != best_file]
                    logger.info(f"📚 Multiple formats found for '{stem}': Selected {best_file.name}, skipped {other_formats}")

        return selected_documents

    def process_directory(self, directory_path: str, recursive: bool = True, parallel: bool = False, max_workers: int = 3) -> Dict[str, int]:
        """
        Process all supported files in a directory with smart format selection

        Args:
            directory_path: Path to directory to process
            recursive: Whether to process subdirectories
            parallel: Whether to use parallel processing with multiple threads
            max_workers: Number of parallel worker threads (default: 3)

        Returns:
            Dictionary with success, failed, and skipped counts
        """
        directory_path = Path(directory_path).resolve()

        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return {"success": 0, "failed": 0, "skipped": 0}

        results = {"success": 0, "failed": 0, "skipped": 0}

        # Helper to consume completed futures and update results
        def _drain_completed(futures_set: set):
            completed = []
            for future in list(futures_set):
                if future.done():
                    completed.append(future)
            for future in completed:
                try:
                    success = future.result()
                    if success:
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                except Exception as e:
                    logger.error(f"Error in parallel task: {e}")
                    results["failed"] += 1
                finally:
                    futures_set.discard(future)

        if recursive:
            # Lazy traversal: walk the tree and process per-directory
            if parallel:
                logger.info(f"🚀 Processing recursively with up to {max_workers} parallel threads")
                pending_futures: set = set()
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for root, dirs, files in os.walk(directory_path):
                        current_dir = Path(root)
                        logger.info(f"📁 Processing directory: {current_dir}")

                        selected_documents = self._find_best_document_format(current_dir)
                        if not selected_documents:
                            continue

                        for file_path in selected_documents.values():
                            # Submit and bound the number of outstanding tasks
                            pending_futures.add(executor.submit(self.process_file, str(file_path)))
                            if len(pending_futures) >= max_workers * 8:
                                # Drain a few completed futures to keep memory bounded
                                concurrent.futures.wait(pending_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                                _drain_completed(pending_futures)

                    # Final drain
                    concurrent.futures.wait(pending_futures, return_when=concurrent.futures.ALL_COMPLETED)
                    _drain_completed(pending_futures)
            else:
                # Sequential, process as we go
                for root, dirs, files in os.walk(directory_path):
                    current_dir = Path(root)
                    logger.info(f"📁 Processing directory: {current_dir}")

                    selected_documents = self._find_best_document_format(current_dir)
                    if not selected_documents:
                        continue

                    for file_path in selected_documents.values():
                        try:
                            success = self.process_file(str(file_path))
                            if success:
                                results["success"] += 1
                            else:
                                results["failed"] += 1
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                            results["failed"] += 1
        else:
            # Non-recursive: process only the specified directory
            logger.info(f"📁 Processing directory: {directory_path}")
            selected_documents = self._find_best_document_format(directory_path)

            if not selected_documents:
                logger.warning(f"No supported files found in {directory_path}")
                return results

            if parallel and len(selected_documents) > 1:
                logger.info(f"🚀 Processing with {max_workers} parallel threads")
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(self.process_file, str(fp)) for fp in selected_documents.values()]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            success = future.result()
                            if success:
                                results["success"] += 1
                            else:
                                results["failed"] += 1
                        except Exception as e:
                            logger.error(f"Error in parallel task: {e}")
                            results["failed"] += 1
            else:
                for file_path in selected_documents.values():
                    try:
                        success = self.process_file(str(file_path))
                        if success:
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        results["failed"] += 1

        return results

    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search in PostgreSQL using proper vector casting via pooled connection
            conn = self.pg_handler._get_conn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            d.file_name, 
                            d.file_stem, 
                            d.document_id, 
                            dc.chunk_index, 
                            dc.content, 
                            dc.embedding_vector <=> %s::vector AS distance
                        FROM documents d
                        JOIN document_chunks dc ON d.document_id = dc.document_id
                        WHERE dc.embedding_vector IS NOT NULL
                        ORDER BY distance
                        LIMIT %s
                    """, (query_embedding, limit))
                    results = cursor.fetchall()
            finally:
                self.pg_handler._put_conn(conn)

            search_results = []
            for result in results:
                search_results.append({
                    "file_name": result[0],
                    "file_stem": result[1],
                    "document_id": result[2],
                    "chunk_index": result[3],
                    "content": result[4],
                    "distance": result[5]
                })

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        pg_stats = self.pg_handler.get_document_stats()

        return {
            "postgresql": pg_stats,
            "embedding_model": self.embedding_model_name
        }

    def close(self):
        """Close connections"""
        self.pg_handler.close()


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Book Vectorization with LangChain and PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options
    parser.add_argument("--file", "-f", help="Path to a single file to process")
    parser.add_argument("--directory", "-d", help="Path to directory to process")
    parser.add_argument("--no-recursive", action="store_true", help="Don't process subdirectories")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing (parallel is enabled by default with 3 threads)")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel worker threads (default: 3)")
    parser.add_argument("--search", "-s", help="Search query")
    parser.add_argument("--stats", action="store_true", help="Show processing statistics")
    parser.add_argument("--cleanup", action="store_true", help="Drop all data and recreate database schema")

    # Configuration options
    parser.add_argument("--from-env", action="store_true", help="Load config from .env file")

    # Database options
    parser.add_argument("--pg-host", help="PostgreSQL host")
    parser.add_argument("--pg-port", type=int, help="PostgreSQL port")
    parser.add_argument("--pg-user", help="PostgreSQL user")
    parser.add_argument("--pg-password", help="PostgreSQL password")
    parser.add_argument("--pg-database", help="PostgreSQL database")

    # Embedding options
    parser.add_argument("--embedding-mode", choices=["sentence-transformers", "ollama", "openai"],
                       help="Embedding mode")
    parser.add_argument("--embedding-model", help="SentenceTransformers model name")
    parser.add_argument("--ollama-host", help="Ollama host")
    parser.add_argument("--ollama-port", type=int, help="Ollama port")

    # Processing options
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    if args.from_env:
        config = {
            'pg_host': args.pg_host or os.getenv('DB_HOST', 'localhost'),
            'pg_port': args.pg_port or int(os.getenv('DB_PORT', '5432')),
            'pg_user': args.pg_user or os.getenv('DB_USER', 'postgres'),
            'pg_password': args.pg_password or os.getenv('DB_PASS', ''),
            'pg_database': args.pg_database or os.getenv('DB_RAG_NAME', 'myMarketSignal'),
            'embedding_mode': args.embedding_mode or os.getenv('EMB_MODE', 'sentence-transformers'),
            'embedding_model': args.embedding_model or os.getenv('EMB_MODEL', 'nomic-embed-text:latest'),
            'ollama_host': args.ollama_host or os.getenv('EMB_HOST', 'localhost'),
            'ollama_port': args.ollama_port or int(os.getenv('EMB_PORT', '11434')),
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.chunk_overlap
        }
    else:
        config = {
            'pg_host': args.pg_host or 'localhost',
            'pg_port': args.pg_port or 5432,
            'pg_user': args.pg_user or 'postgres',
            'pg_password': args.pg_password or '',
            'pg_database': args.pg_database or 'myMarketSignal',
            'embedding_mode': args.embedding_mode or 'sentence-transformers',
            'embedding_model': args.embedding_model or 'nomic-embed-text:latest',
            'ollama_host': args.ollama_host or 'localhost',
            'ollama_port': args.ollama_port or 11434,
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.chunk_overlap
        }

    # Validate required arguments
    if not any([args.file, args.directory, args.search, args.stats, args.cleanup]):
        parser.print_help()
        sys.exit(1)

    try:
        # Initialize vectorizer
        vectorizer = BookVectorizer(**config)

        print(f"📋 Configuration:")
        print(f"   PostgreSQL: {config['pg_host']}:{config['pg_port']}/{config['pg_database']}")
        print(f"   Embedding: {config['embedding_mode']} ({config.get('embedding_model') or config.get('ollama_model')})")

        # Execute operations
        if args.stats:
            print("\n📊 Statistics:")
            stats = vectorizer.get_stats()
            pg_stats = stats.get('postgresql', {})

            print(f"   Documents: {pg_stats.get('total_documents', 0)}")
            print(f"   Chunks: {pg_stats.get('total_chunks', 0) or 0}")
            print(f"   Avg chunks/doc: {pg_stats.get('avg_chunks_per_doc', 0) or 0:.1f}")
            print(f"   Last processed: {pg_stats.get('last_processed', 'Never')}")

        elif args.search:
            print(f"Searching for: '{args.search}'")
            try:
                results = vectorizer.search_documents(args.search, limit=10)
                print(f"Search returned {len(results)} results")

                if results:
                    print(f"\nFound {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {result['file_name']} (Distance: {result['distance']:.3f})")
                        print(f"   Chunk {result['chunk_index']}: {result['content'][:200]}...")
                else:
                    print("No results found")
            except Exception as e:
                print(f"Search error: {e}")
                import traceback
                traceback.print_exc()

        elif args.file:
            print(f"\n📄 Processing file: {args.file}")
            start_time = time.time()
            success = vectorizer.process_file(args.file)
            elapsed = time.time() - start_time

            if success:
                print(f"✅ Successfully processed in {elapsed:.2f} seconds")
            else:
                print("❌ Failed to process file")
                sys.exit(1)

        elif args.directory:
            print(f"\n📁 Processing directory: {args.directory}")
            recursive = not args.no_recursive
            print(f"   Recursive: {recursive}")
            if not args.no_parallel:
                print(f"   Parallel: True (workers={args.workers})")
            else:
                print(f"   Parallel: False")

            start_time = time.time()
            results = vectorizer.process_directory(args.directory, recursive, not args.no_parallel, args.workers)
            elapsed = time.time() - start_time

            print(f"\n📊 Results:")
            print(f"   ✅ Success: {results['success']}")
            print(f"   ❌ Failed: {results['failed']}")
            print(f"   ⏭️ Skipped: {results['skipped']}")
            print(f"   ⏱️ Time: {elapsed:.2f} seconds")
        elif args.cleanup:
            confirm = input("This will drop all data and recreate the schema. Type 'yes' to confirm: ")
            if confirm.lower() == 'yes':
                try:
                    vectorizer.pg_handler.cleanup_and_recreate_schema()
                    print("Schema cleanup and recreation completed successfully")
                except Exception as e:
                    print(f"Error during cleanup and recreation: {e}")
            else:
                print("Cleanup and recreation aborted")

        # Close connections
        vectorizer.close()

    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
