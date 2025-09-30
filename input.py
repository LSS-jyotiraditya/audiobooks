import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
import PyPDF2
from docx import Document
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EbookProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, db_path: str = "./vector_db"):
        """
        Initialize the ebook processor with vector database and embedding model.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
            db_path: Path to store the vector database
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_path = db_path
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name="ebook_chunks",
                metadata={"description": "Chunked ebook content for RAG"}
            )
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text content from Word document."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from DOCX: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If we're not at the end of the text, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary (. ! ?)
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def store_chunks_in_vector_db(self, chunks: List[str], metadata: dict) -> None:
        """
        Store text chunks in vector database with embeddings.
        
        Args:
            chunks: List of text chunks
            metadata: Metadata about the source document
        """
        try:
            if not chunks:
                logger.warning("No chunks to store")
                return
            
            # Generate embeddings for all chunks
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks),
                    'created_at': datetime.now().isoformat()
                }
                metadatas.append(chunk_metadata)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
            
        except Exception as e:
            logger.error(f"Failed to store chunks in vector database: {e}")
            raise
    
    def process_ebook(self, file_path: str) -> Dict:
        """
        Main method to process an ebook file.
        
        Args:
            file_path: Path to the ebook file
            
        Returns:
            Dictionary with processing results
        """
        try:
            filename = os.path.basename(file_path)
            
            # Determine file type and extract text
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                text = self.extract_text_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            if not text:
                raise ValueError("No text content found in the file")
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            if not chunks:
                raise ValueError("No chunks generated from the text")
            
            # Prepare metadata
            metadata = {
                'filename': filename,
                'file_path': file_path,
                'file_type': file_extension,
                'file_size': os.path.getsize(file_path),
                'text_length': len(text),
                'processing_date': datetime.now().isoformat()
            }
            
            # Store in vector database
            self.store_chunks_in_vector_db(chunks, metadata)
            
            return {
                'success': True,
                'filename': filename,
                'chunks_created': len(chunks),
                'text_length': len(text),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to process ebook {file_path}: {e}")
            return {
                'success': False,
                'filename': os.path.basename(file_path),
                'error': str(e)
            }
    
    def search_similar_chunks(self, query: str, n_results: int = 5) -> Dict:
        """
        Search for similar chunks in the vector database.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            return {
                'success': True,
                'query': query,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Failed to search chunks: {e}")
            return {
                'success': False,
                'query': query,
                'error': str(e)
            }
    
    def get_database_info(self) -> Dict:
        """Get information about the vector database."""
        try:
            collection_count = self.collection.count()
            
            info = {
                'total_chunks': collection_count,
                'database_path': self.db_path
            }
            
            # Get sample metadata if chunks exist
            if collection_count > 0:
                sample_results = self.collection.get(limit=10)
                
                if sample_results['metadatas']:
                    files_info = {}
                    for metadata in sample_results['metadatas']:
                        filename = metadata['filename']
                        if filename not in files_info:
                            files_info[filename] = {
                                'chunks': 0,
                                'file_type': metadata['file_type'],
                                'processing_date': metadata['processing_date'],
                                'text_length': metadata['text_length']
                            }
                        files_info[filename]['chunks'] += 1
                    
                    info['files'] = files_info
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {'error': str(e)}
    
    def delete_file_chunks(self, filename: str) -> Dict:
        """Delete all chunks for a specific file."""
        try:
            # Get all chunks for the file
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for file: {filename}")
                return {
                    'success': True,
                    'deleted_chunks': len(results['ids']),
                    'filename': filename
                }
            else:
                return {
                    'success': False,
                    'error': f"No chunks found for file: {filename}"
                }
                
        except Exception as e:
            logger.error(f"Failed to delete chunks for file {filename}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

