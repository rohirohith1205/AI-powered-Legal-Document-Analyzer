from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from mychunk import Chunk
from chunking_system import ChunkingSystem
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ContractAnalysisRAG:
    """A Retrieval-Augmented Generation system for contract analysis using semantic search."""
    def __init__(self, use_semantic_search: bool = True):
        """
        Initialize the RAG system with a SentenceTransformer model and chunking system.
        
        Args:
            use_semantic_search (bool): Whether to use semantic search for retrieval.
        
        Raises:
            RuntimeError: If the SentenceTransformer model fails to load.
        """
        try:
            self.encoder = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")
        self.chunker = ChunkingSystem()
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.use_semantic_search = use_semantic_search

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the RAG system and compute embeddings if semantic search is enabled.
        
        Args:
            documents: List of dictionaries with 'text' and 'metadata' keys.
        
        Raises:
            ValueError: If documents are not in the expected format.
        """
        if not isinstance(documents, list):
            raise ValueError("Documents must be a list of dictionaries")
        
        for doc in documents:
            if not isinstance(doc, dict) or 'text' not in doc or 'metadata' not in doc:
                raise ValueError("Each document must be a dict with 'text' and 'metadata' keys")
            chunks = self.chunker.chunk_document(doc['text'], doc['metadata'])
            self.chunks.extend(chunks)
        
        if self.chunks and self.use_semantic_search:
            chunk_texts = [chunk.text for chunk in self.chunks]
            try:
                self.embeddings = self.encoder.encode(chunk_texts, show_progress_bar=False, batch_size=32)
            except Exception as e:
                logger.error(f"Failed to compute embeddings: {e}")
                raise RuntimeError(f"Failed to compute embeddings: {e}")

    def retrieve(self, query: str, similarity_threshold: float = 0.5) -> Optional[Chunk]:
        """
        Retrieve the most relevant chunk for a given query using semantic search.
        
        Args:
            query: The query string to match against chunks.
            similarity_threshold: Minimum similarity score to return a chunk.
        
        Returns:
            Optional[Chunk]: The most relevant chunk if similarity exceeds threshold, else None.
        
        Raises:
            ValueError: If query is not a non-empty string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if not self.chunks or self.embeddings is None:
            logger.warning("No chunks or embeddings available for retrieval")
            return None
        try:
            query_embedding = self.encoder.encode([query])[0]
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                logger.warning("Query embedding has zero norm, cannot compute similarity")
                return None
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * query_norm
            )
            best_idx = np.argmax(similarities)
            return self.chunks[best_idx] if similarities[best_idx] > similarity_threshold else None
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return None