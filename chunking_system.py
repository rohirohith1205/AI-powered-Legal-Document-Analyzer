from typing import List, Dict, Any
from mychunk import Chunk

class ChunkingSystem:
    """A system for splitting legal documents into chunks with overlap."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunking system for legal documents.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters.
            chunk_overlap (int): Number of overlapping characters between chunks.
        
        Raises:
            ValueError: If chunk_overlap is not less than chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Split a document into chunks with specified size and overlap.
        
        Args:
            text (str): The document text to chunk.
            metadata (Dict[str, Any], optional): Metadata to attach to each chunk (e.g., document_type, source).
        
        Returns:
            List[Chunk]: List of Chunk objects containing text and metadata.
        
        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
        if not text.strip():
            return []

        chunks = []
        text_length = len(text)
        start = 0
        chunk_id = 0
        max_word_length = self.chunk_size // 2  # Arbitrary limit to detect overly long words

        while start < text_length:
            # Calculate end index for the chunk
            end = min(start + self.chunk_size, text_length)
            
            # Adjust end to avoid splitting mid-word
            if end < text_length:
                original_end = end
                while end > start and text[end - 1] not in ' \n\t':
                    end -= 1
                if end == start:
                    # Handle case where a single word is too long
                    if original_end - start > max_word_length:
                        end = start + max_word_length  # Force split if word is too long
                    else:
                        end = original_end
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            if chunk_text:
                # Create chunk with metadata
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['chunk_id'] = chunk_id
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
                chunk_id += 1
            
            # Move start for next chunk
            start += self.chunk_size - self.chunk_overlap
            if start >= text_length:
                break

        return chunks