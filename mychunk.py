from typing import Dict, Any, Optional

class Chunk:
    """A class to represent a chunk of text with associated metadata."""
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}