"""
Embeddings module using HuggingFace models with caching.

Provides BGE-M3 multilingual embeddings with local file-based caching
to avoid recomputing embeddings for the same content.
"""

import logging
from pathlib import Path

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_huggingface import HuggingFaceEmbeddings

from settings import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages document embeddings with caching.
    
    Uses HuggingFace BGE-M3 model for multilingual support.
    Implements file-based caching to speed up repeated operations.
    
    Attributes:
        model_name: HuggingFace model identifier
        cache_path: Path to embedding cache directory
        embedder: Cached embedding instance
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        cache_path: Path | None = None
    ):
        """
        Initialize embedding manager with caching.
        
        Args:
            model_name: HuggingFace model ID (uses settings default if None)
            cache_path: Cache directory path (uses settings default if None)
        """
        self.model_name = model_name or settings.embed_model
        self.cache_path = cache_path or settings.embeddings_cache_path
        
        logger.info(f"Initializing embeddings: {self.model_name}")
        logger.info(f"Cache path: {self.cache_path}")
        
        # Ensure cache directory exists
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize base embeddings
        self.base_embedder = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},  # Use CPU for compatibility
            encode_kwargs={"normalize_embeddings": True},  # L2 normalization
        )
        
        # Wrap with cache (using SHA-256 for security)
        store = LocalFileStore(str(self.cache_path))
        self.embedder = CacheBackedEmbeddings.from_bytes_store(
            self.base_embedder,
            store,
            namespace=self.model_name.replace("/", "_"),
            key_encoder="sha256",  # Use SHA-256 instead of SHA-1
        )
        
        logger.info("✓ Embeddings initialized with caching")
    
    def get_embedder(self) -> CacheBackedEmbeddings:
        """
        Get the cached embedder instance.
        
        Returns:
            CacheBackedEmbeddings instance ready for use with vector stores
        """
        return self.embedder
    
    @property
    def embedding_dimension(self) -> int:
        """
        Get embedding vector dimension.
        
        Returns:
            Dimension of embedding vectors (1024 for BGE-M3)
        """
        # BGE-M3 produces 1024-dimensional embeddings
        if "bge-m3" in self.model_name.lower():
            return 1024
        # Default fallback
        return 768
