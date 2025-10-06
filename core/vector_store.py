"""
Qdrant vector store integration.

Manages Qdrant collections for document chunk storage and retrieval.
Supports both local persistence and remote server modes.
"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore as LangChainQdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

from core.embedder import EmbeddingManager
from settings import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant vector store manager.
    
    Handles collection creation, document ingestion, and retrieval
    using Qdrant for persistent vector storage.
    
    Attributes:
        collection_name: Name of Qdrant collection
        client: Qdrant client instance
        embedder: Embedding manager instance
        vector_store: LangChain Qdrant wrapper
    """
    
    def __init__(
        self,
        collection_name: str | None = None,
        path: str | None = None,
        url: str | None = None,
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            collection_name: Collection name (uses settings default if None)
            path: Local persistence path or :memory: (uses settings if None)
            url: Remote Qdrant URL (uses local if None)
        """
        self.collection_name = collection_name or settings.qdrant_collection_name
        
        # Initialize embeddings
        self.embedding_manager = EmbeddingManager()
        self.embedder = self.embedding_manager.get_embedder()
        
        # Determine connection mode
        if url or settings.qdrant_url:
            # Remote mode
            qdrant_url = url or settings.qdrant_url
            logger.info(f"Connecting to remote Qdrant: {qdrant_url}")
            self.client = QdrantClient(url=qdrant_url)
        else:
            # Local mode
            qdrant_path = path or settings.qdrant_path
            logger.info(f"Using local Qdrant: {qdrant_path}")
            self.client = QdrantClient(path=qdrant_path)
        
        logger.info(f"Collection: {self.collection_name}")
        self.vector_store = None
    
    def create_collection(self, force_recreate: bool = False) -> None:
        """
        Create or recreate Qdrant collection.
        
        Args:
            force_recreate: If True, delete existing collection and recreate
        """
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists:
            if force_recreate:
                logger.warning(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                return
        
        logger.info(f"Creating collection: {self.collection_name}")
        
        # Configure HNSW index for optimized retrieval (v1.3.1)
        # See: https://qdrant.tech/documentation/guides/configuration/
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_manager.embedding_dimension,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=settings.qdrant_hnsw_m,  # Graph connectivity
                    ef_construct=settings.qdrant_hnsw_ef_construct,  # Build quality
                ),
                on_disk=settings.qdrant_on_disk,  # Memory vs speed tradeoff
            ),
        )
        
        logger.info(
            f"✓ Collection created with HNSW: m={settings.qdrant_hnsw_m}, "
            f"ef_construct={settings.qdrant_hnsw_ef_construct}, "
            f"on_disk={settings.qdrant_on_disk}"
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the collection.
        
        Args:
            documents: List of document chunks to add
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to collection")
        
        # Create vector store with existing client and collection
        # Use direct instantiation (not from_documents) when we already have client+collection
        self.vector_store = LangChainQdrant(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedder,
        )
        
        # Add documents to the collection
        self.vector_store.add_documents(documents)
        
        logger.info("✓ Documents added successfully")
    
    def load_existing(self) -> None:
        """Load existing collection as LangChain vector store."""
        logger.info(f"Loading existing collection: {self.collection_name}")
        
        # Use direct instantiation with existing client
        self.vector_store = LangChainQdrant(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedder,
        )
        
        logger.info("✓ Collection loaded")
    
    def as_retriever(self, search_kwargs: dict | None = None):
        """
        Get LangChain retriever interface.
        
        Args:
            search_kwargs: Search parameters (k, score_threshold, etc.)
            
        Returns:
            LangChain retriever instance
        """
        if not self.vector_store:
            self.load_existing()
        
        if search_kwargs is None:
            search_kwargs = {"k": settings.top_k}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def get_stats(self) -> dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection info (count, size, etc.)
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
