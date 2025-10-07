"""
ArquimedesAI Configuration Settings.

Manages all configuration via environment variables and Pydantic Settings.
Following spec §5.1-5.2 for contract-first development.
"""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden via .env file or environment variables
    with ARQ_ prefix (e.g., ARQ_DATA_DIR=/path/to/data).
    
    Attributes:
        data_dir: Directory containing user documents for indexing
        storage_dir: Directory for persistent vector store and embeddings cache
        db: Vector database backend (qdrant or faiss)
        qdrant_url: Qdrant server URL for remote mode
        qdrant_path: Local path for Qdrant persistence (:memory: for in-memory)
        embed_model: HuggingFace model ID for embeddings
        top_k: Number of results to return from retrieval
        fetch_k: Number of candidates to fetch before re-ranking
        chunk_size: Maximum characters per document chunk
        chunk_overlap: Character overlap between chunks
        hybrid: Enable hybrid retrieval (dense + sparse)
        ollama_base: Ollama HTTP API base URL
        ollama_model: Ollama model name (e.g., gemma3:latest)
        ollama_temperature: LLM generation temperature
        discord_token: Discord bot authentication token
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ARQ_",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Paths
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory containing documents to index"
    )
    storage_dir: Path = Field(
        default=Path("./storage"),
        description="Directory for vector store and cache persistence"
    )
    
    # Vector Database
    db: Literal["qdrant", "faiss"] = Field(
        default="qdrant",
        description="Vector database backend"
    )
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL (e.g., http://localhost:6333)"
    )
    qdrant_path: str = Field(
        default="./storage/qdrant",
        description="Local Qdrant persistence path or :memory:"
    )
    
    # Qdrant HNSW Optimization (v1.3.1)
    qdrant_hnsw_m: int = Field(
        default=16,
        ge=4,
        le=64,
        description="HNSW graph connectivity (higher = better accuracy, more memory)"
    )
    qdrant_hnsw_ef_construct: int = Field(
        default=100,
        ge=4,
        le=512,
        description="HNSW build quality (higher = better index, slower indexing)"
    )
    qdrant_on_disk: bool = Field(
        default=False,
        description="Store vectors on disk (lower memory, slightly slower queries)"
    )
    
    # Embeddings
    embed_model: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace embedding model ID"
    )
    
    # Retrieval Parameters
    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of final results to return"
    )
    fetch_k: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Number of candidates to fetch before re-ranking"
    )
    
    # Chunking (DEPRECATED: Now handled by Docling HybridChunker)
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=2000,
        description="DEPRECATED: Docling HybridChunker uses tokenization-aware chunking"
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="DEPRECATED: Docling HybridChunker uses hierarchical chunking"
    )
    
    # Docling Configuration (v1.3)
    docling_ocr: bool = Field(
        default=False,
        description="Enable OCR for scanned PDFs and images"
    )
    docling_table_mode: Literal["fast", "accurate"] = Field(
        default="accurate",
        description="Table extraction mode (accurate recommended for quality)"
    )
    
    # Features
    hybrid: bool = Field(
        default=True,
        description="Enable hybrid retrieval (dense + sparse)"
    )
    bm25_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 (sparse) retriever in hybrid mode"
    )
    dense_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for dense (semantic) retriever in hybrid mode"
    )
    rerank_enabled: bool = Field(
        default=False,
        description="Enable cross-encoder reranking after retrieval"
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HuggingFace cross-encoder model for reranking"
    )
    rerank_top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to return after reranking (increased to better utilize 128K context window)"
    )
    
    # LLM (Ollama)
    ollama_base: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    ollama_model: str = Field(
        default="gemma3:latest",
        description="Ollama model name"
    )
    ollama_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="LLM generation temperature (0=deterministic, 1=creative)"
    )
    
    # Conversational Memory (v1.4 Phase 1)
    enable_conversation_memory: bool = Field(
        default=False,
        description="Enable in-session conversational memory (RunnableWithMessageHistory)"
    )
    max_history_messages: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of messages to keep in conversation history"
    )
    
    # Semantic Routing (v2.0)
    disable_routing: bool = Field(
        default=False,
        env="ARQ_DISABLE_ROUTING",
        description="Disable semantic routing (default: False, routing enabled)"
    )
    
    # Structured Citations (v1.4 Phase 1)
    use_structured_citations: bool = Field(
        default=False,
        description="Enable structured citations with Pydantic schemas (QuotedAnswer)"
    )
    citation_style: Literal["quoted", "id"] = Field(
        default="quoted",
        description="Citation style: 'quoted' (with verbatim quotes) or 'id' (source IDs only)"
    )
    
    # Discord Bot
    discord_token: SecretStr | None = Field(
        default=None,
        description="Discord bot authentication token"
    )
    
    @property
    def embeddings_cache_path(self) -> Path:
        """Derived path for embeddings cache."""
        return self.storage_dir / "embeddings_cache"
    
    @property
    def qdrant_collection_name(self) -> str:
        """Collection name for Qdrant vector store."""
        return "arquimedes_chunks"
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_cache_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Debug: Log loaded settings on import (helpful for troubleshooting .env issues)
import logging
_logger = logging.getLogger(__name__)
_logger.debug("=" * 60)
_logger.debug("ArquimedesAI Settings Loaded")
_logger.debug("=" * 60)
_logger.debug(f"Ollama Model: {settings.ollama_model}")
_logger.debug(f"Ollama Base URL: {settings.ollama_base}")
_logger.debug(f"Conversation Memory: {settings.enable_conversation_memory}")
_logger.debug(f"Structured Citations: {settings.use_structured_citations}")
_logger.debug(f"Data Directory: {settings.data_dir}")
_logger.debug(f"Qdrant Path: {settings.qdrant_path}")
_logger.debug("=" * 60)
