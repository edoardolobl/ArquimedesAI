"""
Cross-Encoder Reranker for ArquimedesAI.

Implements document reranking using HuggingFace cross-encoder models
to improve retrieval quality. Following LangChain ContextualCompression pattern.
"""

from typing import TYPE_CHECKING

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from settings import settings

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever


class RerankerManager:
    """
    Manages cross-encoder reranking for document retrieval.
    
    Wraps a base retriever with a ContextualCompressionRetriever that uses
    a cross-encoder model to rerank results based on query relevance.
    
    Example:
        >>> base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        >>> reranker = RerankerManager()
        >>> compression_retriever = reranker.create_compression_retriever(base_retriever)
        >>> docs = compression_retriever.invoke("What is the main topic?")
    
    Attributes:
        model_name: HuggingFace model ID for cross-encoder
        top_n: Number of documents to return after reranking
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        top_n: int | None = None,
    ):
        """
        Initialize reranker with cross-encoder model.
        
        Args:
            model_name: HuggingFace cross-encoder model ID.
                Defaults to settings.rerank_model (BAAI/bge-reranker-v2-m3).
            top_n: Number of documents to return after reranking.
                Defaults to settings.rerank_top_n (3).
        """
        self.model_name = model_name or settings.rerank_model
        self.top_n = top_n or settings.rerank_top_n
        self._cross_encoder = None
        self._compressor = None
    
    @property
    def cross_encoder(self) -> HuggingFaceCrossEncoder:
        """
        Lazy-load HuggingFace cross-encoder model.
        
        Returns:
            Initialized HuggingFaceCrossEncoder instance.
        """
        if self._cross_encoder is None:
            self._cross_encoder = HuggingFaceCrossEncoder(
                model_name=self.model_name
            )
        return self._cross_encoder
    
    @property
    def compressor(self) -> CrossEncoderReranker:
        """
        Get or create document compressor.
        
        Returns:
            CrossEncoderReranker configured with cross-encoder model.
        """
        if self._compressor is None:
            self._compressor = CrossEncoderReranker(
                model=self.cross_encoder,
                top_n=self.top_n
            )
        return self._compressor
    
    def create_compression_retriever(
        self,
        base_retriever: "BaseRetriever"
    ) -> ContextualCompressionRetriever:
        """
        Wrap base retriever with contextual compression for reranking.
        
        The compression retriever will:
        1. Fetch candidates from base retriever (e.g., top-20)
        2. Rerank using cross-encoder based on query-document relevance
        3. Return top-N most relevant documents
        
        Args:
            base_retriever: Retriever to wrap (hybrid or dense).
        
        Returns:
            ContextualCompressionRetriever with reranking enabled.
        """
        return ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )


def create_reranking_retriever(
    base_retriever: "BaseRetriever",
    enabled: bool | None = None,
    model_name: str | None = None,
    top_n: int | None = None,
) -> "BaseRetriever":
    """
    Create retriever with optional reranking.
    
    Convenience function that wraps base retriever with reranking if enabled,
    otherwise returns base retriever unchanged.
    
    Args:
        base_retriever: Base retriever to wrap.
        enabled: Enable reranking. Defaults to settings.rerank_enabled.
        model_name: Cross-encoder model. Defaults to settings.rerank_model.
        top_n: Number of results. Defaults to settings.rerank_top_n.
    
    Returns:
        Retriever with or without reranking based on enabled flag.
    
    Example:
        >>> from core.hybrid_retriever import create_hybrid_retriever
        >>> from core.reranker import create_reranking_retriever
        >>> 
        >>> base = create_hybrid_retriever(vector_store)
        >>> retriever = create_reranking_retriever(base, enabled=True)
    """
    if enabled is None:
        enabled = settings.rerank_enabled
    
    if not enabled:
        return base_retriever
    
    reranker = RerankerManager(model_name=model_name, top_n=top_n)
    return reranker.create_compression_retriever(base_retriever)
