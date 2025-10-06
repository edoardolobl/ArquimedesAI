"""
Hybrid retriever combining dense (semantic) and sparse (BM25) retrieval.

Implements EnsembleRetriever pattern per spec §8 for improved retrieval quality.
"""

import logging
from typing import List, Dict, Any

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, Runnable

from core.vector_store import QdrantVectorStore
from settings import settings

logger = logging.getLogger(__name__)


class InputKeyExtractorRetriever(BaseRetriever):
    """
    Wrapper retriever that extracts 'input' key from dict before passing to child retriever.
    
    This handles cases where create_retrieval_chain passes a dict but the child
    retriever expects a string.
    """
    
    retriever: BaseRetriever
    
    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        """Get relevant documents, ensuring query is a string."""
        # If query is a dict (shouldn't happen but defensive), extract 'input' key
        if isinstance(query, dict):
            query = query.get("input", query)
        
        # Use invoke instead of deprecated get_relevant_documents
        config = {"callbacks": run_manager.get_child()} if run_manager else None
        return self.retriever.invoke(query, config=config)
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        """Async get relevant documents."""
        # If query is a dict (shouldn't happen but defensive), extract 'input' key  
        if isinstance(query, dict):
            query = query.get("input", query)
        
        # Use ainvoke instead of deprecated aget_relevant_documents
        config = {"callbacks": run_manager.get_child()} if run_manager else None
        return await self.retriever.ainvoke(query, config=config)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 (sparse) and dense (semantic) retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both retrievers.
    BM25 excels at keyword matching, while dense retrieval handles semantic similarity.
    
    Attributes:
        vector_store: Qdrant vector store for dense retrieval
        bm25_retriever: BM25 retriever for sparse keyword matching
        ensemble_retriever: Combined retriever using RRF
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_weight: float | None = None,
        dense_weight: float | None = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Qdrant vector store instance
            bm25_weight: Weight for BM25 retriever (uses settings if None)
            dense_weight: Weight for dense retriever (uses settings if None)
        """
        logger.info("Initializing hybrid retriever")
        
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight or settings.bm25_weight
        self.dense_weight = dense_weight or settings.dense_weight
        
        # Create ensemble retriever
        self.ensemble_retriever = self._create_ensemble()
        
        logger.info(f"✓ Hybrid retriever ready (BM25: {self.bm25_weight}, Dense: {self.dense_weight})")
    
    def _load_documents_from_store(self) -> List[Document]:
        """
        Load all documents from Qdrant for BM25 indexing.
        
        BM25Retriever needs access to original documents, not just embeddings.
        
        Returns:
            List of Document objects from the vector store
        """
        logger.info("Loading documents from Qdrant for BM25 indexing...")
        
        try:
            # Retrieve all documents from collection
            # Note: This uses scroll API to get all documents
            client = self.vector_store.client
            collection_name = settings.qdrant_collection_name
            
            # Scroll through all points
            documents = []
            offset = None
            batch_size = 100
            
            while True:
                records, next_offset = client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # Don't need vectors, just text
                )
                
                if not records:
                    break
                
                # Convert Qdrant records to LangChain Documents
                for record in records:
                    doc = Document(
                        page_content=record.payload.get("page_content", ""),
                        metadata=record.payload.get("metadata", {}),
                    )
                    documents.append(doc)
                
                if next_offset is None:
                    break
                
                offset = next_offset
            
            logger.info(f"✓ Loaded {len(documents)} documents for BM25")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load documents from Qdrant: {e}")
            logger.warning("Falling back to dense-only retrieval")
            return []
    
    def _create_ensemble(self) -> BaseRetriever:
        """
        Create ensemble retriever combining BM25 and dense retrievers.
        
        Returns:
            EnsembleRetriever with configured weights
        """
        logger.info("Creating ensemble retriever...")
        
        # Get dense retriever from Qdrant
        dense_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": settings.top_k}
        )
        
        # Load documents for BM25
        documents = self._load_documents_from_store()
        
        if not documents:
            logger.warning("No documents available for BM25, using dense-only retrieval")
            return dense_retriever
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=settings.top_k,
        )
        
        # Wrap BM25 retriever to ensure it receives string input
        # (handles edge cases where dict might be passed)
        wrapped_bm25 = InputKeyExtractorRetriever(retriever=bm25_retriever)
        
        # Combine with EnsembleRetriever
        ensemble = EnsembleRetriever(
            retrievers=[wrapped_bm25, dense_retriever],
            weights=[self.bm25_weight, self.dense_weight],
        )
        
        logger.info("✓ Ensemble retriever created")
        return ensemble
    
    def as_retriever(self) -> BaseRetriever:
        """
        Get the configured retriever for use in chains.
        
        Returns:
            BaseRetriever instance (EnsembleRetriever or fallback)
        """
        return self.ensemble_retriever


def create_hybrid_retriever(vector_store: QdrantVectorStore) -> BaseRetriever:
    """
    Convenience function to create hybrid retriever.
    
    Args:
        vector_store: Qdrant vector store instance
        
    Returns:
        Configured retriever (hybrid or dense-only based on settings)
    """
    if settings.hybrid:
        logger.info("Creating hybrid retriever (BM25 + dense)")
        hybrid = HybridRetriever(vector_store)
        return hybrid.as_retriever()
    else:
        logger.info("Creating dense-only retriever")
        return vector_store.as_retriever(search_kwargs={"k": settings.top_k})
