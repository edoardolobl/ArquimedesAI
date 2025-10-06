"""
RAG chain orchestration using LangChain LCEL.

Combines retrieval and generation into a unified chain for question answering.
"""

import logging
from typing import Dict, Any

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from core.llm_local import LLMManager
from core.vector_store import QdrantVectorStore
from core.hybrid_retriever import create_hybrid_retriever
from core.reranker import create_reranking_retriever
from prompts.templates import SYSTEM_PROMPT, GROUNDED_PROMPT
from settings import settings

logger = logging.getLogger(__name__)


class RAGChain:
    """
    Retrieval-Augmented Generation chain.
    
    Orchestrates retrieval from Qdrant and generation with Ollama LLM
    using LangChain Expression Language (LCEL).
    
    Attributes:
        vector_store: Qdrant vector store instance
        llm_manager: LLM manager instance
        retrieval_chain: Configured LCEL retrieval chain
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        llm_manager: LLMManager | None = None,
        prompt_template: str | None = None,
    ):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: Vector store instance (creates new if None)
            llm_manager: LLM manager instance (creates new if None)
            prompt_template: Custom prompt template string (uses GROUNDED_PROMPT if None)
        """
        logger.info("Initializing RAG chain")
        
        # Initialize components
        self.vector_store = vector_store or QdrantVectorStore()
        self.llm_manager = llm_manager or LLMManager()
        
        # Create prompt template
        self.prompt = self._create_prompt(prompt_template)
        
        # Build chain
        self.retrieval_chain = self._build_chain()
        
        logger.info("✓ RAG chain ready")
    
    def _create_prompt(self, custom_template: str | None = None) -> ChatPromptTemplate:
        """
        Create prompt template for grounded answers.
        
        Args:
            custom_template: Optional custom prompt template string
        
        Returns:
            ChatPromptTemplate with system and user messages
        """
        # Use custom template or default GROUNDED_PROMPT
        template = custom_template or GROUNDED_PROMPT
        
        return ChatPromptTemplate.from_template(template)
    
    def _build_chain(self):
        """
        Build LCEL retrieval chain.
        
        Returns:
            Configured retrieval chain
        """
        logger.info("Building LCEL chain")
        
        # Get retriever (hybrid or dense-only based on settings)
        base_retriever = create_hybrid_retriever(self.vector_store)
        
        # Optionally wrap with reranking (v1.2 feature)
        retriever = create_reranking_retriever(base_retriever)
        
        if settings.rerank_enabled:
            logger.info(f"✓ Reranking enabled with {settings.rerank_model}")
        
        # Create document chain (combines docs with LLM)
        document_chain = create_stuff_documents_chain(
            llm=self.llm_manager.get_llm(),
            prompt=self.prompt,
        )
        
        # Create retrieval chain (retriever + document chain)
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain,
        )
        
        logger.info("✓ Chain built successfully")
        return retrieval_chain
    
    async def ainvoke(self, query: str) -> Dict[str, Any]:
        """
        Async invoke RAG chain.
        
        Args:
            query: User question
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        logger.info(f"Query: {query}")
        
        input_data = {"input": query}
        result = await self.retrieval_chain.ainvoke(input_data)
        
        logger.info("✓ Query processed")
        return result
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Synchronous invoke RAG chain.
        
        Args:
            query: User question
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        logger.info(f"Query: {query}")
        
        input_data = {"input": query}
        result = self.retrieval_chain.invoke(input_data)
        
        logger.info("✓ Query processed")
        return result


def load_rag_chain() -> RAGChain:
    """
    Load RAG chain with existing vector store.
    
    Convenience function to load chain from persisted collection.
    
    Returns:
        Initialized RAG chain
    """
    logger.info("Loading RAG chain from existing collection")
    
    vector_store = QdrantVectorStore()
    vector_store.load_existing()
    
    return RAGChain(vector_store=vector_store)
