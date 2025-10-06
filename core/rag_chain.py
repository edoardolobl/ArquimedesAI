"""
RAG chain orchestration using LangChain LCEL.

Combines retrieval and generation into a unified chain for question answering.
Supports conversational memory (v1.4) and structured citations (v1.4).
"""

import logging
from typing import Dict, Any

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from core.llm_local import LLMManager
from core.vector_store import QdrantVectorStore
from core.hybrid_retriever import create_hybrid_retriever
from core.reranker import create_reranking_retriever
from prompts.templates import (
    SYSTEM_PROMPT, 
    GROUNDED_PROMPT,
    QuotedAnswer,
    CitedAnswer,
    format_docs_with_id,
)
from settings import settings

logger = logging.getLogger(__name__)


class RAGChain:
    """
    Retrieval-Augmented Generation chain.
    
    Orchestrates retrieval from Qdrant and generation with Ollama LLM
    using LangChain Expression Language (LCEL).
    
    Features:
    - Hybrid retrieval (BM25 + Dense)
    - Optional cross-encoder reranking
    - Conversational memory (v1.4)
    - Structured citations (v1.4)
    
    Attributes:
        vector_store: Qdrant vector store instance
        llm_manager: LLM manager instance
        retrieval_chain: Configured LCEL retrieval chain
        history_store: Session-based chat history storage (v1.4)
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        llm_manager: LLMManager | None = None,
        prompt_template: str | None = None,
        use_structured_output: bool = False,
    ):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: Vector store instance (creates new if None)
            llm_manager: LLM manager instance (creates new if None)
            prompt_template: Custom prompt template string (uses GROUNDED_PROMPT if None)
            use_structured_output: Enable structured citations with Pydantic (v1.4)
        """
        logger.info("Initializing RAG chain")
        
        # Initialize components
        self.vector_store = vector_store or QdrantVectorStore()
        self.llm_manager = llm_manager or LLMManager()
        self.use_structured_output = use_structured_output or settings.use_structured_citations
        
        # Conversational memory store (session_id -> ChatMessageHistory)
        self.history_store: Dict[str, InMemoryChatMessageHistory] = {}
        
        # Create prompt template
        self.prompt = self._create_prompt(prompt_template)
        
        # Build chain
        self.retrieval_chain = self._build_chain()
        
        if self.use_structured_output:
            logger.info("✓ Structured citations enabled (QuotedAnswer schema)")
        
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
    
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """
        Get or create chat history for a session.
        
        Used by RunnableWithMessageHistory for conversational memory.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Chat message history for the session
        """
        if session_id not in self.history_store:
            logger.info(f"Creating new chat history for session: {session_id}")
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]
    
    def create_conversational_chain(self):
        """
        Create conversational RAG chain with message history.
        
        Wraps the base retrieval chain with RunnableWithMessageHistory
        to enable multi-turn conversations with context preservation.
        
        Returns:
            RunnableWithMessageHistory wrapping the retrieval chain
            
        Example:
            ```python
            rag = RAGChain()
            conv_chain = rag.create_conversational_chain()
            
            # Use with session config
            config = {"configurable": {"session_id": "user_123"}}
            response1 = conv_chain.invoke({"input": "What is GTM?"}, config)
            response2 = conv_chain.invoke({"input": "Can you elaborate?"}, config)
            # Context from first question is remembered!
            ```
        """
        logger.info("Creating conversational chain with message history")
        
        conversational_chain = RunnableWithMessageHistory(
            self.retrieval_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        logger.info("✓ Conversational chain created")
        return conversational_chain
    
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
