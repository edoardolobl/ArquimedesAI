"""
RAG chain orchestration using LangChain LCEL.

Combines retrieval and generation into a unified chain for question answering.
Supports conversational memory (v1.4), structured citations (v1.4), 
and semantic routing (v1.4).
"""

import logging
from typing import Dict, Any, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.prompt_router import QueryRouter, RouteType

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
    CONCISE_PROMPT,
    CRITIC_PROMPT,
    EXPLAIN_PROMPT,
    QuotedAnswer,
    CitedAnswer,
    format_docs_with_id,
)
from prompts.base_prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    apply_style_modifier,
    format_context_documents,
)
from prompts.gtm_prompts import (
    get_gtm_prompt,
    GTM_QA_SYSTEM_PROMPT,
    GTM_GENERATION_SYSTEM_PROMPT,
    GTM_VALIDATION_SYSTEM_PROMPT,
)
from settings import settings

logger = logging.getLogger(__name__)

# Optional import for semantic routing (v1.4)
try:
    from core.prompt_router import QueryRouter, RouteType
    ROUTING_AVAILABLE = True
except ImportError:
    ROUTING_AVAILABLE = False
    logger.warning("Semantic routing not available (semantic-router not installed)")
    QueryRouter = None  # type: ignore
    RouteType = None  # type: ignore


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
    - Semantic routing (v1.4)
    
    Attributes:
        vector_store: Qdrant vector store instance
        llm_manager: LLM manager instance
        retrieval_chain: Configured LCEL retrieval chain
        history_store: Session-based chat history storage (v1.4)
        router: Optional semantic query router (v1.4)
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        llm_manager: LLMManager | None = None,
        prompt_template: str | None = None,  # DEPRECATED in v2.0, kept for backward compat
        use_structured_output: bool = False,
        enable_routing: bool = True,  # v2.0: Default to True (was False in v1.4)
        router: Any = None,  # QueryRouter instance or None
    ):
        """
        Initialize RAG chain with pre-built domain chains (v2.0).
        
        Args:
            vector_store: Vector store instance (creates new if None)
            llm_manager: LLM manager instance (creates new if None)
            prompt_template: DEPRECATED - Custom prompt template (backward compat only)
            use_structured_output: Enable structured citations with Pydantic (v1.4)
            enable_routing: Enable semantic routing (default: True in v2.0)
            router: Optional QueryRouter instance (creates default if enable_routing=True and None)
            
        Changes in v2.0:
            - enable_routing now defaults to True (intelligent routing by default)
            - Pre-builds all domain chains (3 GTM + 4 general styles)
            - No chain rebuilding (fixes conversational + routing incompatibility)
            - prompt_template parameter deprecated (use pre-built chains instead)
        """
        logger.info("Initializing RAG chain (v2.0)")
        
        # Initialize components
        self.vector_store = vector_store or QdrantVectorStore()
        self.llm_manager = llm_manager or LLMManager()
        self.use_structured_output = use_structured_output or settings.use_structured_citations
        
        # Conversational memory store (session_id -> ChatMessageHistory)
        self.history_store: Dict[str, InMemoryChatMessageHistory] = {}
        
        # Semantic routing (v2.0)
        self.enable_routing = enable_routing and ROUTING_AVAILABLE
        self.router = None
        
        if self.enable_routing:
            if router:
                self.router = router
                logger.info("✓ Using provided QueryRouter")
            elif ROUTING_AVAILABLE and QueryRouter is not None:
                from core.prompt_router import create_default_router
                self.router = create_default_router()
                logger.info("✓ Created default QueryRouter")
            else:
                logger.warning("Routing requested but semantic-router not available, disabling routing")
                self.enable_routing = False
        
        # v2.0: Pre-build all domain chains (no rebuilding during queries)
        if self.enable_routing:
            self._build_domain_chains()
            logger.info("✓ Pre-built 7 domain chains (3 GTM + 4 general styles)")
        else:
            # Backward compatibility: build single default chain
            self.prompt = self._create_prompt(prompt_template)
            self.retrieval_chain = self._build_chain()
            logger.info("✓ Built default chain (routing disabled)")
        
        if self.use_structured_output:
            logger.info("✓ Structured citations enabled (QuotedAnswer schema)")
        
        if self.enable_routing:
            logger.info("✓ Semantic routing enabled")
        
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
    
    def _build_chain_with_prompt(self, prompt_text: str):
        """
        Build LCEL retrieval chain with specific prompt (v2.0 helper).
        
        Helper method for pre-building domain-specific chains without
        modifying self.prompt or self.retrieval_chain.
        
        Args:
            prompt_text: Prompt template string
            
        Returns:
            Configured retrieval chain with the specified prompt
        """
        # Get retriever (hybrid or dense-only based on settings)
        base_retriever = create_hybrid_retriever(self.vector_store)
        
        # Optionally wrap with reranking
        retriever = create_reranking_retriever(base_retriever)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(prompt_text)
        
        # Create document chain (combines docs with LLM)
        document_chain = create_stuff_documents_chain(
            llm=self.llm_manager.get_llm(),
            prompt=prompt,
        )
        
        # Create retrieval chain (retriever + document chain)
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain,
        )
        
        return retrieval_chain
    
    def _build_domain_chains(self):
        """
        Pre-build all domain-specific chains (v2.0).
        
        Builds 7 chains total:
        - 3 GTM domain chains (gtm_qa, gtm_generation, gtm_validation)
        - 4 General style chains (grounded, concise, critic, explain)
        
        These chains are built once during initialization and selected
        dynamically via routing, eliminating chain rebuilding that
        broke conversational memory in v1.4.
        """
        logger.info("Pre-building domain chains...")
        
        # GTM domain chains (use specialized prompts, no style variants)
        self.gtm_qa_chain = self._build_chain_with_prompt(GTM_QA_SYSTEM_PROMPT)
        self.gtm_generation_chain = self._build_chain_with_prompt(GTM_GENERATION_SYSTEM_PROMPT)
        self.gtm_validation_chain = self._build_chain_with_prompt(GTM_VALIDATION_SYSTEM_PROMPT)
        
        logger.debug("✓ Built 3 GTM domain chains")
        
        # General domain chains (with style variants)
        self.general_chains = {
            "grounded": self._build_chain_with_prompt(GROUNDED_PROMPT),
            "concise": self._build_chain_with_prompt(CONCISE_PROMPT),
            "critic": self._build_chain_with_prompt(CRITIC_PROMPT),
            "explain": self._build_chain_with_prompt(EXPLAIN_PROMPT),
        }
        
        logger.debug("✓ Built 4 general style chains")
    
    def _create_routing_chain(self, style: str = "grounded"):
        """
        Create dynamic routing chain using RunnableLambda (v2.1.1 - with metadata).
        
        Returns a Runnable that routes queries to appropriate pre-built chains
        based on semantic routing, invokes them, and adds routing metadata to results.
        
        Args:
            style: Style for general queries (grounded/concise/critic/explain)
            
        Returns:
            RunnableLambda that routes, invokes, and adds metadata
            
        Changes in v2.1.1:
            - Now INVOKES the selected chain (not just returns it)
            - Adds route + confidence metadata to result
            - Fixes conversational mode routing display
        """
        from langchain_core.runnables import RunnableLambda
        
        def route_and_invoke(input_dict):
            """Route query, invoke chain, and add metadata."""
            query = input_dict["input"]
            
            # Route using semantic-router
            route_result = self.router.route(query)
            
            logger.info(
                f"Routed to '{route_result.route}' "
                f"(confidence: {route_result.confidence:.2f})"
            )
            
            # Select appropriate pre-built chain
            if route_result.route == "gtm_qa":
                chain = self.gtm_qa_chain
            elif route_result.route == "gtm_generation":
                chain = self.gtm_generation_chain
            elif route_result.route == "gtm_validation":
                chain = self.gtm_validation_chain
            else:
                # General chat - use style variant
                chain = self.general_chains.get(style, self.general_chains["grounded"])
            
            # Invoke selected chain
            result = chain.invoke(input_dict)
            
            # Add routing metadata to result
            result["route"] = route_result.route
            result["confidence"] = route_result.confidence
            
            return result
        
        return RunnableLambda(route_and_invoke)
    
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
    
    def create_conversational_chain(self, style: str = "grounded"):
        """
        Create conversational RAG chain with message history (v2.0 with routing support).
        
        Wraps the base retrieval chain (or routing chain) with RunnableWithMessageHistory
        to enable multi-turn conversations with context preservation.
        
        Args:
            style: Style for general queries when routing enabled (grounded/concise/critic/explain)
        
        Returns:
            RunnableWithMessageHistory wrapping the chain
            
        Changes in v2.0:
            - Now supports routing + conversational together!
            - If routing enabled: wraps routing chain
            - If routing disabled: wraps default chain (backward compat)
            
        Example:
            ```python
            rag = RAGChain(enable_routing=True)
            conv_chain = rag.create_conversational_chain(style="grounded")
            
            # Use with session config
            config = {"configurable": {"session_id": "user_123"}}
            
            # First query: GTM (routes to gtm_qa_chain)
            response1 = conv_chain.invoke({"input": "What is GTM?"}, config)
            
            # Second query: Follow-up (remembers context, routes appropriately)
            response2 = conv_chain.invoke({"input": "Can you elaborate?"}, config)
            # Context from first question is preserved across routes!
            ```
        """
        logger.info(f"Creating conversational chain (routing: {self.enable_routing}, style: {style})")
        
        # Select base chain
        if self.enable_routing:
            # Use routing chain (dynamically selects pre-built chains)
            base_chain = self._create_routing_chain(style=style)
            logger.debug("Using routing chain for conversational wrapper")
        else:
            # Use default chain (backward compatibility)
            base_chain = self.retrieval_chain
            logger.debug("Using default chain for conversational wrapper")
        
        # Wrap with message history
        conversational_chain = RunnableWithMessageHistory(
            base_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        logger.info("✓ Conversational chain created (routing + memory compatible!)")
        return conversational_chain
    
    async def ainvoke(self, query: str, style: str = "grounded") -> Dict[str, Any]:
        """
        Async invoke RAG chain (v2.0 with routing support).
        
        Args:
            query: User question
            style: Style for general queries when routing enabled (default: grounded)
            
        Returns:
            Dictionary with answer, context, route, and confidence metadata
        """
        logger.info(f"Query: {query}")
        
        input_data = {"input": query}
        
        if self.enable_routing:
            # Explicitly route to capture metadata
            route_result = self.router.route(query)
            
            logger.info(
                f"Routed to '{route_result.route}' "
                f"(confidence: {route_result.confidence:.2f})"
            )
            
            # Select appropriate chain based on route
            if route_result.route == "gtm_qa":
                chain = self.gtm_qa_chain
            elif route_result.route == "gtm_generation":
                chain = self.gtm_generation_chain
            elif route_result.route == "gtm_validation":
                chain = self.gtm_validation_chain
            else:
                # General chat - use style variant
                chain = self.general_chains.get(style, self.general_chains["grounded"])
            
            # Invoke selected chain
            result = await chain.ainvoke(input_data)
            
            # Add routing metadata to result
            result["route"] = route_result.route
            result["confidence"] = route_result.confidence
        else:
            # Use default chain (backward compat)
            result = await self.retrieval_chain.ainvoke(input_data)
            result["route"] = "general_chat"
            result["confidence"] = 1.0
        
        logger.info("✓ Query processed")
        return result
    
    def invoke(self, query: str, style: str = "grounded") -> Dict[str, Any]:
        """
        Synchronous invoke RAG chain (v2.0 with routing support).
        
        Args:
            query: User question
            style: Style for general queries when routing enabled (default: grounded)
            
        Returns:
            Dictionary with answer, context, route, and confidence metadata
        """
        logger.info(f"Query: {query}")
        
        input_data = {"input": query}
        
        if self.enable_routing:
            # Explicitly route to capture metadata
            route_result = self.router.route(query)
            
            logger.info(
                f"Routed to '{route_result.route}' "
                f"(confidence: {route_result.confidence:.2f})"
            )
            
            # Select appropriate chain based on route
            if route_result.route == "gtm_qa":
                chain = self.gtm_qa_chain
            elif route_result.route == "gtm_generation":
                chain = self.gtm_generation_chain
            elif route_result.route == "gtm_validation":
                chain = self.gtm_validation_chain
            else:
                # General chat - use style variant
                chain = self.general_chains.get(style, self.general_chains["grounded"])
            
            # Invoke selected chain
            result = chain.invoke(input_data)
            
            # Add routing metadata to result
            result["route"] = route_result.route
            result["confidence"] = route_result.confidence
        else:
            # Use default chain (backward compat)
            result = self.retrieval_chain.invoke(input_data)
            result["route"] = "general_chat"
            result["confidence"] = 1.0
        
        logger.info("✓ Query processed")
        return result
    
    def query(
        self,
        query: str,
        style: Literal["grounded", "concise", "critic", "explain"] | None = None,
        auto_route: bool = True,
    ) -> Dict[str, Any]:
        """
        DEPRECATED in v2.0: Use invoke() instead.
        
        This method is deprecated because it rebuilt chains on every call,
        which broke conversational memory. In v2.0, routing is always-on
        and chains are pre-built during initialization.
        
        Args:
            query: User question
            style: Style modifier (default: grounded)
            auto_route: Ignored in v2.0 (routing is always on if enabled)
            
        Returns:
            Dictionary with answer, context, and metadata
            
        Migration:
            # Old (v1.4)
            result = rag.query("What is GTM?", style="grounded")
            
            # New (v2.0)
            result = rag.invoke("What is GTM?", style="grounded")
        """
        import warnings
        warnings.warn(
            "query() is deprecated in v2.0 and will be removed in v2.1. "
            "Use invoke() instead, which now supports routing without rebuilding chains.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        # Redirect to invoke()
        return self.invoke(query, style=style or "grounded")
    
    def _get_prompt_for_route(
        self,
        route: str,
        style: Literal["grounded", "concise", "critic", "explain"],
    ) -> str:
        """
        Get appropriate prompt template for route and style.
        
        Args:
            route: Route name (general_chat, gtm_qa, gtm_generation, gtm_validation)
            style: Style modifier to apply
            
        Returns:
            Complete prompt template string
        """
        # GTM-specific routes
        if route == "gtm_qa":
            from prompts.gtm_prompts import GTM_QA_SYSTEM_PROMPT, GTM_QA_EXAMPLES
            base = GTM_QA_SYSTEM_PROMPT
            # Add user prompt structure
            prompt = f"{base}\n\n## CONTEXTO RECUPERADO DOS DOCUMENTOS\n{{context}}\n\n## PERGUNTA DO USUÁRIO\n{{input}}\n\n## SUA RESPOSTA\n"
            
        elif route == "gtm_generation":
            from prompts.gtm_prompts import GTM_GENERATION_SYSTEM_PROMPT
            base = GTM_GENERATION_SYSTEM_PROMPT
            prompt = f"{base}\n\n## CONTEXTO RECUPERADO DOS DOCUMENTOS\n{{context}}\n\n## REQUISIÇÃO DO USUÁRIO\n{{input}}\n\n## SUA RESPOSTA\n"
            
        elif route == "gtm_validation":
            from prompts.gtm_prompts import GTM_VALIDATION_SYSTEM_PROMPT
            base = GTM_VALIDATION_SYSTEM_PROMPT
            # Validation uses CoT, so more structured
            prompt = f"{base}\n\nPense passo a passo:\n1. Analise o nome fornecido\n2. Compare com regras do guia\n3. Identifique problemas\n4. Sugira correções\n\n## CONTEXTO RECUPERADO DOS DOCUMENTOS\n{{context}}\n\n## NOMENCLATURA A VALIDAR\n{{input}}\n\n## SUA ANÁLISE\n"
            
        else:
            # General chat route
            base = GENERAL_CHAT_SYSTEM_PROMPT
            prompt = f"{base}\n\n## CONTEXTO RECUPERADO DOS DOCUMENTOS\n{{context}}\n\n## PERGUNTA DO USUÁRIO\n{{input}}\n\n## SUA RESPOSTA\n"
        
        # Apply style modifier
        prompt = apply_style_modifier(prompt, style)
        
        return prompt


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
