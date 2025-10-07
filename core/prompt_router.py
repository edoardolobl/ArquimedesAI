"""
Semantic query router for ArquimedesAI (v1.4).

Uses semantic-router library with hybrid routing (dense + sparse encoders)
to classify user queries into domain-specific routes:
- general_chat: Generic questions (fallback)
- gtm_qa: GTM taxonomy questions
- gtm_generation: GTM tag/trigger/variable creation requests
- gtm_validation: GTM configuration validation requests

Two-stage routing:
1. Keyword pre-filtering: Check for GTM context indicators
2. Semantic classification: Use BGE-M3 embeddings + BM25 for route selection
"""

import logging
from typing import Literal, Optional
from dataclasses import dataclass

from semantic_router import Route
from semantic_router.routers import HybridRouter
from semantic_router.encoders import HuggingFaceEncoder, BM25Encoder
from semantic_router.schema import Message

from prompts.gtm_prompts import (
    GTM_QA_UTTERANCES,
    GTM_GENERATION_UTTERANCES,
    GTM_VALIDATION_UTTERANCES,
)

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTE DEFINITIONS
# ============================================================================

RouteType = Literal["general_chat", "gtm_qa", "gtm_generation", "gtm_validation"]


@dataclass
class QueryRoute:
    """
    Result of routing a query.
    
    Attributes:
        route: Route type selected
        confidence: Confidence score (0-1)
        is_gtm_context: Whether GTM keywords were detected
    """
    route: RouteType
    confidence: float
    is_gtm_context: bool


# GTM context keywords for pre-filtering (Stage 1)
GTM_CONTEXT_KEYWORDS = {
    # Core entities
    "tag", "tags",
    "trigger", "triggers",
    "variável", "variaveis", "variáveis",
    "container",
    "workspace", "workspaces",
    
    # GTM-specific
    "gtm", "google tag manager",
    "datalayer", "data layer",
    "pageview", "evento", "eventos",
    
    # Actions
    "validar", "revisar", "auditar", "verificar", "checar",
    "criar", "gerar", "montar", "construir",
    
    # Common pages/events from dictionaries
    "cart", "carrinho",
    "checkout",
    "pdp", "produto",
    "homepage",
    
    # Platforms
    "ga4", "google analytics",
    "meta", "facebook", "pixel",
    "tiktok",
}


# ============================================================================
# ROUTE CREATION
# ============================================================================

def create_routes() -> list[Route]:
    """
    Create semantic routes with PT-BR utterances.
    
    Returns:
        List of Route objects for HybridRouter
        
    Routes:
        - gtm_qa: Questions about GTM concepts, rules, dictionaries
        - gtm_generation: Requests to create tags, triggers, variables
        - gtm_validation: Requests to validate/review configurations
        - general_chat: Fallback for non-GTM queries
    """
    routes = [
        # GTM Q&A Route
        Route(
            name="gtm_qa",
            utterances=GTM_QA_UTTERANCES,
        ),
        
        # GTM Generation Route
        Route(
            name="gtm_generation",
            utterances=GTM_GENERATION_UTTERANCES,
        ),
        
        # GTM Validation Route
        Route(
            name="gtm_validation",
            utterances=GTM_VALIDATION_UTTERANCES,
        ),
        
        # General Chat Route (fallback)
        Route(
            name="general_chat",
            utterances=[
                # Generic questions (fallback examples)
                "Como funciona o ArquimedesAI?",
                "O que você pode fazer?",
                "Quem criou este sistema?",
                "Como usar este assistente?",
                "Explique como funciona RAG",
                "O que é retrieval augmented generation?",
            ],
        ),
    ]
    
    return routes


# ============================================================================
# HYBRID ROUTER SETUP
# ============================================================================

class QueryRouter:
    """
    Semantic router for classifying user queries into domain routes.
    
    Uses HybridRouter with:
    - Dense encoder: HuggingFaceEncoder with BGE-M3
    - Sparse encoder: BM25Encoder for keyword matching
    - Two-stage routing: keyword pre-filter → semantic classification
    
    Attributes:
        router: HybridRouter instance
        alpha: Balance between sparse (0) and dense (1) encoders
        
    Example:
        >>> router = QueryRouter()
        >>> result = router.route("Validar tag: Cart - Load - GA4")
        >>> print(result.route)  # "gtm_validation"
        >>> print(result.confidence)  # 0.85
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        alpha: float = 0.3,
        threshold: float = 0.5,
    ):
        """
        Initialize QueryRouter with hybrid encoders.
        
        Args:
            model_name: HuggingFace model for dense encoder (default: BGE-M3)
            alpha: Balance sparse (0) vs dense (1) encoder. Default 0.3 (30% sparse, 70% dense)
            threshold: Minimum confidence for non-fallback routes (default: 0.5)
            
        Note:
            - BGE-M3 is reused from core.embedder (same model used for retrieval)
            - Alpha=0.3 balances keyword precision (BM25) with semantic context (BGE-M3)
            - Lower alpha = more keyword-driven, higher alpha = more semantic
        """
        self.alpha = alpha
        self.threshold = threshold
        
        logger.info(f"Initializing QueryRouter with model={model_name}, alpha={alpha}")
        
        # Dense encoder (semantic similarity via BGE-M3)
        dense_encoder = HuggingFaceEncoder(name=model_name)
        
        # Sparse encoder (keyword matching via BM25)
        sparse_encoder = BM25Encoder()
        
        # Create routes
        routes = create_routes()
        
        # Initialize HybridRouter with auto_sync to build index automatically
        self.router = HybridRouter(
            encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            routes=routes,
            alpha=alpha,
            auto_sync="local",  # Auto-sync to build index
        )
        
        logger.info(f"Router initialized with {len(routes)} routes")
    
    def _detect_gtm_context(self, query: str) -> bool:
        """
        Stage 1: Detect if query contains GTM-related keywords.
        
        Args:
            query: User query (PT-BR)
            
        Returns:
            True if GTM keywords detected, False otherwise
            
        Purpose:
            Fast pre-filtering to avoid semantic routing for obviously non-GTM queries.
            Improves performance and reduces false positives.
        """
        query_lower = query.lower()
        
        # Check if any GTM keyword is present
        for keyword in GTM_CONTEXT_KEYWORDS:
            if keyword in query_lower:
                logger.debug(f"GTM context detected: keyword='{keyword}'")
                return True
        
        return False
    
    def route(self, query: str) -> QueryRoute:
        """
        Route a user query to the appropriate domain handler.
        
        Two-stage routing:
        1. Keyword pre-filtering: Check for GTM context
        2. Semantic classification: Use HybridRouter with BGE-M3 + BM25
        
        Args:
            query: User query in Brazilian Portuguese
            
        Returns:
            QueryRoute with selected route, confidence, and GTM context flag
            
        Example:
            >>> router = QueryRouter()
            >>> result = router.route("O que é uma tag?")
            >>> print(result.route)  # "gtm_qa"
            >>> print(result.is_gtm_context)  # True
            
            >>> result = router.route("Como funciona RAG?")
            >>> print(result.route)  # "general_chat"
            >>> print(result.is_gtm_context)  # False
        """
        # Stage 1: Keyword pre-filtering
        is_gtm = self._detect_gtm_context(query)
        
        if not is_gtm:
            # Fast path: No GTM keywords, route to general_chat
            logger.info("No GTM context detected, routing to general_chat")
            return QueryRoute(
                route="general_chat",
                confidence=1.0,
                is_gtm_context=False,
            )
        
        # Stage 2: Semantic classification with HybridRouter
        logger.debug(f"GTM context detected, running semantic classification for: {query[:50]}...")
        
        try:
            # Route using HybridRouter
            result = self.router(query)
            
            if result is None or result.name is None:
                # Fallback to general_chat if no route selected
                logger.warning("Router returned None, falling back to general_chat")
                return QueryRoute(
                    route="general_chat",
                    confidence=0.0,
                    is_gtm_context=is_gtm,
                )
            
            # Extract route name and confidence
            route_name = result.name
            
            # Get similarity score (if available)
            # Note: semantic-router may provide score in different formats
            confidence = getattr(result, 'score', None) or getattr(result, 'similarity', 0.5)
            
            # Validate route name
            if route_name not in ["general_chat", "gtm_qa", "gtm_generation", "gtm_validation"]:
                logger.warning(f"Unknown route '{route_name}', falling back to general_chat")
                return QueryRoute(
                    route="general_chat",
                    confidence=confidence,
                    is_gtm_context=is_gtm,
                )
            
            # Check confidence threshold for non-general routes
            if route_name != "general_chat" and confidence < self.threshold:
                logger.info(
                    f"Confidence {confidence:.2f} below threshold {self.threshold}, "
                    f"falling back to general_chat"
                )
                return QueryRoute(
                    route="general_chat",
                    confidence=confidence,
                    is_gtm_context=is_gtm,
                )
            
            logger.info(f"Routed to '{route_name}' with confidence {confidence:.2f}")
            
            return QueryRoute(
                route=route_name,  # type: ignore
                confidence=confidence,
                is_gtm_context=is_gtm,
            )
        
        except Exception as e:
            logger.error(f"Error during routing: {e}", exc_info=True)
            # Fallback to general_chat on error
            return QueryRoute(
                route="general_chat",
                confidence=0.0,
                is_gtm_context=is_gtm,
            )
    
    def route_batch(self, queries: list[str]) -> list[QueryRoute]:
        """
        Route multiple queries in batch.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of QueryRoute results (same order as input)
            
        Note:
            Currently processes sequentially. Could be optimized with
            batch encoding in the future.
        """
        return [self.route(query) for query in queries]
    
    def get_available_routes(self) -> list[str]:
        """
        Get list of available route names.
        
        Returns:
            List of route names
        """
        return [route.name for route in self.router.routes]
    
    def get_route_info(self, route_name: str) -> dict:
        """
        Get information about a specific route.
        
        Args:
            route_name: Name of the route
            
        Returns:
            Dict with route info (name, utterance count)
            
        Raises:
            ValueError: If route not found
        """
        for route in self.router.routes:
            if route.name == route_name:
                return {
                    "name": route.name,
                    "utterance_count": len(route.utterances),
                    "sample_utterances": route.utterances[:3],  # First 3 examples
                }
        
        raise ValueError(f"Route '{route_name}' not found")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_router() -> QueryRouter:
    """
    Create QueryRouter with default settings.
    
    Returns:
        QueryRouter instance with BGE-M3 and alpha=0.3
    """
    return QueryRouter(
        model_name="BAAI/bge-m3",
        alpha=0.3,
        threshold=0.5,
    )


# ============================================================================
# TESTING UTILITIES (for development/debugging)
# ============================================================================

if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    router = create_default_router()
    
    # Test queries
    test_queries = [
        "O que é uma tag?",
        "Criar tag para carrinho ao carregar",
        "Validar: Cart - Load - GA4",
        "Como funciona o ArquimedesAI?",
        "Gerar trigger de clique no checkout",
        "Quais plataformas estão no dicionário?",
    ]
    
    print("\n" + "=" * 80)
    print("TESTING QUERY ROUTER")
    print("=" * 80 + "\n")
    
    for query in test_queries:
        result = router.route(query)
        print(f"Query: {query}")
        print(f"  → Route: {result.route}")
        print(f"  → Confidence: {result.confidence:.2f}")
        print(f"  → GTM Context: {result.is_gtm_context}")
        print()
    
    # Show route info
    print("\n" + "=" * 80)
    print("AVAILABLE ROUTES")
    print("=" * 80 + "\n")
    
    for route_name in router.get_available_routes():
        info = router.get_route_info(route_name)
        print(f"{info['name']}:")
        print(f"  Utterances: {info['utterance_count']}")
        print(f"  Samples: {info['sample_utterances']}")
        print()
