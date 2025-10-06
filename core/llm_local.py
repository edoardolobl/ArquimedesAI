"""
LLM integration using Ollama for local inference.

Provides interface to Ollama models running locally for generation.
Uses ChatOllama for structured output support.
"""

import logging

from langchain_ollama import ChatOllama

from settings import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages local LLM via Ollama.
    
    Provides a configured Ollama instance for use in RAG chains.
    
    Attributes:
        model_name: Ollama model identifier (e.g., gemma3:latest)
        base_url: Ollama API endpoint
        temperature: Generation temperature
        llm: Ollama LLM instance
    """
    
    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ):
        """
        Initialize Ollama LLM.
        
        Args:
            model_name: Ollama model name (uses settings default if None)
            base_url: Ollama API URL (uses settings default if None)
            temperature: Generation temperature (uses settings default if None)
        """
        self.model_name = model_name or settings.ollama_model
        self.base_url = base_url or settings.ollama_base
        self.temperature = temperature or settings.ollama_temperature
        
        logger.info(f"Initializing Ollama LLM: {self.model_name}")
        logger.info(f"API endpoint: {self.base_url}")
        logger.info(f"Temperature: {self.temperature}")
        
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
        )
        
        logger.info("✓ LLM initialized (ChatOllama with structured output support)")
    
    def get_llm(self) -> ChatOllama:
        """
        Get Ollama LLM instance.
        
        Returns:
            Configured ChatOllama instance for use in chains (supports structured output)
        """
        return self.llm
