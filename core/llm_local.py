"""
LLM integration using Ollama for local inference.

Provides interface to Ollama models running locally for generation.
"""

import logging

from langchain_ollama import OllamaLLM

from settings import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manages local LLM via Ollama.
    
    Provides a configured Ollama instance for use in RAG chains.
    
    Attributes:
        model_name: Ollama model identifier (e.g., gemma2:1b)
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
        
        self.llm = OllamaLLM(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
        )
        
        logger.info("✓ LLM initialized")
    
    def get_llm(self) -> OllamaLLM:
        """
        Get Ollama LLM instance.
        
        Returns:
            Configured Ollama instance for use in chains
        """
        return self.llm
