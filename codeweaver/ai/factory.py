"""
Factory for creating and managing AI service instances.
Ensures lazy loading and centralized configuration of AI components.
"""
from typing import Optional
from functools import lru_cache

from .embeddings import BaseEmbeddingService, create_embedding_service, EmbeddingProvider
from ..config.embedding_config import get_embedding_config

class AIServiceFactory:
    """
    A singleton factory for accessing AI services.
    This ensures that services like the embedding model are initialized only once
    and shared across the application.
    """
    _embedding_service: Optional[BaseEmbeddingService] = None

    @classmethod
    def get_embedding_service(cls) -> Optional[BaseEmbeddingService]:
        """
        Get the singleton instance of the embedding service.
        It lazily initializes the service on first request, choosing the best
        available provider.
        """
        if cls._embedding_service is None:
            try:
                # create_embedding_service will intelligently pick the best
                # configured provider (Gemini > OpenAI > fallback).
                cls._embedding_service = create_embedding_service()
                print(f"Initialized Embedding Service with provider: {cls._embedding_service.get_model_info()['provider']}")
            except Exception as e:
                print(f"Could not initialize any embedding service: {e}")
                # Return None, but don't set the class variable so it can retry.
                return None
        return cls._embedding_service

# A global instance for easy access
ai_factory = AIServiceFactory()