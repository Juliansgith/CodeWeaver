# AI-powered features for CodeWeaver

from .embeddings import (
    GeminiEmbeddingService,
    SemanticCodeSearch,
    CodeChunk,
    SemanticMatch,
    EmbeddingCache
)

from .optimization_engine import (
    OptimizationEngine,
    PurposeAnalyzer,
    TaskType,
    PurposeAnalysis,
    OptimizationResult,
    FileRelevanceScore
)

__all__ = [
    'GeminiEmbeddingService',
    'SemanticCodeSearch', 
    'CodeChunk',
    'SemanticMatch',
    'EmbeddingCache',
    'OptimizationEngine',
    'PurposeAnalyzer',
    'TaskType',
    'PurposeAnalysis', 
    'OptimizationResult',
    'FileRelevanceScore'
]