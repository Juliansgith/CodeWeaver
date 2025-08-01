"""
Centralized LLM and AI model configuration.
Makes it easy to change models, providers, and pricing across the entire application.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
import os

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class ModelCapability(Enum):
    EMBEDDING = "embedding"
    COMPLETION = "completion"
    RERANKING = "reranking"
    CHAT = "chat"

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: LLMProvider
    capabilities: List[ModelCapability]
    max_tokens: int
    context_window: int
    input_price_per_1k: float  # USD per 1K tokens
    output_price_per_1k: float  # USD per 1K tokens
    embedding_dimensions: Optional[int] = None
    rate_limit_per_minute: int = 60
    preferred_for: Optional[List[str]] = None  # Use cases this model is preferred for

@dataclass 
class ProviderConfig:
    """Configuration for an AI provider."""
    name: str
    provider: LLMProvider
    api_key_env_vars: List[str]  # Environment variables to check for API key
    base_url: Optional[str] = None
    default_models: Dict[ModelCapability, str] = None

class LLMConfigManager:
    """Manages LLM model configurations and provides easy access."""
    
    def __init__(self):
        self.models = self._initialize_models()
        self.providers = self._initialize_providers()
    
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize all available models with their configurations."""
        return {
            # OpenAI Models
            "gpt-4.1-nano": ModelConfig(
                name="gpt-4.1-nano",
                provider=LLMProvider.OPENAI,
                capabilities=[ModelCapability.COMPLETION, ModelCapability.CHAT, ModelCapability.RERANKING],
                max_tokens=16384,
                context_window=128000,
                input_price_per_1k=0.00015,  # $0.15 per 1M tokens
                output_price_per_1k=0.0006,  # $0.60 per 1M tokens
                preferred_for=["reranking", "quick_analysis", "cost_efficient"]
            ),
            
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                provider=LLMProvider.OPENAI,
                capabilities=[ModelCapability.COMPLETION, ModelCapability.CHAT, ModelCapability.RERANKING],
                max_tokens=16384,
                context_window=128000,
                input_price_per_1k=0.00015,
                output_price_per_1k=0.0006,
                preferred_for=["general", "cost_efficient"]
            ),
            
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                provider=LLMProvider.OPENAI,
                capabilities=[ModelCapability.COMPLETION, ModelCapability.CHAT],
                max_tokens=4096,
                context_window=128000,
                input_price_per_1k=0.0025,
                output_price_per_1k=0.01,
                preferred_for=["complex_analysis", "high_quality"]
            ),
            
            "text-embedding-3-small": ModelConfig(
                name="text-embedding-3-small",
                provider=LLMProvider.OPENAI,
                capabilities=[ModelCapability.EMBEDDING],
                max_tokens=8191,
                context_window=8191,
                input_price_per_1k=0.00002,  # $0.02 per 1M tokens
                output_price_per_1k=0.0,
                embedding_dimensions=1536,
                preferred_for=["embedding", "cost_efficient"]
            ),
            
            "text-embedding-3-large": ModelConfig(
                name="text-embedding-3-large",
                provider=LLMProvider.OPENAI,
                capabilities=[ModelCapability.EMBEDDING],
                max_tokens=8191,
                context_window=8191,
                input_price_per_1k=0.00013,  # $0.13 per 1M tokens
                output_price_per_1k=0.0,
                embedding_dimensions=3072,
                preferred_for=["embedding", "high_quality"]
            ),
            
            "text-embedding-ada-002": ModelConfig(
                name="text-embedding-ada-002",
                provider=LLMProvider.OPENAI,
                capabilities=[ModelCapability.EMBEDDING],
                max_tokens=8191,
                context_window=8191,
                input_price_per_1k=0.0001,  # $0.10 per 1M tokens
                output_price_per_1k=0.0,
                embedding_dimensions=1536,
                preferred_for=["embedding", "legacy"]
            ),
            
            # Anthropic Models
            "claude-3.5-sonnet": ModelConfig(
                name="claude-3.5-sonnet",
                provider=LLMProvider.ANTHROPIC,
                capabilities=[ModelCapability.COMPLETION, ModelCapability.CHAT, ModelCapability.RERANKING],
                max_tokens=8192,
                context_window=200000,
                input_price_per_1k=0.003,
                output_price_per_1k=0.015,
                preferred_for=["complex_analysis", "code_understanding", "high_quality"]
            ),
            
            "claude-3-haiku": ModelConfig(
                name="claude-3-haiku",
                provider=LLMProvider.ANTHROPIC,
                capabilities=[ModelCapability.COMPLETION, ModelCapability.CHAT, ModelCapability.RERANKING],
                max_tokens=4096,
                context_window=200000,
                input_price_per_1k=0.00025,
                output_price_per_1k=0.00125,
                preferred_for=["quick_analysis", "cost_efficient"]
            ),
            
            # Gemini Models
            "gemini-1.5-flash": ModelConfig(
                name="gemini-1.5-flash",
                provider=LLMProvider.GEMINI,
                capabilities=[ModelCapability.COMPLETION, ModelCapability.CHAT, ModelCapability.RERANKING],
                max_tokens=8192,
                context_window=1000000,
                input_price_per_1k=0.000075,
                output_price_per_1k=0.0003,
                preferred_for=["cost_efficient", "large_context"]
            ),
            
            "gemini-1.5-pro": ModelConfig(
                name="gemini-1.5-pro", 
                provider=LLMProvider.GEMINI,
                capabilities=[ModelCapability.COMPLETION, ModelCapability.CHAT],
                max_tokens=8192,
                context_window=2000000,
                input_price_per_1k=0.00125,
                output_price_per_1k=0.005,
                preferred_for=["complex_analysis", "large_context", "high_quality"]
            ),
            
            "text-embedding-004": ModelConfig(
                name="text-embedding-004",
                provider=LLMProvider.GEMINI,
                capabilities=[ModelCapability.EMBEDDING],
                max_tokens=2048,
                context_window=2048,
                input_price_per_1k=0.00001,  # $0.01 per 1M tokens
                output_price_per_1k=0.0,
                embedding_dimensions=768,
                preferred_for=["embedding", "cost_efficient"]
            ),
        }
    
    def _initialize_providers(self) -> Dict[LLMProvider, ProviderConfig]:
        """Initialize provider configurations."""
        return {
            LLMProvider.OPENAI: ProviderConfig(
                name="OpenAI",
                provider=LLMProvider.OPENAI,
                api_key_env_vars=["OPENAI_API_KEY"],
                base_url="https://api.openai.com/v1",
                default_models={
                    ModelCapability.EMBEDDING: "text-embedding-3-small",
                    ModelCapability.COMPLETION: "gpt-4.1-nano",
                    ModelCapability.RERANKING: "gpt-4.1-nano",
                    ModelCapability.CHAT: "gpt-4.1-nano"
                }
            ),
            
            LLMProvider.ANTHROPIC: ProviderConfig(
                name="Anthropic",
                provider=LLMProvider.ANTHROPIC,
                api_key_env_vars=["ANTHROPIC_API_KEY"],
                base_url="https://api.anthropic.com",
                default_models={
                    ModelCapability.COMPLETION: "claude-3-haiku",
                    ModelCapability.RERANKING: "claude-3-haiku",
                    ModelCapability.CHAT: "claude-3.5-sonnet"
                }
            ),
            
            LLMProvider.GEMINI: ProviderConfig(
                name="Google Gemini",
                provider=LLMProvider.GEMINI,
                api_key_env_vars=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                default_models={
                    ModelCapability.EMBEDDING: "text-embedding-004",
                    ModelCapability.COMPLETION: "gemini-1.5-flash",
                    ModelCapability.RERANKING: "gemini-1.5-flash",
                    ModelCapability.CHAT: "gemini-1.5-pro"
                }
            )
        }
    
    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self.models.get(model_name)
    
    def get_provider(self, provider: LLMProvider) -> Optional[ProviderConfig]:
        """Get provider configuration."""
        return self.providers.get(provider)
    
    def get_models_by_capability(self, capability: ModelCapability) -> List[ModelConfig]:
        """Get all models that support a specific capability."""
        return [model for model in self.models.values() 
                if capability in model.capabilities]
    
    def get_models_by_provider(self, provider: LLMProvider) -> List[ModelConfig]:
        """Get all models from a specific provider."""
        return [model for model in self.models.values() 
                if model.provider == provider]
    
    def get_preferred_model(self, capability: ModelCapability, 
                          provider: Optional[LLMProvider] = None,
                          use_case: Optional[str] = None) -> Optional[ModelConfig]:
        """Get the preferred model for a capability and use case."""
        
        # Filter by capability
        candidates = self.get_models_by_capability(capability)
        
        # Filter by provider if specified
        if provider:
            candidates = [m for m in candidates if m.provider == provider]
        
        if not candidates:
            return None
        
        # If use case specified, prefer models optimized for it
        if use_case:
            preferred = [m for m in candidates 
                        if m.preferred_for and use_case in m.preferred_for]
            if preferred:
                candidates = preferred
        
        # Return the first candidate (could add more sophisticated selection logic)
        return candidates[0]
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of providers that have API keys configured."""
        available = []
        
        for provider, config in self.providers.items():
            for env_var in config.api_key_env_vars:
                if os.getenv(env_var):
                    available.append(provider)
                    break
        
        return available
    
    def get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Get API key for a provider."""
        config = self.get_provider(provider)
        if not config:
            return None
        
        for env_var in config.api_key_env_vars:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key
        
        return None
    
    def get_default_model(self, capability: ModelCapability, 
                         provider: Optional[LLMProvider] = None) -> Optional[str]:
        """Get default model name for a capability."""
        
        if provider:
            config = self.get_provider(provider)
            if config and config.default_models:
                return config.default_models.get(capability)
        
        # Fall back to first available provider
        for prov in self.get_available_providers():
            config = self.get_provider(prov)
            if config and config.default_models:
                default_name = config.default_models.get(capability)
                if default_name:
                    return default_name
        
        return None
    
    def update_model_pricing(self, model_name: str, 
                           input_price: Optional[float] = None,
                           output_price: Optional[float] = None):
        """Update pricing for a model."""
        if model_name in self.models:
            if input_price is not None:
                self.models[model_name].input_price_per_1k = input_price
            if output_price is not None:
                self.models[model_name].output_price_per_1k = output_price
    
    def get_cost_estimate(self, model_name: str, input_tokens: int, 
                         output_tokens: int = 0) -> float:
        """Calculate cost estimate for using a model."""
        model = self.get_model(model_name)
        if not model:
            return 0.0
        
        input_cost = (input_tokens / 1000) * model.input_price_per_1k
        output_cost = (output_tokens / 1000) * model.output_price_per_1k
        
        return input_cost + output_cost
    
    def list_models(self, capability: Optional[ModelCapability] = None,
                   provider: Optional[LLMProvider] = None) -> List[str]:
        """List available model names with optional filtering."""
        models = list(self.models.keys())
        
        if capability:
            models = [name for name in models 
                     if capability in self.models[name].capabilities]
        
        if provider:
            models = [name for name in models 
                     if self.models[name].provider == provider]
        
        return sorted(models)


# Global configuration manager instance
_global_llm_config: Optional[LLMConfigManager] = None

def get_llm_config() -> LLMConfigManager:
    """Get or create global LLM configuration manager."""
    global _global_llm_config
    if _global_llm_config is None:
        _global_llm_config = LLMConfigManager()
    return _global_llm_config


# Convenience functions for common operations
def get_reranking_model(provider: Optional[LLMProvider] = None) -> Optional[str]:
    """Get the preferred model for re-ranking tasks."""
    config = get_llm_config()
    model = config.get_preferred_model(
        ModelCapability.RERANKING, 
        provider, 
        "cost_efficient"
    )
    return model.name if model else None

def get_embedding_model(provider: Optional[LLMProvider] = None) -> Optional[str]:
    """Get the preferred model for embedding tasks."""
    config = get_llm_config()
    model = config.get_preferred_model(
        ModelCapability.EMBEDDING,
        provider,
        "cost_efficient"
    )
    return model.name if model else None

def get_chat_model(provider: Optional[LLMProvider] = None,
                  use_case: str = "general") -> Optional[str]:
    """Get the preferred model for chat/completion tasks."""
    config = get_llm_config()
    model = config.get_preferred_model(
        ModelCapability.CHAT,
        provider,
        use_case
    )
    return model.name if model else None


# Current configuration summary
def print_current_config():
    """Print current model configuration for easy reference."""
    config = get_llm_config()
    available_providers = config.get_available_providers()
    
    print("=== CodeWeaver LLM Configuration ===")
    print(f"Available Providers: {[p.value for p in available_providers]}")
    print()
    
    for capability in ModelCapability:
        print(f"{capability.value.upper()}:")
        for provider in available_providers:
            model = get_llm_config().get_preferred_model(capability, provider)
            if model:
                cost_est = config.get_cost_estimate(model.name, 1000, 100)
                print(f"  {provider.value}: {model.name} (${cost_est:.4f}/1K+100 tokens)")
        print()
    
    print("To change models, edit the preferred_for lists in llm_config.py")
    print("Or modify the default_models in ProviderConfig")


if __name__ == "__main__":
    print_current_config()