"""
Configuration system for embedding providers.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Import LLM config for integration
try:
    from .llm_config import get_llm_config, ModelCapability, LLMProvider as LLMProviderEnum
    HAS_LLM_CONFIG = True
except ImportError:
    HAS_LLM_CONFIG = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding services."""
    provider: EmbeddingProvider
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 8192
    batch_size: int = 100
    rate_limit_per_minute: int = 1000
    embedding_dimensions: Optional[int] = None
    additional_params: Optional[Dict[str, Any]] = None


class EmbeddingConfigManager:
    """Manages embedding configuration with multiple sources."""
    
    DEFAULT_CONFIGS = {
        EmbeddingProvider.GEMINI: EmbeddingConfig(
            provider=EmbeddingProvider.GEMINI,
            model_name="models/embedding-001",
            max_tokens=2048,
            batch_size=100,
            rate_limit_per_minute=1500,
            embedding_dimensions=768
        ),
        EmbeddingProvider.OPENAI: EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-small",
            max_tokens=8192,
            batch_size=2048,
            rate_limit_per_minute=3000,
            embedding_dimensions=1536
        )
    }
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / '.codeweaver'
        self.config_file = self.config_dir / 'embedding_config.json'
        self.config_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[EmbeddingProvider, EmbeddingConfig]:
        """Load configuration from file and environment variables."""
        config = {}
        
        # Start with defaults
        for provider, default_config in self.DEFAULT_CONFIGS.items():
            config[provider] = EmbeddingConfig(**asdict(default_config))
        
        # Load from config file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                for provider_name, provider_config in file_config.items():
                    try:
                        provider = EmbeddingProvider(provider_name)
                        if provider in config:
                            # Update existing config
                            for key, value in provider_config.items():
                                if hasattr(config[provider], key):
                                    setattr(config[provider], key, value)
                    except ValueError:
                        continue  # Skip unknown providers
                        
            except Exception as e:
                print(f"Warning: Failed to load embedding config from {self.config_file}: {e}")
        
        # Override with environment variables
        self._load_from_environment(config)
        
        return config
    
    def _load_from_environment(self, config: Dict[EmbeddingProvider, EmbeddingConfig]):
        """Load configuration from environment variables."""
        # Environment variable patterns:
        # CODEWEAVER_GEMINI_API_KEY
        # CODEWEAVER_OPENAI_API_KEY
        # CODEWEAVER_GEMINI_MODEL
        # CODEWEAVER_OPENAI_MODEL
        
        env_mappings = {
            'CODEWEAVER_GEMINI_API_KEY': (EmbeddingProvider.GEMINI, 'api_key'),
            'CODEWEAVER_GEMINI_MODEL': (EmbeddingProvider.GEMINI, 'model_name'),
            'CODEWEAVER_OPENAI_API_KEY': (EmbeddingProvider.OPENAI, 'api_key'),
            'CODEWEAVER_OPENAI_MODEL': (EmbeddingProvider.OPENAI, 'model_name'),
            'GEMINI_API_KEY': (EmbeddingProvider.GEMINI, 'api_key'),  # Common alternative
            'OPENAI_API_KEY': (EmbeddingProvider.OPENAI, 'api_key'),  # Common alternative
        }
        
        for env_var, (provider, attr) in env_mappings.items():
            value = os.getenv(env_var)
            if value and provider in config:
                setattr(config[provider], attr, value)
    
    def get_config(self, provider: EmbeddingProvider) -> EmbeddingConfig:
        """Get configuration for a specific provider."""
        return self._config.get(provider, self.DEFAULT_CONFIGS[provider])
    
    def set_api_key(self, provider: EmbeddingProvider, api_key: str):
        """Set API key for a provider."""
        if provider not in self._config:
            self._config[provider] = EmbeddingConfig(**asdict(self.DEFAULT_CONFIGS[provider]))
        
        self._config[provider].api_key = api_key
        self.save_config()
    
    def set_model(self, provider: EmbeddingProvider, model_name: str):
        """Set model name for a provider."""
        if provider not in self._config:
            self._config[provider] = EmbeddingConfig(**asdict(self.DEFAULT_CONFIGS[provider]))
        
        self._config[provider].model_name = model_name
        self.save_config()
    
    def update_config(self, provider: EmbeddingProvider, **kwargs):
        """Update multiple config values for a provider."""
        if provider not in self._config:
            self._config[provider] = EmbeddingConfig(**asdict(self.DEFAULT_CONFIGS[provider]))
        
        config = self._config[provider]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.save_config()
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            config_dict = {}
            for provider, config in self._config.items():
                # Don't save API keys to file for security
                config_data = asdict(config)
                config_data['api_key'] = None  # Remove API key from saved config
                config_dict[provider.value] = config_data
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save embedding config: {e}")
    
    def get_available_providers(self) -> Dict[EmbeddingProvider, bool]:
        """Get available providers and their API key status."""
        available = {}
        for provider, config in self._config.items():
            available[provider] = bool(config.api_key)
        return available
    
    def validate_config(self, provider: EmbeddingProvider) -> Tuple[bool, Optional[str]]:
        """Validate configuration for a provider."""
        if provider not in self._config:
            return False, f"No configuration found for {provider.value}"
        
        config = self._config[provider]
        
        if not config.api_key:
            return False, f"API key not configured for {provider.value}"
        
        if not config.model_name:
            return False, f"Model name not configured for {provider.value}"
        
        return True, None
    
    def get_setup_instructions(self, provider: EmbeddingProvider) -> str:
        """Get setup instructions for a provider."""
        if provider == EmbeddingProvider.GEMINI:
            return """
To set up Gemini embeddings:

1. Get API key from Google AI Studio: https://makersuite.google.com/app/apikey
2. Set it via environment variable:
   export CODEWEAVER_GEMINI_API_KEY="your-api-key-here"
   
3. Or configure via command line:
   codeweaver config set-embedding-key gemini "your-api-key-here"

4. Or set directly in Python:
   from codeweaver.config.embedding_config import EmbeddingConfigManager
   config = EmbeddingConfigManager()
   config.set_api_key(EmbeddingProvider.GEMINI, "your-api-key-here")
"""
        elif provider == EmbeddingProvider.OPENAI:
            return """
To set up OpenAI embeddings:

1. Get API key from OpenAI: https://platform.openai.com/api-keys
2. Set it via environment variable:
   export CODEWEAVER_OPENAI_API_KEY="your-api-key-here"
   
3. Or configure via command line:
   codeweaver config set-embedding-key openai "your-api-key-here"

4. Or set directly in Python:
   from codeweaver.config.embedding_config import EmbeddingConfigManager
   config = EmbeddingConfigManager()
   config.set_api_key(EmbeddingProvider.OPENAI, "your-api-key-here")
"""
        else:
            return f"No setup instructions available for {provider.value}"


# Global config manager instance
_config_manager = None

def get_embedding_config() -> EmbeddingConfigManager:
    """Get the global embedding configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = EmbeddingConfigManager()
    return _config_manager