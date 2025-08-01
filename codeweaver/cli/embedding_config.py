"""
CLI commands for configuring embedding providers.
"""

import click
import sys
from pathlib import Path
from typing import Optional

from ..config.embedding_config import (
    get_embedding_config, EmbeddingProvider, EmbeddingConfig
)
from ..ai.embeddings import create_embedding_service

@click.group('embedding')
def embedding_cli():
    """Configure embedding providers for semantic search."""
    pass

@embedding_cli.command()
@click.argument('provider', type=click.Choice(['gemini', 'openai']))
@click.argument('api_key')
def set_key(provider: str, api_key: str):
    """Set API key for an embedding provider."""
    try:
        provider_enum = EmbeddingProvider(provider)
        config_manager = get_embedding_config()
        
        config_manager.set_api_key(provider_enum, api_key)
        
        # Test the API key
        click.echo(f"Testing {provider} API key...")
        service = create_embedding_service(provider_enum)
        
        # Try a simple embedding to verify the key works
        import asyncio
        
        async def test_embedding():
            # Use get_embeddings as it's a public method
            result = await service.get_embeddings(["test"])
            return result is not None and len(result) > 0
        
        if asyncio.run(test_embedding()):
            click.echo(f"✅ {provider.title()} API key configured and verified successfully!")
        else:
            click.echo(f"⚠️  {provider.title()} API key configured but verification failed. Please check the key is valid.")
            
    except Exception as e:
        click.echo(f"❌ Failed to configure {provider} API key: {e}", err=True)
        sys.exit(1)

@embedding_cli.command()
@click.argument('provider', type=click.Choice(['gemini', 'openai']))
@click.argument('model_name')
def set_model(provider: str, model_name: str):
    """Set the model name for an embedding provider."""
    try:
        provider_enum = EmbeddingProvider(provider)
        config_manager = get_embedding_config()
        
        config_manager.set_model(provider_enum, model_name)
        click.echo(f"✅ {provider.title()} model set to: {model_name}")
        
    except Exception as e:
        click.echo(f"❌ Failed to set {provider} model: {e}", err=True)
        sys.exit(1)

@embedding_cli.command()
def status():
    """Show status of embedding providers."""
    config_manager = get_embedding_config()
    available_providers = config_manager.get_available_providers()
    
    click.echo("🔧 Embedding Provider Status:\n")
    
    for provider, has_key in available_providers.items():
        config = config_manager.get_config(provider)
        status_icon = "✅" if has_key else "❌"
        
        click.echo(f"{status_icon} {provider.value.title()}:")
        click.echo(f"   API Key: {'Configured' if has_key else 'Not configured'}")
        click.echo(f"   Model: {config.model_name}")
        click.echo(f"   Max Tokens: {config.max_tokens}")
        click.echo(f"   Embedding Dimensions: {config.embedding_dimensions}")
        
        if not has_key:
            click.echo(f"   To configure: codeweaver embedding set-key {provider.value} YOUR_API_KEY")
        
        click.echo()
    
    # Show which provider would be auto-selected
    try:
        service = create_embedding_service()
        auto_provider = service.config.provider.value
        click.echo(f"🎯 Auto-selected provider: {auto_provider.title()}")
    except Exception as e:
        click.echo(f"⚠️  No providers available: {e}")

@embedding_cli.command()
@click.argument('provider', type=click.Choice(['gemini', 'openai']))
def setup(provider: str):
    """Show setup instructions for an embedding provider."""
    provider_enum = EmbeddingProvider(provider)
    config_manager = get_embedding_config()
    
    instructions = config_manager.get_setup_instructions(provider_enum)
    click.echo(instructions)

@embedding_cli.command()
@click.argument('provider', type=click.Choice(['gemini', 'openai']))
def test(provider: str):
    """Test an embedding provider configuration."""
    try:
        provider_enum = EmbeddingProvider(provider)
        config_manager = get_embedding_config()
        
        # Validate configuration
        is_valid, error_message = config_manager.validate_config(provider_enum)
        if not is_valid:
            click.echo(f"❌ Configuration invalid: {error_message}", err=True)
            return
        
        # Create service and test
        service = create_embedding_service(provider_enum)
        
        click.echo(f"Testing {provider} embedding service...")
        click.echo(f"Model: {service.config.model_name}")
        
        import asyncio
        
        async def run_test():
            test_texts = [
                "This is a test of the embedding service.",
                "def hello_world(): print('Hello, World!')",
                "function calculateSum(a, b) { return a + b; }"
            ]
            
            click.echo(f"Generating embeddings for {len(test_texts)} test texts...")
            embeddings = await service.get_embeddings(test_texts)
            
            if embeddings:
                click.echo(f"✅ Success! Generated {len(embeddings)} embeddings")
                click.echo(f"   Embedding dimensions: {len(embeddings[0])}")
                
                # Test similarity calculation
                from ..ai.embeddings import calculate_cosine_similarity
                similarity = calculate_cosine_similarity(embeddings[0], embeddings[1])
                click.echo(f"   Sample similarity score: {similarity:.3f}")
                
                return True
            else:
                click.echo("❌ Failed to generate embeddings")
                return False
        
        success = asyncio.run(run_test())
        
        if success:
            click.echo(f"🎉 {provider.title()} embedding service is working correctly!")
        else:
            click.echo(f"❌ {provider.title()} embedding service test failed")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Test failed: {e}", err=True)
        sys.exit(1)

@embedding_cli.command()
def models():
    """Show available models for each provider."""
    click.echo("📚 Available Embedding Models:\n")
    
    click.echo("🔵 Gemini (Google AI):")
    click.echo("   • models/embedding-001 (768 dims, general purpose)")
    click.echo("   • models/text-embedding-004 (768 dims, latest)")
    click.echo("   Get API key: https://makersuite.google.com/app/apikey")
    click.echo()
    
    click.echo("🟢 OpenAI:")
    click.echo("   • text-embedding-3-small (1536 dims, cost-effective)")
    click.echo("   • text-embedding-3-large (3072 dims, best performance)")
    click.echo("   • text-embedding-ada-002 (1536 dims, legacy)")
    click.echo("   Get API key: https://platform.openai.com/api-keys")
    click.echo()
    
    click.echo("💡 Recommendations:")
    click.echo("   • For cost-effectiveness: Gemini embedding-001")
    click.echo("   • For best performance: OpenAI text-embedding-3-large")
    click.echo("   • For balanced usage: OpenAI text-embedding-3-small")

@embedding_cli.command()
@click.option('--provider', type=click.Choice(['gemini', 'openai']), help='Specific provider to reset')
def reset(provider: Optional[str]):
    """Reset embedding configuration to defaults."""
    config_manager = get_embedding_config()
    
    if provider:
        provider_enum = EmbeddingProvider(provider)
        config_manager.update_config(provider_enum, api_key=None)
        click.echo(f"✅ Reset {provider} configuration to defaults")
    else:
        # Reset all providers
        for provider_enum in EmbeddingProvider:
            config_manager.update_config(provider_enum, api_key=None)
        click.echo("✅ Reset all embedding configurations to defaults")
    
    config_manager.save_config()

if __name__ == '__main__':
    embedding_cli()