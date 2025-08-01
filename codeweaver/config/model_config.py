#!/usr/bin/env python3
"""
Model Configuration Utility

Easy way to view and modify LLM model configurations.
Run this script to see current settings and test model selection.

Usage:
    python -m codeweaver.config.model_config
    python -m codeweaver.config.model_config --test-reranking
    python -m codeweaver.config.model_config --estimate-cost "gpt-4.1-nano" 1000 500
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from codeweaver.config.llm_config import (
        get_llm_config, 
        LLMProvider, 
        ModelCapability,
        get_reranking_model,
        get_embedding_model, 
        get_chat_model,
        print_current_config
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="CodeWeaver Model Configuration")
    parser.add_argument("--test-reranking", action="store_true", 
                       help="Test re-ranking model selection")
    parser.add_argument("--test-embedding", action="store_true",
                       help="Test embedding model selection")
    parser.add_argument("--estimate-cost", nargs=3, metavar=("MODEL", "INPUT_TOKENS", "OUTPUT_TOKENS"),
                       help="Estimate cost for using a model")
    parser.add_argument("--list-models", action="store_true",
                       help="List all available models")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"],
                       help="Filter by provider")
    
    args = parser.parse_args()
    
    config = get_llm_config()
    
    if args.list_models:
        capability = None
        provider = None
        if args.provider:
            provider = LLMProvider(args.provider)
        
        models = config.list_models(capability, provider)
        print(f"Available models{' for ' + args.provider if args.provider else ''}:")
        for model_name in models:
            model = config.get_model(model_name)
            if model:
                caps = [c.value for c in model.capabilities]
                cost = config.get_cost_estimate(model_name, 1000, 100)
                print(f"  {model_name} ({model.provider.value})")
                print(f"    Capabilities: {', '.join(caps)}")
                print(f"    Cost estimate: ${cost:.4f} per 1K input + 100 output tokens")
                print()
        return
    
    if args.estimate_cost:
        model_name, input_tokens, output_tokens = args.estimate_cost
        try:
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
            cost = config.get_cost_estimate(model_name, input_tokens, output_tokens)
            print(f"Cost estimate for {model_name}:")
            print(f"  Input tokens: {input_tokens:,}")
            print(f"  Output tokens: {output_tokens:,}")
            print(f"  Total cost: ${cost:.6f}")
            
            # Show per-token breakdown
            model = config.get_model(model_name)
            if model:
                input_cost = (input_tokens / 1000) * model.input_price_per_1k
                output_cost = (output_tokens / 1000) * model.output_price_per_1k
                print(f"  Input cost: ${input_cost:.6f}")
                print(f"  Output cost: ${output_cost:.6f}")
        except ValueError:
            print("Error: Token counts must be integers")
        return
    
    if args.test_reranking:
        print("Testing re-ranking model selection:")
        provider_filter = LLMProvider(args.provider) if args.provider else None
        
        if provider_filter:
            model = get_reranking_model(provider_filter)
            print(f"  {provider_filter.value}: {model or 'None available'}")
        else:
            for provider in LLMProvider:
                model = get_reranking_model(provider)
                if model:
                    cost = config.get_cost_estimate(model, 1000, 100)
                    print(f"  {provider.value}: {model} (${cost:.4f}/1K+100 tokens)")
        return
    
    if args.test_embedding:
        print("Testing embedding model selection:")
        provider_filter = LLMProvider(args.provider) if args.provider else None
        
        if provider_filter:
            model = get_embedding_model(provider_filter)
            print(f"  {provider_filter.value}: {model or 'None available'}")
        else:
            for provider in LLMProvider:
                model = get_embedding_model(provider)
                if model:
                    cost = config.get_cost_estimate(model, 1000, 0)
                    print(f"  {provider.value}: {model} (${cost:.4f}/1K tokens)")
        return
    
    # Default: show full configuration
    print_current_config()
    
    # Show quick examples
    print("\n=== Quick Examples ===")
    print("View specific models:")
    print("  python -m codeweaver.config.model_config --test-reranking")
    print("  python -m codeweaver.config.model_config --test-embedding")
    print()
    print("Cost estimation:")
    print("  python -m codeweaver.config.model_config --estimate-cost gpt-4.1-nano 2000 500")
    print()
    print("List models:")
    print("  python -m codeweaver.config.model_config --list-models")
    print("  python -m codeweaver.config.model_config --list-models --provider openai")


if __name__ == "__main__":
    main()