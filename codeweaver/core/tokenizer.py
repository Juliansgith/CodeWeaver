import re
from typing import Dict, Tuple
from enum import Enum


class LLMProvider(Enum):
    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    LLAMA = "llama"


class TokenEstimator:
    """
    Realistic token estimation for different LLM providers.
    Based on observed tokenization patterns from various models.
    """
    
    # Context limits for different models (in tokens)
    CONTEXT_LIMITS = {
        LLMProvider.CLAUDE: {
            "claude-3-haiku": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-opus": 200000,
            "claude-3.5-sonnet": 200000,
        },
        LLMProvider.GPT: {
            "gpt-4": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 16385,
        },
        LLMProvider.GEMINI: {
            "gemini-pro": 32768,
            "gemini-1.5-pro": 2000000,  # 2M tokens
            "gemini-2.5-flash": 1000000,  # 1M tokens
        },
        LLMProvider.LLAMA: {
            "llama-2-70b": 4096,
            "llama-3-70b": 8192,
            "codellama-34b": 16384,
        }
    }

    @staticmethod
    def estimate_tokens(text: str, provider: LLMProvider = LLMProvider.CLAUDE) -> Dict[str, int]:
        """
        Estimate tokens for different LLM providers with realistic calculations.
        Returns estimates for multiple models within the provider.
        """
        if provider == LLMProvider.CLAUDE:
            return TokenEstimator._estimate_claude_tokens(text)
        elif provider == LLMProvider.GPT:
            return TokenEstimator._estimate_gpt_tokens(text)
        elif provider == LLMProvider.GEMINI:
            return TokenEstimator._estimate_gemini_tokens(text)
        elif provider == LLMProvider.LLAMA:
            return TokenEstimator._estimate_llama_tokens(text)
        else:
            # Fallback to simple estimation
            return {"unknown": len(text) // 4}

    @staticmethod
    def _estimate_claude_tokens(text: str) -> Dict[str, int]:
        """
        Claude tokenization estimation based on observed patterns.
        Claude tends to be more efficient with code and structured text.
        """
        # Base estimation: ~3.5-4 chars per token for mixed content
        base_estimate = len(text) / 3.7
        
        # Adjustments for different content types
        code_ratio = TokenEstimator._get_code_ratio(text)
        whitespace_ratio = len(text.strip()) / len(text) if text else 1
        
        # Code is typically more token-efficient in Claude
        if code_ratio > 0.5:
            base_estimate *= 0.9  # 10% fewer tokens for code-heavy content
        
        # Heavy whitespace/formatting affects tokenization
        if whitespace_ratio < 0.8:
            base_estimate *= 1.1  # 10% more tokens for whitespace-heavy content
        
        estimated_tokens = int(base_estimate)
        
        return {
            "claude-3-haiku": estimated_tokens,
            "claude-3-sonnet": estimated_tokens,
            "claude-3-opus": estimated_tokens,
            "claude-3.5-sonnet": estimated_tokens,
        }

    @staticmethod
    def _estimate_gpt_tokens(text: str) -> Dict[str, int]:
        """
        GPT tokenization estimation (BPE-based, more aggressive splitting).
        GPT models tend to use more tokens for the same content.
        """
        # Base estimation: ~3.2-3.8 chars per token
        base_estimate = len(text) / 3.4
        
        # GPT tends to split more aggressively
        code_ratio = TokenEstimator._get_code_ratio(text)
        if code_ratio > 0.5:
            base_estimate *= 1.05  # 5% more tokens for code
        
        # Special characters and punctuation increase token count
        special_chars = len(re.findall(r'[^\w\s]', text))
        if special_chars > len(text) * 0.1:
            base_estimate *= 1.1
        
        estimated_tokens = int(base_estimate)
        
        return {
            "gpt-4": estimated_tokens,
            "gpt-4-turbo": estimated_tokens,
            "gpt-4o": estimated_tokens,
            "gpt-3.5-turbo": estimated_tokens,
        }

    @staticmethod
    def _estimate_gemini_tokens(text: str) -> Dict[str, int]:
        """
        Gemini tokenization estimation (SentencePiece-based).
        """
        # Base estimation: ~3.8-4.2 chars per token
        base_estimate = len(text) / 3.9
        
        # Gemini handles Unicode and international text well
        non_ascii_ratio = len([c for c in text if ord(c) > 127]) / len(text) if text else 0
        if non_ascii_ratio > 0.1:
            base_estimate *= 0.95  # Slightly more efficient with Unicode
        
        estimated_tokens = int(base_estimate)
        
        # Gemini 2.5 Flash is slightly more efficient
        flash_tokens = int(base_estimate * 0.96)
        
        return {
            "gemini-pro": estimated_tokens,
            "gemini-1.5-pro": estimated_tokens,
            "gemini-2.5-flash": flash_tokens,
        }

    @staticmethod
    def _estimate_llama_tokens(text: str) -> Dict[str, int]:
        """
        LLaMA tokenization estimation (SentencePiece-based).
        """
        # Base estimation: ~3.5-4 chars per token
        base_estimate = len(text) / 3.6
        
        # Code-specific adjustments for CodeLlama
        code_ratio = TokenEstimator._get_code_ratio(text)
        
        estimated_tokens = int(base_estimate)
        code_llama_tokens = int(base_estimate * 0.92) if code_ratio > 0.3 else estimated_tokens
        
        return {
            "llama-2-70b": estimated_tokens,
            "llama-3-70b": estimated_tokens,
            "codellama-34b": code_llama_tokens,
        }

    @staticmethod
    def _get_code_ratio(text: str) -> float:
        """
        Estimate the ratio of code-like content in the text.
        """
        if not text:
            return 0.0
        
        # Count code indicators
        code_indicators = 0
        code_indicators += len(re.findall(r'\b(def|class|import|from|function|const|let|var)\b', text))
        code_indicators += len(re.findall(r'[{}();]', text))
        code_indicators += len(re.findall(r'^\s*(#|//|/\*)', text, re.MULTILINE))
        code_indicators += len(re.findall(r'[=<>!]=|[+\-*/%]=', text))
        
        # Rough heuristic: normalize by text length
        return min(1.0, code_indicators / (len(text.split()) + 1) * 10)

    @staticmethod
    def get_context_usage(token_count: int, provider: LLMProvider, model: str = None) -> Tuple[float, str]:
        """
        Get context window usage percentage and status.
        Returns (usage_percentage, status_message)
        """
        limits = TokenEstimator.CONTEXT_LIMITS.get(provider, {})
        
        if model and model in limits:
            limit = limits[model]
        else:
            # Use the most common/default model for the provider
            if provider == LLMProvider.CLAUDE:
                limit = limits.get("claude-3.5-sonnet", 200000)
            elif provider == LLMProvider.GPT:
                limit = limits.get("gpt-4", 128000)
            elif provider == LLMProvider.GEMINI:
                limit = limits.get("gemini-2.5-flash", 1000000)
            elif provider == LLMProvider.LLAMA:
                limit = limits.get("llama-3-70b", 8192)
            else:
                return 0.0, "Unknown model"
        
        usage_percent = (token_count / limit) * 100
        
        if usage_percent < 50:
            status = "OK"
        elif usage_percent < 75:
            status = "Moderate"
        elif usage_percent < 90:
            status = "High"
        elif usage_percent < 100:
            status = "Near limit"
        else:
            status = "Exceeds limit"
        
        return usage_percent, status

    @staticmethod
    def get_all_estimates(text: str) -> Dict[str, Dict[str, int]]:
        """Get token estimates for all supported LLM providers."""
        return {
            "claude": TokenEstimator.estimate_tokens(text, LLMProvider.CLAUDE),
            "gpt": TokenEstimator.estimate_tokens(text, LLMProvider.GPT),
            "gemini": TokenEstimator.estimate_tokens(text, LLMProvider.GEMINI),
            "llama": TokenEstimator.estimate_tokens(text, LLMProvider.LLAMA),
        }