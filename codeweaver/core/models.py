from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ProcessingStats:
    file_count: int
    file_size_kb: float
    estimated_tokens: int
    token_estimates: Optional[Dict[str, Dict[str, int]]] = None  # LLM-specific token counts
    token_analysis: Optional[Any] = None  # Token analysis data (file and directory breakdown)


@dataclass
class ProcessingResult:
    success: bool
    output_path: Optional[str] = None
    stats: Optional[ProcessingStats] = None
    message: Optional[str] = None
    files: Optional[List[Path]] = None


@dataclass
class ProcessingOptions:
    input_dir: str
    ignore_patterns: List[str]
    size_limit_mb: float
    mode: str
    strip_comments: bool = False
    optimize_whitespace: bool = False
    intelligent_sampling: bool = False
    sampling_strategy: str = "balanced"  # head_tail, key_sections, semantic, change_based, balanced
    sampling_max_lines: int = 300        # Maximum lines per file after sampling
    sampling_purpose: Optional[str] = None  # User's purpose for semantic sampling