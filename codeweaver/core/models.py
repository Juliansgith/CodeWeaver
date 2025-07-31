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
    mode: str = 'digest'  # 'digest' or 'preview'