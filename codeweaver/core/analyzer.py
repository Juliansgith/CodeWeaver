from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .tokenizer import TokenEstimator, LLMProvider


@dataclass
class FileTokenInfo:
    path: Path
    relative_path: str
    tokens: int
    size_bytes: int
    percentage: float


@dataclass
class DirectoryTokenInfo:
    path: str
    tokens: int
    file_count: int
    percentage: float
    subdirectories: Dict[str, 'DirectoryTokenInfo']
    files: List[FileTokenInfo]


class TokenAnalyzer:
    """
    Analyzes token distribution across files and directories.
    """
    
    def __init__(self, provider: LLMProvider = LLMProvider.CLAUDE, model: str = "claude-3.5-sonnet"):
        self.provider = provider
        self.model = model
    
    def analyze_files(self, project_files: List[Path], root_path: Path) -> Tuple[List[FileTokenInfo], int]:
        """
        Analyze token consumption for individual files.
        Returns (file_info_list, total_tokens)
        """
        file_infos = []
        total_tokens = 0
        
        for file_path in project_files:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Estimate tokens for this file
                estimates = TokenEstimator.estimate_tokens(content, self.provider)
                file_tokens = estimates.get(self.model, 0)
                
                # Get file size
                file_size = file_path.stat().st_size
                
                # Create file info
                relative_path = str(file_path.relative_to(root_path)).replace("\\", "/")
                file_info = FileTokenInfo(
                    path=file_path,
                    relative_path=relative_path,
                    tokens=file_tokens,
                    size_bytes=file_size,
                    percentage=0.0  # Will be calculated later
                )
                
                file_infos.append(file_info)
                total_tokens += file_tokens
                
            except Exception as e:
                # Skip files that can't be read
                continue
        
        # Calculate percentages
        for file_info in file_infos:
            file_info.percentage = (file_info.tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        return file_infos, total_tokens
    
    def analyze_directories(self, file_infos: List[FileTokenInfo], total_tokens: int) -> DirectoryTokenInfo:
        """
        Analyze token consumption by directory structure.
        """
        # Build directory tree
        root = DirectoryTokenInfo(
            path="",
            tokens=0,
            file_count=0,
            percentage=100.0,
            subdirectories={},
            files=[]
        )
        
        for file_info in file_infos:
            path_parts = Path(file_info.relative_path).parts
            current_dir = root
            
            # Navigate/create directory structure
            for i, part in enumerate(path_parts[:-1]):  # Exclude filename
                if part not in current_dir.subdirectories:
                    current_dir.subdirectories[part] = DirectoryTokenInfo(
                        path="/".join(path_parts[:i+1]),
                        tokens=0,
                        file_count=0,
                        percentage=0.0,
                        subdirectories={},
                        files=[]
                    )
                current_dir = current_dir.subdirectories[part]
            
            # Add file to its directory
            current_dir.files.append(file_info)
        
        # Calculate directory totals (bottom-up)
        self._calculate_directory_totals(root, total_tokens)
        
        return root
    
    def _calculate_directory_totals(self, directory: DirectoryTokenInfo, total_tokens: int):
        """
        Recursively calculate token totals for directories.
        """
        # Calculate tokens from files in this directory
        directory.tokens = sum(f.tokens for f in directory.files)
        directory.file_count = len(directory.files)
        
        # Add tokens from subdirectories
        for subdir in directory.subdirectories.values():
            self._calculate_directory_totals(subdir, total_tokens)
            directory.tokens += subdir.tokens
            directory.file_count += subdir.file_count
        
        # Calculate percentage
        directory.percentage = (directory.tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    def get_top_files(self, file_infos: List[FileTokenInfo], limit: int = 20) -> List[FileTokenInfo]:
        """Get top files by token consumption."""
        return sorted(file_infos, key=lambda f: f.tokens, reverse=True)[:limit]
    
    def get_top_directories(self, root_dir: DirectoryTokenInfo, limit: int = 15) -> List[DirectoryTokenInfo]:
        """Get top directories by token consumption."""
        directories = []
        self._collect_directories(root_dir, directories)
        
        # Filter out root and sort by tokens
        directories = [d for d in directories if d.path]  # Exclude root (empty path)
        return sorted(directories, key=lambda d: d.tokens, reverse=True)[:limit]
    
    def _collect_directories(self, directory: DirectoryTokenInfo, collection: List[DirectoryTokenInfo]):
        """Recursively collect all directories."""
        collection.append(directory)
        for subdir in directory.subdirectories.values():
            self._collect_directories(subdir, collection)
    
    def get_directory_suggestions(self, root_dir: DirectoryTokenInfo, threshold_percentage: float = 5.0) -> List[str]:
        """
        Get suggestions for directories that might be worth ignoring.
        Returns directory paths that consume significant tokens but might be non-essential.
        """
        suggestions = []
        
        # Common non-essential directory patterns
        ignore_patterns = [
            'test', 'tests', '__tests__', 'spec', 'specs',
            'log', 'logs', 'temp', 'tmp', 'cache',
            'docs', 'documentation', 'examples', 'sample', 'samples',
            'node_modules', '__pycache__', '.git', 'venv', '.venv',
            'dist', 'build', 'target', 'out', 'output',
            'assets', 'static', 'public', 'resources'
        ]
        
        def check_directory(directory: DirectoryTokenInfo):
            if directory.percentage >= threshold_percentage:
                dir_name = Path(directory.path).name.lower() if directory.path else ""
                
                # Check if directory name matches common non-essential patterns
                for pattern in ignore_patterns:
                    if pattern in dir_name:
                        suggestions.append(f"{directory.path}/ ({directory.percentage:.1f}% - {directory.tokens:,} tokens)")
                        break
            
            # Check subdirectories
            for subdir in directory.subdirectories.values():
                check_directory(subdir)
        
        check_directory(root_dir)
        return suggestions