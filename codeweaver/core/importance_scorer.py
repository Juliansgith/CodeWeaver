import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .dependency_analyzer import DependencyGraphAnalyzer, DependencyNode


class FileType(Enum):
    """Categorizes files by their role in the project."""
    ENTRY_POINT = "entry_point"        # main.py, index.js, App.js, etc.
    CORE_LIBRARY = "core_library"      # Heavily imported utilities
    BUSINESS_LOGIC = "business_logic"   # Domain-specific code
    CONFIGURATION = "configuration"    # Config files, settings
    TEST = "test"                      # Test files
    DOCUMENTATION = "documentation"     # Docs, README files
    BUILD_SCRIPT = "build_script"      # Build, deployment scripts
    GENERATED = "generated"            # Auto-generated code
    ASSETS = "assets"                  # Images, stylesheets, etc.
    DEPRECATED = "deprecated"          # Old/unused files
    UNKNOWN = "unknown"                # Unclassified files


@dataclass
class ImportanceFactors:
    """Factors that contribute to a file's importance score."""
    base_type_score: float = 0.0
    centrality_score: float = 0.0
    size_factor: float = 0.0
    recency_factor: float = 0.0
    naming_convention_bonus: float = 0.0
    content_complexity_factor: float = 0.0
    dependency_depth_penalty: float = 0.0
    
    def total_score(self) -> float:
        """Calculate the total importance score."""
        return (
            self.base_type_score +
            self.centrality_score +
            self.size_factor +
            self.recency_factor +
            self.naming_convention_bonus +
            self.content_complexity_factor -
            self.dependency_depth_penalty
        )


@dataclass
class FileImportanceInfo:
    """Complete importance information for a file."""
    path: Path
    relative_path: str
    file_type: FileType
    importance_score: float
    factors: ImportanceFactors = field(default_factory=ImportanceFactors)
    tokens: int = 0
    size_bytes: int = 0
    efficiency_ratio: float = 0.0  # importance per token


class FileImportanceScorer:
    """
    Sophisticated file importance scoring system that considers multiple factors:
    - File type and role in project
    - Dependency centrality 
    - Code complexity and quality
    - Naming conventions
    - File size and recency
    - Token efficiency
    """
    
    # Base scores for different file types
    BASE_TYPE_SCORES = {
        FileType.ENTRY_POINT: 100.0,
        FileType.CORE_LIBRARY: 80.0,
        FileType.BUSINESS_LOGIC: 70.0,
        FileType.CONFIGURATION: 50.0,
        FileType.BUILD_SCRIPT: 40.0,
        FileType.TEST: 30.0,
        FileType.DOCUMENTATION: 20.0,
        FileType.ASSETS: 10.0,
        FileType.GENERATED: -50.0,
        FileType.DEPRECATED: -30.0,
        FileType.UNKNOWN: 0.0
    }
    
    # Patterns to identify different file types
    FILE_TYPE_PATTERNS = {
        FileType.ENTRY_POINT: [
            r'^main\.(py|js|ts|java|cpp|c|go|rs)$',
            r'^index\.(js|ts|jsx|tsx)$',
            r'^app\.(py|js|ts|jsx|tsx)$',
            r'^run\.(py|js|sh)$',
            r'^server\.(py|js|ts)$',
            r'^__main__\.py$',
            r'^manage\.py$',
            r'^.*\.exe$',
            r'^.*App\.(js|ts|jsx|tsx)$',
        ],
        FileType.CORE_LIBRARY: [
            r'^utils?\.(py|js|ts)$',
            r'^helpers?\.(py|js|ts)$',
            r'^lib(rary)?\.(py|js|ts)$',
            r'^core\.(py|js|ts)$',
            r'^base\.(py|js|ts)$',
            r'^common\.(py|js|ts)$',
            r'^constants?\.(py|js|ts)$',
            r'^.*\.so$',
            r'^.*\.dll$',
            r'^.*\.a$',
        ],
        FileType.CONFIGURATION: [
            r'^config(uration)?\.(py|js|ts|json|yaml|yml|toml|ini)$',
            r'^settings?\.(py|js|ts|json|yaml|yml)$',
            r'^.*\.env$',
            r'^\.env.*$',
            r'^docker.*$',
            r'^.*\.json$',
            r'^.*\.yaml$',
            r'^.*\.yml$',
            r'^.*\.toml$',
            r'^.*\.ini$',
            r'^.*\.cfg$',
            r'^package\.json$',
            r'^requirements\.txt$',
            r'^setup\.py$',
            r'^pyproject\.toml$',
            r'^Cargo\.toml$',
            r'^pom\.xml$',
            r'^build\.gradle$',
        ],
        FileType.TEST: [
            r'^test.*\.(py|js|ts)$',
            r'^.*test\.(py|js|ts)$',
            r'^.*_test\.(py|js|ts|go)$',
            r'^.*Test\.(java|js|ts)$',
            r'^.*\.test\.(js|ts)$',
            r'^.*\.spec\.(js|ts)$',
            r'^spec.*\.(py|js|ts)$',
        ],
        FileType.DOCUMENTATION: [
            r'^readme\.(md|txt|rst)$',
            r'^.*\.md$',
            r'^.*\.rst$',
            r'^.*\.txt$',
            r'^docs?/.*$',
            r'^documentation/.*$',
            r'^.*\.doc[x]?$',
            r'^.*\.pdf$',
        ],
        FileType.BUILD_SCRIPT: [
            r'^.*\.(sh|bash|bat|cmd|ps1)$',
            r'^makefile$',
            r'^cmake.*$',
            r'^build\.(py|js|ts)$',
            r'^deploy\.(py|js|ts|sh)$',
            r'^install\.(py|js|ts|sh)$',
            r'^.*\.mk$',
        ],
        FileType.GENERATED: [
            r'^.*_pb2\.py$',
            r'^.*\.g\.(py|js|ts)$',
            r'^.*\.generated\.(py|js|ts)$',
            r'^.*\.auto\.(py|js|ts)$',
            r'^.*\.min\.(js|css)$',
            r'^.*\.bundle\.(js|css)$',
            r'^dist/.*$',
            r'^build/.*$',
            r'^target/.*$',
            r'^.*\.class$',
            r'^.*\.pyc$',
            r'^__pycache__/.*$',
        ],
        FileType.ASSETS: [
            r'^.*\.(png|jpg|jpeg|gif|svg|ico)$',
            r'^.*\.(css|scss|sass|less)$',
            r'^.*\.(woff|woff2|ttf|eot)$',
            r'^.*\.(mp4|mp3|wav|ogg)$',
            r'^assets?/.*$',
            r'^static/.*$',
            r'^public/.*$',
        ],
        FileType.DEPRECATED: [
            r'^.*\.old$',
            r'^.*\.bak$',
            r'^.*\.deprecated$',
            r'^.*\.legacy$',
            r'^old_.*$',
            r'^legacy_.*$',
            r'^deprecated_.*$',
        ]
    }
    
    # Important naming patterns that get bonus points
    IMPORTANT_NAME_PATTERNS = [
        r'^(auth|login|security|user)',  # Authentication/security
        r'^(api|router|controller|handler)',  # API/routing
        r'^(database|db|model|entity)',  # Data layer
        r'^(service|business|domain)',  # Business logic
        r'^(middleware|filter|interceptor)',  # Middleware
        r'^(error|exception|logging)',  # Error handling
        r'^(validation|validator)',  # Validation
    ]
    
    def __init__(self, dependency_analyzer: DependencyGraphAnalyzer):
        self.dependency_analyzer = dependency_analyzer
        
    def score_files(self, file_paths: List[Path], root_path: Path, 
                   token_info: Optional[Dict[str, int]] = None) -> List[FileImportanceInfo]:
        """
        Score all files based on their importance to the project.
        
        Args:
            file_paths: List of file paths to score
            root_path: Root path of the project
            token_info: Optional dict mapping file paths to token counts
            
        Returns:
            List of FileImportanceInfo objects sorted by importance
        """
        file_scores = []
        
        # Get dependency analysis
        dependency_graph = self.dependency_analyzer.analyze_dependencies(file_paths)
        
        for file_path in file_paths:
            try:
                relative_path = str(file_path.relative_to(root_path)).replace("\\", "/")
                
                # Get basic file info
                file_type = self._classify_file_type(file_path)
                size_bytes = file_path.stat().st_size
                tokens = token_info.get(relative_path, 0) if token_info else 0
                
                # Get dependency info if available
                dependency_node = dependency_graph.get(relative_path)
                
                # Calculate importance factors
                factors = self._calculate_importance_factors(
                    file_path, file_type, size_bytes, tokens, dependency_node
                )
                
                # Calculate efficiency ratio
                efficiency_ratio = factors.total_score() / max(tokens, 1) if tokens > 0 else 0
                
                file_info = FileImportanceInfo(
                    path=file_path,
                    relative_path=relative_path,
                    file_type=file_type,
                    importance_score=factors.total_score(),
                    factors=factors,
                    tokens=tokens,
                    size_bytes=size_bytes,
                    efficiency_ratio=efficiency_ratio
                )
                
                file_scores.append(file_info)
                
            except Exception:
                # Skip files that can't be processed
                continue
        
        # Sort by importance score (descending)
        return sorted(file_scores, key=lambda f: f.importance_score, reverse=True)
    
    def _classify_file_type(self, file_path: Path) -> FileType:
        """Classify a file based on its name and path patterns."""
        file_name = file_path.name.lower()
        relative_path = str(file_path).lower()
        
        # Check each file type pattern
        for file_type, patterns in self.FILE_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, file_name, re.IGNORECASE) or \
                   re.search(pattern, relative_path, re.IGNORECASE):
                    return file_type
        
        # Special logic for business logic detection
        if self._is_business_logic_file(file_path):
            return FileType.BUSINESS_LOGIC
            
        return FileType.UNKNOWN
    
    def _is_business_logic_file(self, file_path: Path) -> bool:
        """Determine if a file contains business logic based on various heuristics."""
        # Check directory structure
        path_parts = [part.lower() for part in file_path.parts]
        
        business_logic_indicators = [
            'models', 'controllers', 'services', 'handlers', 'views',
            'business', 'logic', 'domain', 'entities', 'repositories',
            'api', 'endpoints', 'routes', 'middleware'
        ]
        
        for indicator in business_logic_indicators:
            if indicator in path_parts:
                return True
        
        # Check file content patterns (if small enough to analyze quickly)
        if file_path.stat().st_size < 100000:  # Only analyze files < 100KB
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1000)  # Read first 1KB
                    
                # Look for class/function definitions that suggest business logic
                business_patterns = [
                    r'class\s+\w+(Service|Controller|Handler|Repository|Manager)',
                    r'def\s+\w*(process|handle|manage|execute|validate|calculate)',
                    r'function\s+\w*(process|handle|manage|execute|validate|calculate)',
                ]
                
                for pattern in business_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return True
                        
            except Exception:
                pass
        
        return False
    
    def _calculate_importance_factors(self, file_path: Path, file_type: FileType, 
                                    size_bytes: int, tokens: int, 
                                    dependency_node: Optional[DependencyNode]) -> ImportanceFactors:
        """Calculate all factors that contribute to file importance."""
        factors = ImportanceFactors()
        
        # Base type score
        factors.base_type_score = self.BASE_TYPE_SCORES.get(file_type, 0.0)
        
        # Centrality score from dependency analysis
        if dependency_node:
            # Normalize centrality score to 0-50 range
            factors.centrality_score = min(dependency_node.centrality_score * 10, 50.0)
        
        # Size factor (moderate size gets bonus, too small/large gets penalty)
        if size_bytes > 0:
            # Sweet spot is around 1KB-10KB for most files
            if 1000 <= size_bytes <= 10000:
                factors.size_factor = 10.0
            elif 500 <= size_bytes < 1000:
                factors.size_factor = 5.0
            elif size_bytes < 100:
                factors.size_factor = -5.0  # Too small, likely not important
            elif size_bytes > 100000:
                factors.size_factor = -10.0  # Too large, might be generated/data
        
        # Naming convention bonus
        factors.naming_convention_bonus = self._calculate_naming_bonus(file_path)
        
        # Content complexity factor (based on tokens if available)
        if tokens > 0:
            # Moderate complexity gets bonus
            if 100 <= tokens <= 1000:
                factors.content_complexity_factor = 15.0
            elif 50 <= tokens < 100:
                factors.content_complexity_factor = 10.0
            elif tokens < 20:
                factors.content_complexity_factor = -5.0
            elif tokens > 5000:
                factors.content_complexity_factor = -10.0
        
        # Dependency depth penalty
        if dependency_node:
            # Files that import too many things might be overly complex
            import_count = len(dependency_node.imports_from)
            if import_count > 20:
                factors.dependency_depth_penalty = (import_count - 20) * 2
            elif import_count > 50:
                factors.dependency_depth_penalty = (import_count - 20) * 5
        
        return factors
    
    def _calculate_naming_bonus(self, file_path: Path) -> float:
        """Calculate bonus points based on naming conventions that suggest importance."""
        bonus = 0.0
        file_name = file_path.name.lower()
        path_str = str(file_path).lower()
        
        # Check for important naming patterns
        for pattern in self.IMPORTANT_NAME_PATTERNS:
            if re.search(pattern, file_name) or re.search(pattern, path_str):
                bonus += 5.0
        
        # Bonus for files in important directories
        important_dirs = ['src', 'lib', 'core', 'main', 'app']
        path_parts = [part.lower() for part in file_path.parts]
        
        for important_dir in important_dirs:
            if important_dir in path_parts:
                bonus += 3.0
                break
        
        return min(bonus, 20.0)  # Cap at 20 points
    
    def get_files_by_type(self, scored_files: List[FileImportanceInfo]) -> Dict[FileType, List[FileImportanceInfo]]:
        """Group files by their type."""
        files_by_type = defaultdict(list)
        
        for file_info in scored_files:
            files_by_type[file_info.file_type].append(file_info)
        
        return dict(files_by_type)
    
    def get_most_efficient_files(self, scored_files: List[FileImportanceInfo], 
                               limit: int = 20) -> List[FileImportanceInfo]:
        """Get files with highest importance-to-token ratio."""
        # Filter out files with zero tokens
        files_with_tokens = [f for f in scored_files if f.tokens > 0]
        
        return sorted(files_with_tokens, key=lambda f: f.efficiency_ratio, reverse=True)[:limit]
    
    def suggest_ignore_patterns(self, scored_files: List[FileImportanceInfo], 
                              min_score_threshold: float = -10.0) -> List[str]:
        """Suggest ignore patterns based on low-importance files."""
        suggestions = set()
        
        # Group low-importance files by patterns
        low_importance_files = [f for f in scored_files if f.importance_score < min_score_threshold]
        
        # Analyze common patterns in low-importance files
        for file_info in low_importance_files:
            path = file_info.relative_path
            
            # Suggest directory patterns
            if '/' in path:
                directory = path.split('/')[0]
                if sum(1 for f in low_importance_files if f.relative_path.startswith(directory + '/')) >= 3:
                    suggestions.add(f"{directory}/*")
            
            # Suggest extension patterns
            if '.' in file_info.path.name:
                extension = file_info.path.suffix
                if sum(1 for f in low_importance_files if f.path.suffix == extension) >= 5:
                    suggestions.add(f"*{extension}")
        
        return sorted(list(suggestions))
    
    def get_importance_summary(self, scored_files: List[FileImportanceInfo]) -> Dict[str, any]:
        """Get a summary of importance analysis."""
        if not scored_files:
            return {}
        
        total_files = len(scored_files)
        total_tokens = sum(f.tokens for f in scored_files)
        
        # Score distribution
        scores = [f.importance_score for f in scored_files]
        
        # File type distribution
        type_counts = defaultdict(int)
        for file_info in scored_files:
            type_counts[file_info.file_type.value] += 1
        
        return {
            'total_files': total_files,
            'total_tokens': total_tokens,
            'average_importance': sum(scores) / len(scores),
            'max_importance': max(scores),
            'min_importance': min(scores),
            'high_importance_files': len([s for s in scores if s > 50]),
            'low_importance_files': len([s for s in scores if s < 0]),
            'file_type_distribution': dict(type_counts),
            'most_important_file': scored_files[0].relative_path if scored_files else None,
            'least_important_file': scored_files[-1].relative_path if scored_files else None
        }