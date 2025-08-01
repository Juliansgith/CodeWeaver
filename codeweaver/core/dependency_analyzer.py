import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class ImportInfo:
    """Represents an import statement found in a file."""
    module: str
    alias: Optional[str] = None
    is_relative: bool = False
    from_module: Optional[str] = None
    line_number: int = 0


@dataclass
class DependencyNode:
    """Represents a file node in the dependency graph."""
    path: Path
    relative_path: str
    imports: List[ImportInfo]
    imported_by: Set[str]  # Files that import this file
    imports_from: Set[str]  # Files this file imports from
    centrality_score: float = 0.0
    importance_score: float = 0.0


class DependencyGraphAnalyzer:
    """
    Analyzes code dependencies to build a graph of file relationships.
    Supports Python, JavaScript/TypeScript, Java, C#, Go, and more.
    """
    
    # Language-specific import patterns
    IMPORT_PATTERNS = {
        '.py': {
            'import': r'^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
            'from_import': r'^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+(.+)',
            'relative_import': r'^\s*from\s+(\.+[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+(.+)'
        },
        '.js': {
            'import': r'^\s*import\s+(?:.*\s+from\s+)?[\'"]([^\'\"]+)[\'"]',
            'require': r'require\([\'"]([^\'\"]+)[\'"]\)',
            'dynamic_import': r'import\([\'"]([^\'\"]+)[\'"]\)'
        },
        '.ts': {
            'import': r'^\s*import\s+(?:.*\s+from\s+)?[\'"]([^\'\"]+)[\'"]',
            'require': r'require\([\'"]([^\'\"]+)[\'"]\)',
            'dynamic_import': r'import\([\'"]([^\'\"]+)[\'"]\)'
        },
        '.jsx': {
            'import': r'^\s*import\s+(?:.*\s+from\s+)?[\'"]([^\'\"]+)[\'"]',
            'require': r'require\([\'"]([^\'\"]+)[\'"]\)'
        },
        '.tsx': {
            'import': r'^\s*import\s+(?:.*\s+from\s+)?[\'"]([^\'\"]+)[\'"]',
            'require': r'require\([\'"]([^\'\"]+)[\'"]\)'
        },
        '.java': {
            'import': r'^\s*import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
        },
        '.cs': {
            'using': r'^\s*using\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
        },
        '.go': {
            'import': r'^\s*import\s+[\'"]([^\'\"]+)[\'"]',
            'import_block': r'import\s*\(\s*([^)]+)\s*\)'
        },
        '.cpp': {
            'include': r'^\s*#include\s+[<"]([^>"]+)[>"]'
        },
        '.hpp': {
            'include': r'^\s*#include\s+[<"]([^>"]+)[>"]'
        },
        '.c': {
            'include': r'^\s*#include\s+[<"]([^>"]+)[>"]'
        },
        '.h': {
            'include': r'^\s*#include\s+[<"]([^>"]+)[>"]'
        }
    }
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.dependency_graph: Dict[str, DependencyNode] = {}
        self.package_structure: Dict[str, Set[str]] = defaultdict(set)
    
    def analyze_dependencies(self, file_paths: List[Path]) -> Dict[str, DependencyNode]:
        """
        Analyze dependencies for all files and build the dependency graph.
        """
        # First pass: extract imports from all files
        for file_path in file_paths:
            try:
                self._analyze_file_imports(file_path)
            except Exception:
                # Skip files that can't be analyzed
                continue
        
        # Second pass: resolve dependencies and build graph
        self._build_dependency_relationships()
        
        # Third pass: calculate centrality scores
        self._calculate_centrality_scores()
        
        return self.dependency_graph
    
    def _analyze_file_imports(self, file_path: Path):
        """Extract import statements from a single file."""
        relative_path = str(file_path.relative_to(self.root_path)).replace("\\", "/")
        suffix = file_path.suffix.lower()
        
        if suffix not in self.IMPORT_PATTERNS:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return
        
        imports = []
        
        # Special handling for Python using AST
        if suffix == '.py':
            imports = self._extract_python_imports_ast(content)
        else:
            imports = self._extract_imports_regex(content, suffix)
        
        # Create dependency node
        node = DependencyNode(
            path=file_path,
            relative_path=relative_path,
            imports=imports,
            imported_by=set(),
            imports_from=set()
        )
        
        self.dependency_graph[relative_path] = node
    
    def _extract_python_imports_ast(self, content: str) -> List[ImportInfo]:
        """Extract Python imports using AST for more accurate parsing."""
        imports = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(ImportInfo(
                            module=alias.name,
                            alias=alias.asname,
                            line_number=node.lineno
                        ))
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        is_relative = node.level > 0
                        module = node.module if not is_relative else ('.' * node.level) + (node.module or '')
                        
                        for alias in node.names:
                            imports.append(ImportInfo(
                                module=alias.name,
                                alias=alias.asname,
                                is_relative=is_relative,
                                from_module=module,
                                line_number=node.lineno
                            ))
        
        except SyntaxError:
            # Fall back to regex if AST parsing fails
            return self._extract_imports_regex(content, '.py')
        
        return imports
    
    def _extract_imports_regex(self, content: str, suffix: str) -> List[ImportInfo]:
        """Extract imports using regex patterns for non-Python files."""
        imports = []
        patterns = self.IMPORT_PATTERNS.get(suffix, {})
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            for pattern_type, pattern in patterns.items():
                matches = re.finditer(pattern, line, re.MULTILINE)
                
                for match in matches:
                    module = match.group(1)
                    
                    # Special handling for Go import blocks
                    if suffix == '.go' and pattern_type == 'import_block':
                        # Parse individual imports within the block
                        import_block = match.group(1)
                        for import_line in import_block.split('\n'):
                            import_match = re.search(r'[\'"]([^\'\"]+)[\'"]', import_line.strip())
                            if import_match:
                                imports.append(ImportInfo(
                                    module=import_match.group(1),
                                    line_number=line_num
                                ))
                    else:
                        imports.append(ImportInfo(
                            module=module,
                            line_number=line_num
                        ))
        
        return imports
    
    def _build_dependency_relationships(self):
        """Build the actual dependency relationships between files."""
        for file_path, node in self.dependency_graph.items():
            for import_info in node.imports:
                # Try to resolve the import to actual files in the project
                resolved_files = self._resolve_import_to_files(import_info, node.path)
                
                for resolved_file in resolved_files:
                    if resolved_file in self.dependency_graph:
                        # This file imports from resolved_file
                        node.imports_from.add(resolved_file)
                        # resolved_file is imported by this file
                        self.dependency_graph[resolved_file].imported_by.add(file_path)
    
    def _resolve_import_to_files(self, import_info: ImportInfo, current_file: Path) -> List[str]:
        """
        Resolve an import statement to actual file paths within the project.
        This is a simplified resolution - in practice, this would need to handle
        module resolution rules for each language.
        """
        resolved = []
        
        # Get current file's directory
        current_dir = current_file.parent
        
        # Handle relative imports (Python)
        if import_info.is_relative:
            if import_info.from_module:
                # Handle relative from imports like "from .module import something"
                relative_path = self._resolve_relative_path(
                    current_dir, import_info.from_module
                )
                if relative_path:
                    resolved.append(relative_path)
        else:
            # Handle absolute imports
            module_path = import_info.module
            if import_info.from_module:
                module_path = import_info.from_module
            
            # Try to find matching files
            potential_files = self._find_files_matching_module(module_path)
            resolved.extend(potential_files)
        
        return resolved
    
    def _resolve_relative_path(self, current_dir: Path, relative_module: str) -> Optional[str]:
        """Resolve relative Python imports to file paths."""
        # Count leading dots
        level = 0
        for char in relative_module:
            if char == '.':
                level += 1
            else:
                break
        
        # Get the module name after the dots
        module_name = relative_module[level:]
        
        # Go up the directory tree
        target_dir = current_dir
        for _ in range(level - 1):
            target_dir = target_dir.parent
            if target_dir == self.root_path.parent:
                break
        
        # Try to find the module file
        if module_name:
            potential_file = target_dir / (module_name.replace('.', '/') + '.py')
            try:
                relative_path = str(potential_file.relative_to(self.root_path)).replace("\\", "/")
                if relative_path in self.dependency_graph:
                    return relative_path
            except ValueError:
                pass
        
        return None
    
    def _find_files_matching_module(self, module_path: str) -> List[str]:
        """Find files that could match the given module path."""
        potential_matches = []
        
        # Convert module path to potential file paths
        module_parts = module_path.split('.')
        
        for file_path in self.dependency_graph.keys():
            path_parts = Path(file_path).with_suffix('').parts
            
            # Check if the file path could match the module
            if len(path_parts) >= len(module_parts):
                if path_parts[-len(module_parts):] == tuple(module_parts):
                    potential_matches.append(file_path)
        
        return potential_matches
    
    def _calculate_centrality_scores(self):
        """Calculate centrality scores using a simplified PageRank-like algorithm."""
        # Initialize scores
        for node in self.dependency_graph.values():
            node.centrality_score = 1.0
        
        # Iterative score calculation (simplified PageRank)
        damping_factor = 0.85
        iterations = 10
        
        for _ in range(iterations):
            new_scores = {}
            
            for file_path, node in self.dependency_graph.items():
                score = (1 - damping_factor)
                
                # Add score from files that import this file
                for importing_file in node.imported_by:
                    if importing_file in self.dependency_graph:
                        importing_node = self.dependency_graph[importing_file]
                        if len(importing_node.imports_from) > 0:
                            score += damping_factor * (importing_node.centrality_score / len(importing_node.imports_from))
                
                new_scores[file_path] = score
            
            # Update scores
            for file_path, score in new_scores.items():
                self.dependency_graph[file_path].centrality_score = score
    
    def get_most_central_files(self, limit: int = 20) -> List[DependencyNode]:
        """Get files with highest centrality scores (most important in dependency graph)."""
        return sorted(
            self.dependency_graph.values(),
            key=lambda node: node.centrality_score,
            reverse=True
        )[:limit]
    
    def get_entry_points(self) -> List[DependencyNode]:
        """Get files that are likely entry points (import many things, imported by few)."""
        entry_points = []
        
        for node in self.dependency_graph.values():
            # Entry points typically import many files but are imported by few
            imports_count = len(node.imports_from)
            imported_by_count = len(node.imported_by)
            
            # Heuristic: entry points have high import ratio
            if imports_count > 0 and (imported_by_count == 0 or imports_count / imported_by_count > 2):
                entry_points.append(node)
        
        return sorted(entry_points, key=lambda node: len(node.imports_from), reverse=True)
    
    def get_dependency_stats(self) -> Dict[str, any]:
        """Get overall dependency statistics."""
        total_files = len(self.dependency_graph)
        total_dependencies = sum(len(node.imports_from) for node in self.dependency_graph.values())
        
        # Calculate complexity metrics
        max_imports = max(len(node.imports_from) for node in self.dependency_graph.values()) if total_files > 0 else 0
        max_imported_by = max(len(node.imported_by) for node in self.dependency_graph.values()) if total_files > 0 else 0
        
        avg_imports = total_dependencies / total_files if total_files > 0 else 0
        
        return {
            'total_files': total_files,
            'total_dependencies': total_dependencies,
            'average_imports_per_file': avg_imports,
            'max_imports_single_file': max_imports,
            'max_times_imported': max_imported_by,
            'dependency_density': total_dependencies / (total_files * total_files) if total_files > 0 else 0
        }
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the project."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(file_path: str, path: List[str]) -> bool:
            if file_path in rec_stack:
                # Found a cycle
                cycle_start = path.index(file_path)
                cycle = path[cycle_start:] + [file_path]
                cycles.append(cycle)
                return True
            
            if file_path in visited:
                return False
            
            visited.add(file_path)
            rec_stack.add(file_path)
            
            node = self.dependency_graph.get(file_path)
            if node:
                for imported_file in node.imports_from:
                    if dfs(imported_file, path + [file_path]):
                        pass  # Continue searching for more cycles
            
            rec_stack.remove(file_path)
            return False
        
        for file_path in self.dependency_graph:
            if file_path not in visited:
                dfs(file_path, [])
        
        return cycles