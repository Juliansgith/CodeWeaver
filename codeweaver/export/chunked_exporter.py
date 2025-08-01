"""
Chunked and split export functionality for large codebases.
Enables splitting large exports into manageable chunks with cross-references.
"""

import json
import hashlib
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .formats import (
    BaseExporter, ExportOptions, FileExportInfo, ExportMetadata,
    MarkdownExporter, JSONExporter, HTMLExporter
)
from ..core.importance_scorer import FileImportanceInfo
from ..core.token_budget import BudgetAllocation


class ChunkStrategy(Enum):
    """Strategy for splitting exports into chunks."""
    BY_SIZE = "by_size"           # Split by total size/tokens
    BY_COUNT = "by_count"         # Split by number of files
    BY_DIRECTORY = "by_directory" # Split by directory structure
    BY_IMPORTANCE = "by_importance" # Split by file importance
    BY_TYPE = "by_type"          # Split by file type/language
    BALANCED = "balanced"         # Balance size and logical grouping


@dataclass
class ChunkReference:
    """Reference to another chunk."""
    chunk_id: str
    chunk_name: str
    file_path: str
    line_number: Optional[int] = None
    context: Optional[str] = None


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    chunk_id: str
    chunk_name: str
    chunk_index: int
    total_chunks: int
    files_count: int
    total_tokens: int
    total_size: int
    chunk_strategy: ChunkStrategy
    created_at: str
    references_to: List[ChunkReference]
    references_from: List[ChunkReference]
    file_paths: List[str]
    main_directories: List[str]
    main_languages: List[str]


@dataclass
class ChunkConfiguration:
    """Configuration for chunked exports."""
    max_tokens_per_chunk: int = 50000
    max_files_per_chunk: int = 50
    max_size_per_chunk: int = 10 * 1024 * 1024  # 10MB
    strategy: ChunkStrategy = ChunkStrategy.BALANCED
    preserve_directory_structure: bool = True
    include_cross_references: bool = True
    reference_context_lines: int = 3
    overlap_threshold: float = 0.1  # Allow 10% overlap between chunks
    min_chunk_size: int = 1000     # Minimum tokens per chunk
    output_format: str = "markdown"
    create_index: bool = True
    create_manifest: bool = True


class CodeReference:
    """Utility class for finding code references between files."""
    
    def __init__(self):
        self.import_patterns = {
            'python': [
                r'from\s+([.\w]+)\s+import',
                r'import\s+([.\w]+)',
            ],
            'javascript': [
                r'import\s+.*?\s+from\s+[\'"]([^\'\"]+)[\'"]',
                r'require\s*\(\s*[\'"]([^\'\"]+)[\'"]\s*\)',
            ],
            'typescript': [
                r'import\s+.*?\s+from\s+[\'"]([^\'\"]+)[\'"]',
                r'require\s*\(\s*[\'"]([^\'\"]+)[\'"]\s*\)',
            ],
            'java': [
                r'import\s+([.\w]+);',
            ],
            'csharp': [
                r'using\s+([.\w]+);',
            ]
        }
    
    def find_references(self, file_content: str, language: str, 
                       available_files: List[str]) -> List[str]:
        """Find references to other files in the codebase."""
        references = []
        language_lower = language.lower()
        
        if language_lower not in self.import_patterns:
            return references
        
        patterns = self.import_patterns[language_lower]
        
        for pattern in patterns:
            import re
            matches = re.finditer(pattern, file_content, re.MULTILINE)
            
            for match in matches:
                ref = match.group(1)
                
                # Try to resolve reference to actual file path
                resolved_path = self._resolve_reference(ref, available_files, language_lower)
                if resolved_path:
                    references.append(resolved_path)
        
        return references
    
    def _resolve_reference(self, reference: str, available_files: List[str], 
                          language: str) -> Optional[str]:
        """Resolve an import reference to an actual file path."""
        # Simple resolution logic - can be enhanced
        for file_path in available_files:
            file_path_lower = file_path.lower()
            
            if language == 'python':
                # Convert module path to file path
                module_path = reference.replace('.', '/') + '.py'
                if module_path in file_path or file_path.endswith(module_path):
                    return file_path
            
            elif language in ['javascript', 'typescript']:
                # Handle relative imports
                if reference.endswith(('.js', '.ts', '.jsx', '.tsx')):
                    if file_path.endswith(reference):
                        return file_path
                else:
                    # Try with common extensions
                    for ext in ['.js', '.ts', '.jsx', '.tsx']:
                        if file_path.endswith(reference + ext):
                            return file_path
            
            elif language == 'java':
                # Convert package path to file path
                class_path = reference.replace('.', '/') + '.java'
                if file_path.endswith(class_path):
                    return file_path
        
        return None


class ChunkedExporter:
    """Main class for chunked exports with cross-references."""
    
    def __init__(self, config: ChunkConfiguration = None):
        self.config = config or ChunkConfiguration()
        self.reference_finder = CodeReference()
    
    def export_chunked(self, files: List[FileExportInfo], metadata: ExportMetadata,
                      output_dir: Path, base_name: str = "codeweaver_chunk") -> bool:
        """Export files in chunks with cross-references."""
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Split files into chunks
            chunks = self._split_into_chunks(files)
            
            # Generate cross-references
            if self.config.include_cross_references:
                chunks = self._generate_cross_references(chunks, files)
            
            # Create chunk metadata
            chunk_metadatas = self._create_chunk_metadata(chunks, metadata)
            
            # Export each chunk
            success = True
            for i, (chunk_files, chunk_meta) in enumerate(zip(chunks, chunk_metadatas)):
                chunk_path = output_dir / f"{base_name}_{i+1:03d}"
                
                if not self._export_single_chunk(chunk_files, chunk_meta, chunk_path):
                    success = False
            
            # Create index and manifest files
            if self.config.create_index:
                self._create_index_file(chunk_metadatas, output_dir, base_name)
            
            if self.config.create_manifest:
                self._create_manifest_file(chunk_metadatas, metadata, output_dir)
            
            return success
            
        except Exception as e:
            print(f"Failed to export chunks: {e}")
            return False
    
    def _split_into_chunks(self, files: List[FileExportInfo]) -> List[List[FileExportInfo]]:
        """Split files into chunks based on the configured strategy."""
        if self.config.strategy == ChunkStrategy.BY_SIZE:
            return self._split_by_size(files)
        elif self.config.strategy == ChunkStrategy.BY_COUNT:
            return self._split_by_count(files)
        elif self.config.strategy == ChunkStrategy.BY_DIRECTORY:
            return self._split_by_directory(files)
        elif self.config.strategy == ChunkStrategy.BY_IMPORTANCE:
            return self._split_by_importance(files)
        elif self.config.strategy == ChunkStrategy.BY_TYPE:
            return self._split_by_type(files)
        else:  # BALANCED
            return self._split_balanced(files)
    
    def _split_by_size(self, files: List[FileExportInfo]) -> List[List[FileExportInfo]]:
        """Split files into chunks based on token/size limits."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_size = 0
        
        # Sort by importance to keep related files together
        sorted_files = sorted(files, key=lambda f: f.importance_score, reverse=True)
        
        for file_info in sorted_files:
            # Check if adding this file would exceed limits
            would_exceed_tokens = current_tokens + file_info.tokens > self.config.max_tokens_per_chunk
            would_exceed_size = current_size + file_info.size_bytes > self.config.max_size_per_chunk
            would_exceed_files = len(current_chunk) >= self.config.max_files_per_chunk
            
            if current_chunk and (would_exceed_tokens or would_exceed_size or would_exceed_files):
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
                current_size = 0
            
            current_chunk.append(file_info)
            current_tokens += file_info.tokens
            current_size += file_info.size_bytes
        
        # Add remaining files
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_count(self, files: List[FileExportInfo]) -> List[List[FileExportInfo]]:
        """Split files into chunks of equal file count."""
        chunk_size = self.config.max_files_per_chunk
        chunks = []
        
        for i in range(0, len(files), chunk_size):
            chunk = files[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_directory(self, files: List[FileExportInfo]) -> List[List[FileExportInfo]]:
        """Split files into chunks by directory structure."""
        # Group files by their top-level directory
        dir_groups = {}
        
        for file_info in files:
            path_parts = file_info.path.split('/')
            top_dir = path_parts[0] if len(path_parts) > 1 else 'root'
            
            if top_dir not in dir_groups:
                dir_groups[top_dir] = []
            dir_groups[top_dir].append(file_info)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for directory, dir_files in dir_groups.items():
            dir_tokens = sum(f.tokens for f in dir_files)
            
            # If directory alone exceeds chunk size, split it
            if dir_tokens > self.config.max_tokens_per_chunk:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                
                # Split large directory
                dir_chunks = self._split_by_size(dir_files)
                chunks.extend(dir_chunks)
            
            # If adding directory would exceed chunk size, start new chunk
            elif current_tokens + dir_tokens > self.config.max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = dir_files.copy()
                current_tokens = dir_tokens
            
            else:
                # Add directory to current chunk
                current_chunk.extend(dir_files)
                current_tokens += dir_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_importance(self, files: List[FileExportInfo]) -> List[List[FileExportInfo]]:
        """Split files into chunks by importance scores."""
        # Sort by importance
        sorted_files = sorted(files, key=lambda f: f.importance_score, reverse=True)
        
        # Calculate importance tiers
        total_files = len(files)
        high_importance = sorted_files[:total_files//3]
        medium_importance = sorted_files[total_files//3:2*total_files//3]
        low_importance = sorted_files[2*total_files//3:]
        
        chunks = []
        
        # Split each tier separately
        for tier in [high_importance, medium_importance, low_importance]:
            if tier:
                tier_chunks = self._split_by_size(tier)
                chunks.extend(tier_chunks)
        
        return chunks
    
    def _split_by_type(self, files: List[FileExportInfo]) -> List[List[FileExportInfo]]:
        """Split files into chunks by file type/language."""
        type_groups = {}
        
        for file_info in files:
            file_type = file_info.language or 'unknown'
            if file_type not in type_groups:
                type_groups[file_type] = []
            type_groups[file_type].append(file_info)
        
        chunks = []
        
        for file_type, type_files in type_groups.items():
            if sum(f.tokens for f in type_files) > self.config.max_tokens_per_chunk:
                # Split large type groups
                type_chunks = self._split_by_size(type_files)
                chunks.extend(type_chunks)
            else:
                chunks.append(type_files)
        
        return chunks
    
    def _split_balanced(self, files: List[FileExportInfo]) -> List[List[FileExportInfo]]:
        """Split files using a balanced approach considering multiple factors."""
        # First try directory-based splitting
        dir_chunks = self._split_by_directory(files)
        
        # If chunks are too large, further split by size
        balanced_chunks = []
        for chunk in dir_chunks:
            chunk_tokens = sum(f.tokens for f in chunk)
            if chunk_tokens > self.config.max_tokens_per_chunk:
                sub_chunks = self._split_by_size(chunk)
                balanced_chunks.extend(sub_chunks)
            else:
                balanced_chunks.append(chunk)
        
        return balanced_chunks
    
    def _generate_cross_references(self, chunks: List[List[FileExportInfo]], 
                                 all_files: List[FileExportInfo]) -> List[List[FileExportInfo]]:
        """Generate cross-references between chunks."""
        # Create file path to chunk mapping
        file_to_chunk = {}
        all_file_paths = []
        
        for chunk_idx, chunk_files in enumerate(chunks):
            for file_info in chunk_files:
                file_to_chunk[file_info.path] = chunk_idx
                all_file_paths.append(file_info.path)
        
        # Find references for each file
        for chunk_idx, chunk_files in enumerate(chunks):
            for file_info in chunk_files:
                if hasattr(file_info, 'content') and file_info.content:
                    references = self.reference_finder.find_references(
                        file_info.content, file_info.language, all_file_paths
                    )
                    
                    # Store references that point to other chunks
                    external_refs = []
                    for ref_path in references:
                        if ref_path in file_to_chunk and file_to_chunk[ref_path] != chunk_idx:
                            external_refs.append({
                                'target_chunk': file_to_chunk[ref_path],
                                'target_file': ref_path,
                                'context': self._extract_reference_context(file_info.content, ref_path)
                            })
                    
                    # Add reference info to file
                    if not hasattr(file_info, 'chunk_references'):
                        file_info.chunk_references = []
                    file_info.chunk_references.extend(external_refs)
        
        return chunks
    
    def _extract_reference_context(self, content: str, referenced_file: str) -> str:
        """Extract context around a reference."""
        lines = content.split('\n')
        context_lines = []
        
        for i, line in enumerate(lines):
            if referenced_file in line or any(part in line for part in referenced_file.split('/')):
                start = max(0, i - self.config.reference_context_lines)
                end = min(len(lines), i + self.config.reference_context_lines + 1)
                context_lines = lines[start:end]
                break
        
        return '\n'.join(context_lines)
    
    def _create_chunk_metadata(self, chunks: List[List[FileExportInfo]], 
                             metadata: ExportMetadata) -> List[ChunkMetadata]:
        """Create metadata for each chunk."""
        chunk_metadatas = []
        
        for i, chunk_files in enumerate(chunks):
            chunk_id = hashlib.md5(f"chunk_{i}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
            
            # Analyze chunk content
            directories = set()
            languages = set()
            total_tokens = 0
            total_size = 0
            file_paths = []
            
            for file_info in chunk_files:
                file_paths.append(file_info.path)
                total_tokens += file_info.tokens
                total_size += file_info.size_bytes
                
                if file_info.language:
                    languages.add(file_info.language)
                
                # Extract directory
                path_parts = file_info.path.split('/')
                if len(path_parts) > 1:
                    directories.add(path_parts[0])
                else:
                    directories.add('root')
            
            # Determine chunk name
            if len(directories) == 1:
                chunk_name = f"Chunk {i+1}: {list(directories)[0]}"
            else:
                chunk_name = f"Chunk {i+1}: Mixed ({', '.join(sorted(directories)[:3])})"
            
            chunk_meta = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_name=chunk_name,
                chunk_index=i,
                total_chunks=len(chunks),
                files_count=len(chunk_files),
                total_tokens=total_tokens,
                total_size=total_size,
                chunk_strategy=self.config.strategy,
                created_at=datetime.now().isoformat(),
                references_to=[],  # Will be populated later
                references_from=[],
                file_paths=file_paths,
                main_directories=sorted(directories),
                main_languages=sorted(languages)
            )
            
            chunk_metadatas.append(chunk_meta)
        
        return chunk_metadatas
    
    def _export_single_chunk(self, chunk_files: List[FileExportInfo], 
                           chunk_meta: ChunkMetadata, output_path: Path) -> bool:
        """Export a single chunk."""
        try:
            # Create chunk-specific metadata
            chunk_metadata = ExportMetadata(
                project_path=chunk_meta.chunk_name,
                generated_at=chunk_meta.created_at,
                total_files=chunk_meta.files_count,
                total_tokens=chunk_meta.total_tokens,
                export_format=self.config.output_format,
                options={
                    'chunk_id': chunk_meta.chunk_id,
                    'chunk_index': chunk_meta.chunk_index,
                    'total_chunks': chunk_meta.total_chunks,
                    'chunk_strategy': chunk_meta.chunk_strategy.value
                }
            )
            
            # Choose exporter based on format
            if self.config.output_format.lower() == 'json':
                exporter = JSONExporter(ExportOptions(include_metadata=True))
                file_extension = '.json'
            elif self.config.output_format.lower() == 'html':
                exporter = HTMLExporter(ExportOptions(include_metadata=True))
                file_extension = '.html'
            else:  # Default to markdown
                exporter = MarkdownExporter(ExportOptions(include_metadata=True))
                file_extension = '.md'
            
            # Add chunk navigation to the beginning of each file
            enhanced_files = self._add_chunk_navigation(chunk_files, chunk_meta)
            
            # Export chunk
            chunk_file_path = output_path.with_suffix(file_extension)
            return exporter.export(enhanced_files, chunk_metadata, chunk_file_path)
            
        except Exception as e:
            print(f"Failed to export chunk {chunk_meta.chunk_name}: {e}")
            return False
    
    def _add_chunk_navigation(self, chunk_files: List[FileExportInfo], 
                            chunk_meta: ChunkMetadata) -> List[FileExportInfo]:
        """Add navigation information to chunk files."""
        if not chunk_files:
            return chunk_files
        
        # Create navigation header
        nav_content = f"""
# {chunk_meta.chunk_name}

**Chunk {chunk_meta.chunk_index + 1} of {chunk_meta.total_chunks}**

## Chunk Information
- **Files:** {chunk_meta.files_count}
- **Total Tokens:** {chunk_meta.total_tokens:,}
- **Total Size:** {chunk_meta.total_size:,} bytes
- **Main Directories:** {', '.join(chunk_meta.main_directories)}
- **Languages:** {', '.join(chunk_meta.main_languages)}
- **Strategy:** {chunk_meta.chunk_strategy.value}

## Navigation
- **Previous Chunk:** {chunk_meta.chunk_index if chunk_meta.chunk_index > 0 else 'N/A'}
- **Next Chunk:** {chunk_meta.chunk_index + 2 if chunk_meta.chunk_index + 1 < chunk_meta.total_chunks else 'N/A'}

## Cross-References
"""
        
        # Add cross-reference information
        cross_refs = set()
        for file_info in chunk_files:
            if hasattr(file_info, 'chunk_references'):
                for ref in file_info.chunk_references:
                    cross_refs.add(f"- **Chunk {ref['target_chunk'] + 1}:** {ref['target_file']}")
        
        if cross_refs:
            nav_content += '\n'.join(sorted(cross_refs))
        else:
            nav_content += "No cross-references found."
        
        nav_content += "\n\n---\n\n"
        
        # Add navigation to first file
        enhanced_files = chunk_files.copy()
        if enhanced_files:
            first_file = enhanced_files[0]
            first_file.content = nav_content + first_file.content
        
        return enhanced_files
    
    def _create_index_file(self, chunk_metadatas: List[ChunkMetadata], 
                          output_dir: Path, base_name: str):
        """Create an index file for all chunks."""
        index_content = f"""# Codeweaver Chunked Export Index

Generated: {datetime.now().isoformat()}
Total Chunks: {len(chunk_metadatas)}
Strategy: {self.config.strategy.value}

## Chunks Overview

"""
        
        total_files = sum(meta.files_count for meta in chunk_metadatas)
        total_tokens = sum(meta.total_tokens for meta in chunk_metadatas)
        total_size = sum(meta.total_size for meta in chunk_metadatas)
        
        index_content += f"""### Summary Statistics
- **Total Files:** {total_files}
- **Total Tokens:** {total_tokens:,}
- **Total Size:** {total_size:,} bytes
- **Average Files per Chunk:** {total_files // len(chunk_metadatas)}
- **Average Tokens per Chunk:** {total_tokens // len(chunk_metadatas):,}

### Chunks Detail

"""
        
        for meta in chunk_metadatas:
            file_ext = '.json' if self.config.output_format == 'json' else '.md'
            chunk_filename = f"{base_name}_{meta.chunk_index + 1:03d}{file_ext}"
            
            index_content += f"""#### [{meta.chunk_name}]({chunk_filename})
- **Files:** {meta.files_count}
- **Tokens:** {meta.total_tokens:,}
- **Size:** {meta.total_size:,} bytes
- **Directories:** {', '.join(meta.main_directories)}
- **Languages:** {', '.join(meta.main_languages)}

"""
        
        # Write index file
        index_path = output_dir / "INDEX.md"
        index_path.write_text(index_content, encoding='utf-8')
    
    def _create_manifest_file(self, chunk_metadatas: List[ChunkMetadata], 
                            metadata: ExportMetadata, output_dir: Path):
        """Create a machine-readable manifest file."""
        manifest = {
            'export_info': asdict(metadata),
            'chunk_config': asdict(self.config),
            'chunks': [asdict(meta) for meta in chunk_metadatas],
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


def create_chunked_export(files: List[FileExportInfo], metadata: ExportMetadata,
                         output_dir: Path, config: ChunkConfiguration = None) -> bool:
    """Convenience function to create a chunked export."""
    exporter = ChunkedExporter(config or ChunkConfiguration())
    return exporter.export_chunked(files, metadata, output_dir)