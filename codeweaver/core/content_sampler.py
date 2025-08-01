"""
Intelligent file sampling for large files and semantic content extraction.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import ast
import json

from .importance_scorer import FileType
from ..ai.embeddings import GeminiEmbeddingService


class SamplingStrategy(Enum):
    """Different strategies for sampling file content."""
    HEAD_TAIL = "head_tail"           # First N + last N lines
    KEY_SECTIONS = "key_sections"     # Extract key sections (functions, classes, etc.)
    SEMANTIC = "semantic"             # AI-powered semantic sampling
    CHANGE_BASED = "change_based"     # Focus on recently modified sections
    BALANCED = "balanced"             # Combination of strategies


@dataclass
class SampleSection:
    """A sampled section of a file."""
    start_line: int
    end_line: int
    content: str
    section_type: str              # 'function', 'class', 'import', 'main', etc.
    importance_score: float        # 0.0 to 1.0
    reasoning: str                 # Why this section was selected


@dataclass
class FileSample:
    """Result of sampling a file."""
    file_path: str
    original_size: int
    sampled_size: int
    reduction_ratio: float
    sections: List[SampleSection]
    strategy_used: SamplingStrategy
    metadata: Dict[str, Any]


class LanguageAnalyzer:
    """Analyzes code structure for different programming languages."""
    
    def __init__(self):
        self.language_patterns = self._build_language_patterns()
    
    def _build_language_patterns(self) -> Dict[str, Dict[str, str]]:
        """Build regex patterns for different languages."""
        return {
            'python': {
                'function': r'^[ \t]*def\s+(\w+)\s*\([^)]*\)\s*:',
                'class': r'^[ \t]*class\s+(\w+).*:',
                'import': r'^[ \t]*(?:import|from)\s+',
                'comment': r'^[ \t]*#',
                'docstring': r'^\s*""".*?"""',
                'decorator': r'^[ \t]*@\w+'
            },
            'javascript': {
                'function': r'^[ \t]*(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))',
                'class': r'^[ \t]*class\s+(\w+)',
                'import': r'^[ \t]*(?:import|export)',
                'comment': r'^[ \t]*(?://|/\*)',
                'arrow_function': r'^[ \t]*(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
            },
            'typescript': {
                'function': r'^[ \t]*(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*:\s*\([^)]*\)\s*=>)',
                'class': r'^[ \t]*(?:export\s+)?class\s+(\w+)',
                'interface': r'^[ \t]*(?:export\s+)?interface\s+(\w+)',
                'type': r'^[ \t]*(?:export\s+)?type\s+(\w+)',
                'import': r'^[ \t]*(?:import|export)',
                'comment': r'^[ \t]*(?://|/\*)'
            },
            'java': {
                'function': r'^[ \t]*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{',
                'class': r'^[ \t]*(?:public|private)?\s*class\s+(\w+)',
                'interface': r'^[ \t]*(?:public|private)?\s*interface\s+(\w+)',
                'import': r'^[ \t]*import\s+',
                'comment': r'^[ \t]*(?://|/\*)',
                'annotation': r'^[ \t]*@\w+'
            },
            'go': {
                'function': r'^[ \t]*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\([^)]*\)',
                'struct': r'^[ \t]*type\s+(\w+)\s+struct',
                'interface': r'^[ \t]*type\s+(\w+)\s+interface',
                'import': r'^[ \t]*import\s*',
                'comment': r'^[ \t]*(?://|/\*)'
            },
            'rust': {
                'function': r'^[ \t]*(?:pub\s+)?fn\s+(\w+)\s*\(',
                'struct': r'^[ \t]*(?:pub\s+)?struct\s+(\w+)',
                'impl': r'^[ \t]*impl(?:\s*<[^>]*>)?\s+(\w+)',
                'trait': r'^[ \t]*(?:pub\s+)?trait\s+(\w+)',
                'use': r'^[ \t]*use\s+',
                'comment': r'^[ \t]*(?://|/\*)'
            },
            'csharp': {
                'function': r'^[ \t]*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)',
                'class': r'^[ \t]*(?:public|private|internal)?\s*class\s+(\w+)',
                'interface': r'^[ \t]*(?:public|private|internal)?\s*interface\s+(\w+)',
                'using': r'^[ \t]*using\s+',
                'comment': r'^[ \t]*(?://|/\*)',
                'attribute': r'^[ \t]*\[\w+.*\]'
            }
        }
    
    def analyze_structure(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Analyze the structure of code content."""
        if language not in self.language_patterns:
            return self._generic_structure_analysis(content)
        
        patterns = self.language_patterns[language]
        lines = content.split('\n')
        structures = []
        
        current_function = None
        current_class = None
        brace_depth = 0
        in_multiline_comment = False
        
        for i, line in enumerate(lines):
            line_info = {
                'line_number': i + 1,
                'content': line,
                'type': 'code',
                'parent': None,
                'importance': 0.5
            }
            
            # Handle multiline comments
            if '/*' in line:
                in_multiline_comment = True
            if '*/' in line:
                in_multiline_comment = False
                
            if in_multiline_comment:
                line_info['type'] = 'comment'
                line_info['importance'] = 0.1
                structures.append(line_info)
                continue
            
            # Track brace depth for scope
            brace_depth += line.count('{') - line.count('}')
            
            # Check patterns
            for pattern_type, pattern in patterns.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    line_info['type'] = pattern_type
                    if match.groups():
                        line_info['name'] = match.group(1)
                    
                    # Set importance based on type
                    importance_map = {
                        'function': 0.9,
                        'class': 0.95,
                        'interface': 0.9,
                        'struct': 0.9,
                        'trait': 0.85,
                        'impl': 0.85,
                        'import': 0.3,
                        'using': 0.3,
                        'comment': 0.1,
                        'docstring': 0.4
                    }
                    line_info['importance'] = importance_map.get(pattern_type, 0.5)
                    
                    # Track current scope
                    if pattern_type in ['function', 'method']:
                        current_function = line_info['name'] if 'name' in line_info else None
                        line_info['parent'] = current_class
                    elif pattern_type in ['class', 'struct']:
                        current_class = line_info['name'] if 'name' in line_info else None
                        current_function = None
                    
                    break
            
            line_info['scope_depth'] = brace_depth
            line_info['current_function'] = current_function
            line_info['current_class'] = current_class
            
            structures.append(line_info)
        
        return structures
    
    def _generic_structure_analysis(self, content: str) -> List[Dict[str, Any]]:
        """Generic structure analysis for unknown languages."""
        lines = content.split('\n')
        structures = []
        
        for i, line in enumerate(lines):
            line_info = {
                'line_number': i + 1,
                'content': line,
                'type': 'code',
                'importance': 0.5
            }
            
            # Simple heuristics
            stripped = line.strip()
            if not stripped:
                line_info['type'] = 'empty'
                line_info['importance'] = 0.0
            elif stripped.startswith('#') or stripped.startswith('//'):
                line_info['type'] = 'comment'
                line_info['importance'] = 0.1
            elif any(keyword in stripped.lower() for keyword in ['import', 'include', 'require', 'use']):
                line_info['type'] = 'import'
                line_info['importance'] = 0.3
            elif any(keyword in stripped.lower() for keyword in ['function', 'def', 'func', 'method']):
                line_info['type'] = 'function'
                line_info['importance'] = 0.9
            elif any(keyword in stripped.lower() for keyword in ['class', 'struct', 'interface']):
                line_info['type'] = 'class'
                line_info['importance'] = 0.95
            
            structures.append(line_info)
        
        return structures


class ContentSampler:
    """Intelligent content sampler for large files."""
    
    def __init__(self, embedding_service: Optional[GeminiEmbeddingService] = None):
        self.embedding_service = embedding_service
        self.language_analyzer = LanguageAnalyzer()
        
        # Sampling configuration
        self.max_lines_threshold = 500      # Files larger than this get sampled
        self.head_tail_lines = 100          # Lines to take from head/tail
        self.max_sampled_lines = 300        # Maximum lines in final sample
        self.min_section_lines = 5          # Minimum lines for a section
    
    async def sample_file(self, 
                         file_path: Path, 
                         content: str,
                         language: str,
                         strategy: SamplingStrategy = SamplingStrategy.BALANCED,
                         target_lines: Optional[int] = None,
                         purpose: Optional[str] = None) -> FileSample:
        """
        Sample file content using the specified strategy.
        
        Args:
            file_path: Path to the file
            content: File content
            language: Programming language
            strategy: Sampling strategy to use
            target_lines: Target number of lines (overrides default)
            purpose: User's purpose for context-aware sampling
        """
        lines = content.split('\n')
        original_size = len(lines)
        
        # If file is small enough, return as-is
        if original_size <= self.max_lines_threshold:
            return FileSample(
                file_path=str(file_path),
                original_size=original_size,
                sampled_size=original_size,
                reduction_ratio=1.0,
                sections=[SampleSection(
                    start_line=1,
                    end_line=original_size,
                    content=content,
                    section_type='full',
                    importance_score=1.0,
                    reasoning="File small enough to include in full"
                )],
                strategy_used=strategy,
                metadata={'no_sampling_needed': True}
            )
        
        target_lines = target_lines or self.max_sampled_lines
        
        # Apply sampling strategy
        if strategy == SamplingStrategy.HEAD_TAIL:
            sections = self._sample_head_tail(lines, target_lines)
        elif strategy == SamplingStrategy.KEY_SECTIONS:
            sections = await self._sample_key_sections(lines, language, target_lines)
        elif strategy == SamplingStrategy.SEMANTIC:
            sections = await self._sample_semantic(lines, language, target_lines, purpose)
        elif strategy == SamplingStrategy.CHANGE_BASED:
            sections = await self._sample_change_based(lines, file_path, target_lines)
        else:  # BALANCED
            sections = await self._sample_balanced(lines, language, target_lines, purpose)
        
        # Calculate final metrics
        sampled_content = '\n'.join([s.content for s in sections])
        sampled_size = len(sampled_content.split('\n'))
        reduction_ratio = sampled_size / original_size if original_size > 0 else 0
        
        return FileSample(
            file_path=str(file_path),
            original_size=original_size,
            sampled_size=sampled_size,
            reduction_ratio=reduction_ratio,
            sections=sections,
            strategy_used=strategy,
            metadata={
                'language': language,
                'target_lines': target_lines,
                'compression_achieved': 1 - reduction_ratio
            }
        )
    
    def _sample_head_tail(self, lines: List[str], target_lines: int) -> List[SampleSection]:
        """Sample head and tail of the file."""
        head_lines = target_lines // 2
        tail_lines = target_lines - head_lines
        
        sections = []
        
        # Head section
        if head_lines > 0:
            head_content = '\n'.join(lines[:head_lines])
            sections.append(SampleSection(
                start_line=1,
                end_line=head_lines,
                content=head_content,
                section_type='head',
                importance_score=0.8,
                reasoning=f"First {head_lines} lines containing imports and initial definitions"
            ))
        
        # Tail section
        if tail_lines > 0 and len(lines) > head_lines:
            tail_start = max(head_lines + 1, len(lines) - tail_lines)
            tail_content = '\n'.join(lines[tail_start:])
            sections.append(SampleSection(
                start_line=tail_start + 1,
                end_line=len(lines),
                content=tail_content,
                section_type='tail',
                importance_score=0.7,
                reasoning=f"Last {tail_lines} lines containing main logic and cleanup"
            ))
        
        return sections
    
    async def _sample_key_sections(self, lines: List[str], language: str, target_lines: int) -> List[SampleSection]:
        """Sample key sections like functions, classes, etc."""
        content = '\n'.join(lines)
        structure = self.language_analyzer.analyze_structure(content, language)
        
        # Group lines by importance and type
        important_lines = []
        for line_info in structure:
            if line_info['importance'] >= 0.7:  # High importance threshold
                important_lines.append(line_info)
        
        # If we don't have enough important lines, lower the threshold
        if len(important_lines) < target_lines // 2:
            important_lines = [l for l in structure if l['importance'] >= 0.5]
        
        # Sort by importance and take top lines
        important_lines.sort(key=lambda x: x['importance'], reverse=True)
        selected_lines = important_lines[:target_lines]
        
        # Group consecutive lines into sections
        sections = self._group_consecutive_lines(selected_lines, lines)
        
        return sections
    
    async def _sample_semantic(self, lines: List[str], language: str, 
                              target_lines: int, purpose: Optional[str]) -> List[SampleSection]:
        """Use AI embeddings to find semantically relevant sections."""
        if not self.embedding_service or not purpose:
            # Fallback to key sections if no embedding service or purpose
            return await self._sample_key_sections(lines, language, target_lines)
        
        try:
            # Split content into chunks (functions, classes, etc.)
            content = '\n'.join(lines)
            structure = self.language_analyzer.analyze_structure(content, language)
            
            # Create chunks from structure
            chunks = self._create_semantic_chunks(structure, lines)
            
            if not chunks:
                return await self._sample_key_sections(lines, language, target_lines)
            
            # Get embeddings for chunks and purpose
            chunk_texts = [chunk['content'] for chunk in chunks]
            chunk_embeddings = await self.embedding_service.get_embeddings(chunk_texts)
            purpose_embedding = await self.embedding_service.get_embeddings([purpose])
            
            if not chunk_embeddings or not purpose_embedding:
                return await self._sample_key_sections(lines, language, target_lines)
            
            # Calculate similarities
            from ..ai.embeddings import calculate_cosine_similarity
            similarities = []
            
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = calculate_cosine_similarity(chunk_embedding, purpose_embedding[0])
                chunks[i]['similarity'] = similarity
                similarities.append((i, similarity))
            
            # Sort by similarity and select top chunks
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            selected_chunks = []
            total_lines = 0
            
            for chunk_idx, similarity in similarities:
                chunk = chunks[chunk_idx]
                chunk_lines = len(chunk['content'].split('\n'))
                
                if total_lines + chunk_lines <= target_lines:
                    selected_chunks.append(chunk)
                    total_lines += chunk_lines
                
                if total_lines >= target_lines:
                    break
            
            # Convert chunks to sections
            sections = []
            for chunk in selected_chunks:
                sections.append(SampleSection(
                    start_line=chunk['start_line'],
                    end_line=chunk['end_line'],
                    content=chunk['content'],
                    section_type=chunk['type'],
                    importance_score=chunk['similarity'],
                    reasoning=f"Semantically relevant to '{purpose}' (similarity: {chunk['similarity']:.3f})"
                ))
            
            return sections
            
        except Exception as e:
            logging.warning(f"Semantic sampling failed: {e}, falling back to key sections")
            return await self._sample_key_sections(lines, language, target_lines)
    
    async def _sample_change_based(self, lines: List[str], file_path: Path, target_lines: int) -> List[SampleSection]:
        """Sample based on recent changes (requires git history)."""
        # This is a placeholder implementation
        # In a full implementation, this would use git blame or similar to find recently changed lines
        logging.info("Change-based sampling not fully implemented, falling back to key sections")
        return await self._sample_key_sections(lines, 'generic', target_lines)
    
    async def _sample_balanced(self, lines: List[str], language: str, 
                              target_lines: int, purpose: Optional[str]) -> List[SampleSection]:
        """Balanced sampling combining multiple strategies."""
        # Allocate lines to different strategies
        head_tail_lines = target_lines // 4
        key_sections_lines = target_lines // 2
        semantic_lines = target_lines - head_tail_lines - key_sections_lines
        
        all_sections = []
        used_line_ranges = set()
        
        # 1. Get head/tail sections
        head_tail_sections = self._sample_head_tail(lines, head_tail_lines)
        for section in head_tail_sections:
            for line_num in range(section.start_line, section.end_line + 1):
                used_line_ranges.add(line_num)
        all_sections.extend(head_tail_sections)
        
        # 2. Get key sections (avoiding overlap)
        key_sections = await self._sample_key_sections(lines, language, key_sections_lines * 2)  # Get more to filter
        filtered_key_sections = []
        key_lines_added = 0
        
        for section in key_sections:
            # Check for overlap
            section_lines = set(range(section.start_line, section.end_line + 1))
            if not section_lines.intersection(used_line_ranges) and key_lines_added < key_sections_lines:
                filtered_key_sections.append(section)
                used_line_ranges.update(section_lines)
                key_lines_added += len(section.content.split('\n'))
        
        all_sections.extend(filtered_key_sections)
        
        # 3. Get semantic sections if possible (avoiding overlap)
        if purpose and semantic_lines > 0:
            semantic_sections = await self._sample_semantic(lines, language, semantic_lines * 2, purpose)
            filtered_semantic_sections = []
            semantic_lines_added = 0
            
            for section in semantic_sections:
                section_lines = set(range(section.start_line, section.end_line + 1))
                if not section_lines.intersection(used_line_ranges) and semantic_lines_added < semantic_lines:
                    filtered_semantic_sections.append(section)
                    used_line_ranges.update(section_lines)
                    semantic_lines_added += len(section.content.split('\n'))
            
            all_sections.extend(filtered_semantic_sections)
        
        # Sort sections by start line
        all_sections.sort(key=lambda x: x.start_line)
        
        return all_sections
    
    def _create_semantic_chunks(self, structure: List[Dict[str, Any]], lines: List[str]) -> List[Dict[str, Any]]:
        """Create semantic chunks from code structure."""
        chunks = []
        current_chunk = None
        
        for line_info in structure:
            line_num = line_info['line_number']
            line_type = line_info['type']
            
            # Start new chunk for important sections
            if line_type in ['function', 'class', 'interface', 'struct'] and line_info['importance'] >= 0.7:
                if current_chunk:
                    chunks.append(current_chunk)
                
                current_chunk = {
                    'start_line': line_num,
                    'end_line': line_num,
                    'type': line_type,
                    'content': line_info['content'],
                    'importance': line_info['importance']
                }
            elif current_chunk:
                # Extend current chunk
                current_chunk['end_line'] = line_num
                current_chunk['content'] += '\n' + line_info['content']
            elif line_info['importance'] >= 0.5:
                # Start a new chunk for moderately important lines
                current_chunk = {
                    'start_line': line_num,
                    'end_line': line_num,
                    'type': line_type,
                    'content': line_info['content'],
                    'importance': line_info['importance']
                }
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Filter out very small chunks
        filtered_chunks = [c for c in chunks if (c['end_line'] - c['start_line'] + 1) >= self.min_section_lines]
        
        return filtered_chunks
    
    def _group_consecutive_lines(self, selected_lines: List[Dict[str, Any]], 
                                all_lines: List[str]) -> List[SampleSection]:
        """Group consecutive lines into sections."""
        if not selected_lines:
            return []
        
        # Sort by line number
        selected_lines.sort(key=lambda x: x['line_number'])
        
        sections = []
        current_group = [selected_lines[0]]
        
        for line_info in selected_lines[1:]:
            # If line is consecutive or very close, add to current group
            if line_info['line_number'] - current_group[-1]['line_number'] <= 3:
                current_group.append(line_info)
            else:
                # Finalize current group and start new one
                if current_group:
                    sections.append(self._create_section_from_group(current_group, all_lines))
                current_group = [line_info]
        
        # Add final group
        if current_group:
            sections.append(self._create_section_from_group(current_group, all_lines))
        
        return sections
    
    def _create_section_from_group(self, group: List[Dict[str, Any]], 
                                  all_lines: List[str]) -> SampleSection:
        """Create a SampleSection from a group of consecutive lines."""
        start_line = group[0]['line_number']
        end_line = group[-1]['line_number']
        
        # Expand to include a bit more context if needed
        context_lines = 2
        actual_start = max(1, start_line - context_lines)
        actual_end = min(len(all_lines), end_line + context_lines)
        
        content_lines = all_lines[actual_start-1:actual_end]
        content = '\n'.join(content_lines)
        
        # Determine section type
        types = [line_info.get('type', 'code') for line_info in group]
        section_type = max(set(types), key=types.count)  # Most common type
        
        # Calculate average importance
        importance = sum(line_info.get('importance', 0.5) for line_info in group) / len(group)
        
        # Generate reasoning
        unique_types = list(set(types))
        if len(unique_types) == 1:
            reasoning = f"Contains {section_type} definitions"
        else:
            reasoning = f"Contains {', '.join(unique_types)} (mixed content)"
        
        return SampleSection(
            start_line=actual_start,
            end_line=actual_end,
            content=content,
            section_type=section_type,
            importance_score=importance,
            reasoning=reasoning
        )
    
    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get statistics about the content sampler."""
        return {
            'max_lines_threshold': self.max_lines_threshold,
            'head_tail_lines': self.head_tail_lines,
            'max_sampled_lines': self.max_sampled_lines,
            'min_section_lines': self.min_section_lines,
            'supported_languages': list(self.language_analyzer.language_patterns.keys()),
            'sampling_strategies': [s.value for s in SamplingStrategy]
        }