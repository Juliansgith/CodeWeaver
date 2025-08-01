import os
import asyncio
from pathlib import Path
from typing import List, Callable, Optional
import queue

from .models import ProcessingResult, ProcessingStats, ProcessingOptions
from .tokenizer import TokenEstimator, LLMProvider
from .analyzer import TokenAnalyzer
from .content_filter import ContentFilter
from .content_sampler import ContentSampler, SamplingStrategy, FileSample


class TreeGenerator:
    @staticmethod
    def generate_project_tree(root_dir: str, file_paths: List[Path]) -> str:
        tree = {}
        root_path = Path(root_dir)
        
        for path in file_paths:
            try:
                relative_path = path.relative_to(root_path)
            except ValueError:
                continue
            
            parts = relative_path.parts
            current_level = tree
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

        def _build_tree_lines(d, prefix=''):
            lines = []
            items = sorted(d.keys())
            for i, name in enumerate(items):
                connector = '└── ' if i == len(items) - 1 else '├── '
                lines.append(prefix + connector + name)
                if d[name]:
                    extension = '│   ' if i < len(items) - 1 else '    '
                    lines.extend(_build_tree_lines(d[name], prefix + extension))
            return lines

        tree_lines = [os.path.basename(root_dir)]
        tree_lines.extend(_build_tree_lines(tree))
        return "\n".join(tree_lines)


class FileFilter:
    def __init__(self, ignore_patterns: List[str], size_limit_mb: float):
        self.ignore_patterns = ignore_patterns
        self.size_limit_bytes = size_limit_mb * 1024 * 1024

    def should_ignore_directory(self, dir_path: Path) -> bool:
        return any(dir_path.match(pattern) for pattern in self.ignore_patterns)

    def should_ignore_file(self, file_path: Path) -> bool:
        if any(file_path.match(pattern) for pattern in self.ignore_patterns):
            return True
        
        try:
            if file_path.stat().st_size > self.size_limit_bytes:
                return True
        except (FileNotFoundError, OSError):
            return True
        
        return False


class CodebaseProcessor:
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None,
                 progress_callback: Optional[Callable[[tuple], None]] = None):
        self.log_callback = log_callback or (lambda x: None)
        self.progress_callback = progress_callback or (lambda x: None)
        self.content_filter = ContentFilter()
        self.content_sampler = ContentSampler()

    def process(self, options: ProcessingOptions) -> ProcessingResult:
        try:
            self.log_callback(f"--- Starting {options.mode} for: {options.input_dir} ---")
            self.progress_callback(('indeterminate',))
            
            project_files = self._collect_files(options)
            
            if not project_files:
                return ProcessingResult(
                    success=False,
                    message='No files to include after filtering.'
                )

            if options.mode == 'preview':
                return ProcessingResult(
                    success=True,
                    files=sorted(project_files)
                )

            return asyncio.run(self._generate_digest(options.input_dir, project_files, options))
            
        except Exception as e:
            import traceback
            self.log_callback(f"!!! AN ERROR OCCURRED: {e}")
            self.log_callback(traceback.format_exc())
            return ProcessingResult(success=False, message=str(e))

    def _collect_files(self, options: ProcessingOptions) -> List[Path]:
        file_filter = FileFilter(options.ignore_patterns, options.size_limit_mb)
        project_files = []
        input_path_obj = Path(options.input_dir)

        for root, dirs, files in os.walk(options.input_dir, topdown=True):
            root_path = Path(root)
            
            # Filter directories in-place to avoid traversing ignored directories
            dirs[:] = [d for d in dirs if not file_filter.should_ignore_directory(Path(root, d))]
            
            for file in files:
                file_path = root_path / file
                if not file_path.is_file():
                    continue

                if file_filter.should_ignore_file(file_path):
                    if file_path.stat().st_size > file_filter.size_limit_bytes:
                        self.log_callback(f"  - Ignoring (too large): {file_path.relative_to(input_path_obj)}")
                    continue
                
                project_files.append(file_path)
        
        return project_files
    
    async def _apply_intelligent_sampling(self, file_path: Path, content: str, 
                                         language: str, options: ProcessingOptions) -> str:
        """Apply intelligent sampling to file content."""
        try:
            # Map string strategy to enum
            strategy_map = {
                'head_tail': SamplingStrategy.HEAD_TAIL,
                'key_sections': SamplingStrategy.KEY_SECTIONS,
                'semantic': SamplingStrategy.SEMANTIC,
                'change_based': SamplingStrategy.CHANGE_BASED,
                'balanced': SamplingStrategy.BALANCED
            }
            
            strategy = strategy_map.get(options.sampling_strategy, SamplingStrategy.BALANCED)
            
            # Apply sampling
            sample_result = await self.content_sampler.sample_file(
                file_path=file_path,
                content=content,
                language=language,
                strategy=strategy,
                target_lines=options.sampling_max_lines,
                purpose=options.sampling_purpose
            )
            
            # If no sampling was needed, return original content
            if sample_result.reduction_ratio == 1.0:
                return content
            
            # Build sampled content with metadata
            sampled_content = ""
            
            # Add sampling information
            if len(sample_result.sections) > 1:
                sampled_content += f"// File sampled: {sample_result.original_size} → {sample_result.sampled_size} lines "
                sampled_content += f"({sample_result.reduction_ratio:.1%} of original)\n"
                sampled_content += f"// Strategy: {sample_result.strategy_used.value}\n\n"
            
            # Add sections
            for i, section in enumerate(sample_result.sections):
                if len(sample_result.sections) > 1:
                    sampled_content += f"// --- Section {i+1}: {section.section_type} "
                    sampled_content += f"(lines {section.start_line}-{section.end_line}) ---\n"
                    sampled_content += f"// {section.reasoning}\n\n"
                
                sampled_content += section.content
                
                if i < len(sample_result.sections) - 1:
                    sampled_content += "\n\n"
            
            return sampled_content
            
        except Exception as e:
            self.log_callback(f"Intelligent sampling failed for {file_path}: {e}")
            # Fall back to simple head/tail sampling
            lines = content.split('\n')
            if len(lines) > options.sampling_max_lines:
                head_lines = options.sampling_max_lines // 2
                tail_lines = options.sampling_max_lines - head_lines
                head_content = '\n'.join(lines[:head_lines])
                tail_content = '\n'.join(lines[-tail_lines:])
                return f"{head_content}\n\n// ... (content truncated) ...\n\n{tail_content}"
            return content

    async def _generate_digest(self, input_dir: str, project_files: List[Path], options: ProcessingOptions) -> ProcessingResult:
        input_path_obj = Path(input_dir)
        output_file = input_path_obj / "codebase.md"
        
        self.log_callback(f"Output will be: {output_file}")
        self.progress_callback(('determinate', len(project_files)))
        
        total_content = ""
        tree_string = TreeGenerator.generate_project_tree(input_dir, project_files)
        initial_content = f"# Project Structure\n\n```\n{tree_string}\n```\n\n---\n\n# File Contents\n\n"
        total_content += initial_content

        with open(output_file, 'w', encoding='utf-8', errors='ignore') as md_file:
            md_file.write(initial_content)
            sorted_files = sorted(project_files)
            
            for i, file_path in enumerate(sorted_files):
                relative_path_str = str(file_path.relative_to(input_path_obj)).replace("\\", "/")
                self.log_callback(f"  + Including: {relative_path_str}")
                
                header = f"---\n\n### `{relative_path_str}`\n\n"
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    lang = file_path.suffix[1:].lower() if file_path.suffix else "text"
                    
                    # Apply intelligent sampling if enabled
                    if options.intelligent_sampling:
                        content = await self._apply_intelligent_sampling(
                            file_path, content, lang, options
                        )

                    if options.strip_comments:
                        content = self.content_filter.strip_comments(content, lang)
                    if options.optimize_whitespace:
                        content = self.content_filter.optimize_whitespace(content)

                    file_block = f"```{lang}\n{content.strip()}\n```\n\n"
                    md_file.write(header + file_block)
                    total_content += header + file_block
                except Exception as e:
                    error_block = f"```\n[Could not read file content: {e}]\n```\n\n"
                    md_file.write(header + error_block)
                    total_content += header + error_block
                
                self.progress_callback(('update', i + 1))

        file_size_kb = output_file.stat().st_size / 1024
        
        # Use realistic token estimation
        token_estimates = TokenEstimator.get_all_estimates(total_content)
        # Use Claude as default for backward compatibility
        claude_estimate = token_estimates.get("claude", {}).get("claude-3.5-sonnet", len(total_content) // 4)
        
        # Perform token analysis
        self.log_callback("Analyzing token distribution...")
        analyzer = TokenAnalyzer(LLMProvider.CLAUDE, "claude-3.5-sonnet")
        file_infos, analysis_total_tokens = analyzer.analyze_files(project_files, input_path_obj)
        directory_analysis = analyzer.analyze_directories(file_infos, analysis_total_tokens)
        
        token_analysis = {
            'file_infos': file_infos,
            'directory_analysis': directory_analysis,
            'top_files': analyzer.get_top_files(file_infos, 20),
            'top_directories': analyzer.get_top_directories(directory_analysis, 15),
            'suggestions': analyzer.get_directory_suggestions(directory_analysis, 3.0)
        }
        
        stats = ProcessingStats(
            file_count=len(project_files),
            file_size_kb=file_size_kb,
            estimated_tokens=claude_estimate,
            token_estimates=token_estimates,
            token_analysis=token_analysis
        )
        
        return ProcessingResult(
            success=True,
            output_path=str(output_file),
            stats=stats
        )