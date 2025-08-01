import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from codeweaver.core.models import (
    ProcessingStats, ProcessingResult, ProcessingOptions
)
from codeweaver.core.processor import (
    TreeGenerator, FileFilter, CodebaseProcessor
)
from codeweaver.core.tokenizer import TokenEstimator, LLMProvider
from codeweaver.core.analyzer import TokenAnalyzer


class TestProcessingModels:
    """Test the core data models."""
    
    def test_processing_stats_creation(self):
        """Test ProcessingStats dataclass creation and attributes."""
        stats = ProcessingStats(
            file_count=10,
            file_size_kb=512.5,
            estimated_tokens=15000,
            token_estimates={"claude": {"claude-3.5-sonnet": 15000}},
            token_analysis={"total_files": 10}
        )
        
        assert stats.file_count == 10
        assert stats.file_size_kb == 512.5
        assert stats.estimated_tokens == 15000
        assert stats.token_estimates["claude"]["claude-3.5-sonnet"] == 15000
        assert stats.token_analysis["total_files"] == 10
    
    def test_processing_result_success(self):
        """Test ProcessingResult for successful processing."""
        stats = ProcessingStats(file_count=5, file_size_kb=100.0, estimated_tokens=5000)
        result = ProcessingResult(
            success=True,
            output_path="/path/to/output.md",
            stats=stats,
            message="Processing completed successfully"
        )
        
        assert result.success is True
        assert result.output_path == "/path/to/output.md"
        assert result.stats.file_count == 5
        assert "successfully" in result.message
    
    def test_processing_result_failure(self):
        """Test ProcessingResult for failed processing."""
        result = ProcessingResult(
            success=False,
            message="Processing failed due to invalid input"
        )
        
        assert result.success is False
        assert result.output_path is None
        assert result.stats is None
        assert "failed" in result.message
    
    def test_processing_options_defaults(self):
        """Test ProcessingOptions with default values."""
        options = ProcessingOptions(
            input_dir="/test/path",
            ignore_patterns=["*.pyc"],
            size_limit_mb=5.0,
            mode="digest"
        )
        
        assert options.input_dir == "/test/path"
        assert options.ignore_patterns == ["*.pyc"]
        assert options.size_limit_mb == 5.0
        assert options.mode == "digest"
        assert options.strip_comments is False
        assert options.optimize_whitespace is False
        assert options.intelligent_sampling is False
        assert options.sampling_strategy == "balanced"
        assert options.sampling_max_lines == 300
        assert options.sampling_purpose is None
    
    def test_processing_options_with_sampling(self):
        """Test ProcessingOptions with sampling enabled."""
        options = ProcessingOptions(
            input_dir="/test/path",
            ignore_patterns=[],
            size_limit_mb=10.0,
            mode="digest",
            intelligent_sampling=True,
            sampling_strategy="semantic",
            sampling_max_lines=500,
            sampling_purpose="debug analysis"
        )
        
        assert options.intelligent_sampling is True
        assert options.sampling_strategy == "semantic"
        assert options.sampling_max_lines == 500
        assert options.sampling_purpose == "debug analysis"


class TestTreeGenerator:
    """Test the project tree generation functionality."""
    
    def test_generate_project_tree_simple(self, temp_project_dir):
        """Test tree generation with a simple project structure."""
        file_paths = [
            temp_project_dir / "main.py",
            temp_project_dir / "utils.py",
            temp_project_dir / "models.py",
            temp_project_dir / "README.md"
        ]
        
        tree = TreeGenerator.generate_project_tree(str(temp_project_dir), file_paths)
        
        # Check that the tree contains expected elements
        assert "test_project" in tree
        assert "main.py" in tree
        assert "utils.py" in tree
        assert "models.py" in tree
        assert "README.md" in tree
        
        # Check for tree structure symbols
        assert "├──" in tree or "└──" in tree
    
    def test_generate_project_tree_with_directories(self, temp_project_dir):
        """Test tree generation with nested directories."""
        file_paths = [
            temp_project_dir / "main.py",
            temp_project_dir / "tests" / "test_main.py"
        ]
        
        tree = TreeGenerator.generate_project_tree(str(temp_project_dir), file_paths)
        
        assert "tests" in tree
        assert "test_main.py" in tree
        assert "│" in tree or "└──" in tree  # Check for proper indentation
    
    def test_generate_project_tree_empty(self, temp_project_dir):
        """Test tree generation with no files."""
        tree = TreeGenerator.generate_project_tree(str(temp_project_dir), [])
        
        # Should only contain the project root directory name
        assert "test_project" in tree
        assert len(tree.strip().split('\n')) == 1


class TestFileFilter:
    """Test the file filtering functionality."""
    
    def test_file_filter_initialization(self):
        """Test FileFilter initialization."""
        ignore_patterns = ["*.pyc", "__pycache__", "*.log"]
        size_limit_mb = 5.0
        
        filter_obj = FileFilter(ignore_patterns, size_limit_mb)
        
        assert filter_obj.ignore_patterns == ignore_patterns
        assert filter_obj.size_limit_bytes == 5.0 * 1024 * 1024
    
    def test_should_ignore_directory(self, temp_project_dir):
        """Test directory ignoring logic."""
        filter_obj = FileFilter(["__pycache__", "*.git", "node_modules"], 10.0)
        
        # Create test directories
        pycache_dir = temp_project_dir / "__pycache__"
        pycache_dir.mkdir()
        normal_dir = temp_project_dir / "src"
        normal_dir.mkdir()
        
        assert filter_obj.should_ignore_directory(pycache_dir) is True
        assert filter_obj.should_ignore_directory(normal_dir) is False
    
    def test_should_ignore_file_by_pattern(self, temp_project_dir):
        """Test file ignoring by pattern."""
        filter_obj = FileFilter(["*.pyc", "*.log", "test_*"], 10.0)
        
        # Create test files
        pyc_file = temp_project_dir / "module.pyc"
        pyc_file.write_text("compiled python")
        
        log_file = temp_project_dir / "app.log"
        log_file.write_text("log content")
        
        normal_file = temp_project_dir / "main.py"
        normal_file.write_text("print('hello')")
        
        assert filter_obj.should_ignore_file(pyc_file) is True
        assert filter_obj.should_ignore_file(log_file) is True
        assert filter_obj.should_ignore_file(normal_file) is False
    
    def test_should_ignore_file_by_size(self, temp_project_dir):
        """Test file ignoring by size limit."""
        filter_obj = FileFilter([], 0.001)  # 1KB limit
        
        # Create small file
        small_file = temp_project_dir / "small.txt"
        small_file.write_text("small content")
        
        # Create large file
        large_file = temp_project_dir / "large.txt"
        large_file.write_text("x" * 2000)  # 2KB file
        
        assert filter_obj.should_ignore_file(small_file) is False
        assert filter_obj.should_ignore_file(large_file) is True


class TestCodebaseProcessor:
    """Test the main codebase processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.log_messages = []
        self.progress_updates = []
        
        def log_callback(message):
            self.log_messages.append(message)
        
        def progress_callback(update):
            self.progress_updates.append(update)
        
        self.processor = CodebaseProcessor(log_callback, progress_callback)
    
    def test_processor_initialization(self):
        """Test CodebaseProcessor initialization."""
        assert self.processor.log_callback is not None
        assert self.processor.progress_callback is not None
        assert self.processor.content_filter is not None
        assert self.processor.content_sampler is not None
    
    def test_collect_files(self, sample_processing_options):
        """Test file collection functionality."""
        files = self.processor._collect_files(sample_processing_options)
        
        # Should collect Python files, markdown, JSON, etc.
        file_names = [f.name for f in files]
        
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "models.py" in file_names
        assert "README.md" in file_names
        assert "config.json" in file_names
        
        # Should have logged progress
        assert len(self.log_messages) > 0
    
    def test_process_preview_mode(self, sample_processing_options):
        """Test processing in preview mode."""
        preview_options = ProcessingOptions(
            input_dir=sample_processing_options.input_dir,
            ignore_patterns=sample_processing_options.ignore_patterns,
            size_limit_mb=sample_processing_options.size_limit_mb,
            mode="preview"
        )
        
        result = self.processor.process(preview_options)
        
        assert result.success is True
        assert result.files is not None
        assert len(result.files) > 0
        assert result.output_path is None
        assert result.stats is None
    
    @pytest.mark.asyncio
    async def test_generate_digest(self, sample_processing_options, temp_project_dir):
        """Test digest generation functionality."""
        files = self.processor._collect_files(sample_processing_options)
        
        with patch('codeweaver.core.processor.TokenAnalyzer') as mock_analyzer_class:
            # Mock the token analyzer
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_files.return_value = ([], 10000)
            mock_analyzer.analyze_directories.return_value = {}
            mock_analyzer.get_top_files.return_value = []
            mock_analyzer.get_top_directories.return_value = []
            mock_analyzer.get_directory_suggestions.return_value = []
            mock_analyzer_class.return_value = mock_analyzer
            
            result = await self.processor._generate_digest(
                str(temp_project_dir), files, sample_processing_options
            )
        
        assert result.success is True
        assert result.output_path is not None
        assert result.stats is not None
        assert result.stats.file_count == len(files)
        assert result.stats.estimated_tokens > 0
        
        # Check that the output file was created
        output_path = Path(result.output_path)
        assert output_path.exists()
        
        # Check content structure
        content = output_path.read_text(encoding='utf-8')
        assert "# Project Structure" in content
        assert "# File Contents" in content
        assert "main.py" in content
        assert "def main():" in content
    
    @pytest.mark.asyncio
    async def test_apply_intelligent_sampling(self, temp_project_dir):
        """Test intelligent sampling functionality."""
        # Create a large file for sampling
        large_file = temp_project_dir / "large_file.py"
        large_content = "\n".join([f"# Line {i}\nprint(f'This is line {i}')" for i in range(200)])
        large_file.write_text(large_content)
        
        options = ProcessingOptions(
            input_dir=str(temp_project_dir),
            ignore_patterns=[],
            size_limit_mb=10.0,
            mode="digest",
            intelligent_sampling=True,
            sampling_max_lines=50
        )
        
        # Mock the content sampler to avoid complex sampling logic
        with patch.object(self.processor.content_sampler, 'sample_file') as mock_sample:
            from codeweaver.core.content_sampler import FileSample, SampleSection, SamplingStrategy
            
            # Create a mock sample result
            mock_sample_result = FileSample(
                file_path=large_file,
                strategy_used=SamplingStrategy.HEAD_TAIL,
                original_size=200,
                sampled_size=50,
                reduction_ratio=0.25,
                sections=[
                    SampleSection(
                        content="# Line 0\nprint(f'This is line 0')",
                        start_line=1,
                        end_line=2,
                        section_type="head",
                        reasoning="Important beginning section"
                    )
                ]
            )
            mock_sample.return_value = mock_sample_result
            
            sampled_content = await self.processor._apply_intelligent_sampling(
                large_file, large_content, "python", options
            )
            
            assert len(sampled_content) < len(large_content)
            assert "File sampled:" in sampled_content
            assert "Strategy:" in sampled_content
    
    def test_process_error_handling(self, temp_project_dir):
        """Test error handling in processing."""
        # Create invalid options that will cause an error
        invalid_options = ProcessingOptions(
            input_dir="/nonexistent/path",
            ignore_patterns=[],
            size_limit_mb=10.0,
            mode="digest"
        )
        
        result = self.processor.process(invalid_options)
        
        assert result.success is False
        assert result.message is not None
        assert len(self.log_messages) > 0
        assert any("ERROR" in msg for msg in self.log_messages)


class TestTokenEstimator:
    """Test token estimation functionality."""
    
    def test_estimate_tokens_claude(self):
        """Test token estimation for Claude models."""
        text = "This is a sample text for token estimation. It contains multiple sentences and some code: def hello(): return 'world'"
        
        estimates = TokenEstimator.estimate_tokens(text, LLMProvider.CLAUDE)
        
        assert "claude-3.5-sonnet" in estimates
        assert "claude-3-haiku" in estimates
        assert all(isinstance(count, int) for count in estimates.values())
        assert all(count > 0 for count in estimates.values())
    
    def test_estimate_tokens_gpt(self):
        """Test token estimation for GPT models."""
        text = "def calculate(x, y): return x + y  # This is a simple function"
        
        estimates = TokenEstimator.estimate_tokens(text, LLMProvider.GPT)
        
        assert "gpt-4" in estimates
        assert "gpt-4-turbo" in estimates
        assert all(isinstance(count, int) for count in estimates.values())
        assert all(count > 0 for count in estimates.values())
    
    def test_estimate_tokens_gemini(self):
        """Test token estimation for Gemini models."""
        text = "import numpy as np\ndata = np.array([1, 2, 3, 4, 5])\nprint(f'Mean: {np.mean(data)}')"
        
        estimates = TokenEstimator.estimate_tokens(text, LLMProvider.GEMINI)
        
        assert "gemini-pro" in estimates
        assert "gemini-1.5-pro" in estimates
        assert all(isinstance(count, int) for count in estimates.values())
        assert all(count > 0 for count in estimates.values())
    
    def test_get_all_estimates(self):
        """Test getting estimates for all providers."""
        text = "Hello, world! This is a test."
        
        all_estimates = TokenEstimator.get_all_estimates(text)
        
        assert "claude" in all_estimates
        assert "gpt" in all_estimates
        assert "gemini" in all_estimates
        assert "llama" in all_estimates
        
        # Check that each provider has model estimates
        for provider_estimates in all_estimates.values():
            assert isinstance(provider_estimates, dict)
            assert len(provider_estimates) > 0
    
    def test_get_context_usage(self):
        """Test context usage calculation."""
        # Test with moderate token count
        usage_percent, status = TokenEstimator.get_context_usage(
            50000, LLMProvider.CLAUDE, "claude-3.5-sonnet"
        )
        
        assert 0 <= usage_percent <= 100
        assert status in ["OK", "Moderate", "High", "Near limit", "Exceeds limit"]
        assert usage_percent == 25.0  # 50000 / 200000 * 100
        assert status == "OK"
        
        # Test with high token count
        usage_percent, status = TokenEstimator.get_context_usage(
            180000, LLMProvider.CLAUDE, "claude-3.5-sonnet"
        )
        
        assert usage_percent == 90.0
        assert status == "Near limit"
    
    def test_code_ratio_estimation(self):
        """Test code ratio estimation helper function."""
        # Pure text
        text_content = "This is just regular text with no code elements."
        text_ratio = TokenEstimator._get_code_ratio(text_content)
        assert text_ratio < 0.2
        
        # Code-heavy content
        code_content = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

if __name__ == "__main__":
    result = calculate_sum([1, 2, 3, 4, 5])
    print(f"Sum: {result}")
"""
        code_ratio = TokenEstimator._get_code_ratio(code_content)
        assert code_ratio > 0.5
        
        # Mixed content
        mixed_content = "Here's some text followed by code: def hello(): return 'world'"
        mixed_ratio = TokenEstimator._get_code_ratio(mixed_content)
        assert 0.2 < mixed_ratio < 0.8
