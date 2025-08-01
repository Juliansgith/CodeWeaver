import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from click.testing import CliRunner

from codeweaver.cli.main import main as cli_main
from codeweaver.cli.chunked_export import chunked_cli
from codeweaver.cli.embedding_config import embedding_cli
from codeweaver.core.models import ProcessingResult, ProcessingStats
from codeweaver.ai.optimization_engine import OptimizationResult, FileRelevanceScore
from codeweaver.core.token_budget import BudgetAllocation, BudgetStrategy
from codeweaver.export.formats import ExportOptions


class TestMainCLI:
    """Test the main CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_dir = self.temp_dir / "test_project"
        self.project_dir.mkdir()
        
        # Create test files
        (self.project_dir / "main.py").write_text("""
def main():
    print("Hello, World!")
    return "success"

if __name__ == "__main__":
    main()
""")
        
        (self.project_dir / "utils.py").write_text("""
def helper_function(x):
    return x * 2

def another_helper(y):
    return y + 1
""")
        
        (self.project_dir / "config.json").write_text('{
    "app_name": "test",
    "version": "1.0.0"
}')
    
    def teardown_method(self):
        """Clean up test resources."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_main_help(self):
        """Test main CLI help command."""
        result = self.runner.invoke(cli_main, ['--help'])
        
        assert result.exit_code == 0
        assert "CodeWeaver CLI" in result.output
        assert "Intelligent code packaging" in result.output
        assert "digest" in result.output
        assert "chunked" in result.output
        assert "embedding" in result.output
    
    def test_main_version(self):
        """Test version display."""
        result = self.runner.invoke(cli_main, ['--version'])
        
        assert result.exit_code == 0
        # Should contain version information
        assert "version" in result.output.lower() or "." in result.output
    
    @patch('codeweaver.cli.main.OptimizationEngine')
    @patch('codeweaver.cli.main.CodebaseProcessor')
    @patch('codeweaver.cli.main.ExportManager')
    def test_digest_command_basic(self, mock_export_manager, mock_processor_cls, mock_engine_cls):
        """Test basic digest command functionality."""
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.process.return_value = ProcessingResult(
            success=True,
            files=[self.project_dir / "main.py", self.project_dir / "utils.py"]
        )
        mock_processor_cls.return_value = mock_processor
        
        # Mock optimization engine
        mock_engine = MagicMock()
        
        # Create mock optimization result
        optimization_result = OptimizationResult(
            selected_files=[self.project_dir / "main.py"],
            file_scores=[
                FileRelevanceScore(
                    file_path=str(self.project_dir / "main.py"),
                    base_importance=0.8,
                    semantic_relevance=0.7,
                    pattern_match=0.6,
                    dependency_relevance=0.5,
                    combined_score=0.75,
                    reasoning=["High importance file"]
                )
            ],
            budget_allocation=BudgetAllocation(
                selected_files=[], filtered_files=[], budget_used=5000,
                budget_remaining=45000, total_budget=50000, 
                over_budget_files=0, under_budget_threshold=0, strategy=BudgetStrategy.BALANCED
            ),
            optimization_strategy="test strategy",
            confidence_score=0.85,
            recommendations=["Good selection"],
            execution_time=0.5
        )
        
        # Set up async mock
        async def mock_optimize(*args, **kwargs):
            return optimization_result
        
        mock_engine.optimize_file_selection = mock_optimize
        mock_engine_cls.return_value = mock_engine
        
        # Mock export manager
        mock_export_manager_instance = MagicMock()
        mock_export_manager_instance.export_files.return_value = True
        mock_export_manager.return_value = mock_export_manager_instance
        
        # Run digest command
        result = self.runner.invoke(cli_main, [
            'digest',
            str(self.project_dir),
            '--purpose', 'test purpose',
            '--budget', '50000'
        ])
        
        assert result.exit_code == 0
        
        # Verify that the optimization engine was called
        mock_engine_cls.assert_called_once_with(self.project_dir)
        
        # Verify export manager was called
        mock_export_manager.assert_called_once()
        mock_export_manager_instance.export_files.assert_called_once()
    
    @patch('codeweaver.cli.main.OptimizationEngine')
    @patch('codeweaver.cli.main.CodebaseProcessor')
    def test_digest_preview_mode(self, mock_processor_cls, mock_engine_cls):
        """Test digest command in preview mode."""
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.process.return_value = ProcessingResult(
            success=True,
            files=[self.project_dir / "main.py", self.project_dir / "utils.py"]
        )
        mock_processor_cls.return_value = mock_processor
        
        # Mock optimization engine
        mock_engine = MagicMock()
        
        optimization_result = OptimizationResult(
            selected_files=[self.project_dir / "main.py"],
            file_scores=[
                FileRelevanceScore(
                    file_path=str(self.project_dir / "main.py"),
                    base_importance=0.8, semantic_relevance=0.7, pattern_match=0.6,
                    dependency_relevance=0.5, combined_score=0.75, reasoning=[]
                )
            ],
            budget_allocation=BudgetAllocation(
                selected_files=[], filtered_files=[], budget_used=5000,
                budget_remaining=45000, total_budget=50000, 
                over_budget_files=0, under_budget_threshold=0, strategy=BudgetStrategy.BALANCED
            ),
            optimization_strategy="test", confidence_score=0.85,
            recommendations=[], execution_time=0.1
        )
        
        async def mock_optimize(*args, **kwargs):
            return optimization_result
        
        mock_engine.optimize_file_selection = mock_optimize
        mock_engine_cls.return_value = mock_engine
        
        # Run preview command
        result = self.runner.invoke(cli_main, [
            'digest',
            str(self.project_dir),
            '--purpose', 'test purpose',
            '--preview'
        ])
        
        assert result.exit_code == 0
        assert "AI-Selected Files for Digest" in result.output
        assert "main.py" in result.output
        assert "Score:" in result.output
    
    def test_digest_no_files_error(self):
        """Test digest command when no files are found."""
        empty_dir = self.temp_dir / "empty_project"
        empty_dir.mkdir()
        
        with patch('codeweaver.cli.main.CodebaseProcessor') as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process.return_value = ProcessingResult(
                success=False,
                message="No files found"
            )
            mock_processor_cls.return_value = mock_processor
            
            result = self.runner.invoke(cli_main, [
                'digest',
                str(empty_dir)
            ])
            
            assert result.exit_code == 1
    
    def test_digest_with_exclude_patterns(self):
        """Test digest command with exclude patterns."""
        # Create additional files to exclude
        (self.project_dir / "test_file.pyc").write_text("compiled")
        (self.project_dir / "debug.log").write_text("log content")
        
        with patch('codeweaver.cli.main.OptimizationEngine') as mock_engine_cls, \
             patch('codeweaver.cli.main.CodebaseProcessor') as mock_processor_cls, \
             patch('codeweaver.cli.main.ExportManager') as mock_export_manager:
            
            # Set up mocks
            mock_processor = MagicMock()
            mock_processor.process.return_value = ProcessingResult(
                success=True,
                files=[self.project_dir / "main.py"]  # Should exclude .pyc and .log files
            )
            mock_processor_cls.return_value = mock_processor
            
            mock_engine = MagicMock()
            optimization_result = OptimizationResult(
                selected_files=[self.project_dir / "main.py"],
                file_scores=[], budget_allocation=MagicMock(),
                optimization_strategy="test", confidence_score=0.8,
                recommendations=[], execution_time=0.1
            )
            
            async def mock_optimize(*args, **kwargs):
                return optimization_result
            
            mock_engine.optimize_file_selection = mock_optimize
            mock_engine_cls.return_value = mock_engine
            
            mock_export_manager_instance = MagicMock()
            mock_export_manager_instance.export_files.return_value = True
            mock_export_manager.return_value = mock_export_manager_instance
            
            result = self.runner.invoke(cli_main, [
                'digest',
                str(self.project_dir),
                '--exclude', '*.pyc',
                '--exclude', '*.log',
                '--purpose', 'test'
            ])
            
            assert result.exit_code == 0
            
            # Check that exclude patterns were passed to processor
            call_args = mock_processor.process.call_args[0][0]
            assert '*.pyc' in call_args.ignore_patterns
            assert '*.log' in call_args.ignore_patterns
    
    def test_digest_format_options(self):
        """Test digest command with different output formats."""
        formats = ['markdown', 'json', 'xml', 'html', 'csv']
        
        for fmt in formats:
            with patch('codeweaver.cli.main.OptimizationEngine') as mock_engine_cls, \
                 patch('codeweaver.cli.main.CodebaseProcessor') as mock_processor_cls, \
                 patch('codeweaver.cli.main.ExportManager') as mock_export_manager:
                
                # Set up basic mocks
                mock_processor = MagicMock()
                mock_processor.process.return_value = ProcessingResult(
                    success=True, files=[self.project_dir / "main.py"]
                )
                mock_processor_cls.return_value = mock_processor
                
                mock_engine = MagicMock()
                optimization_result = OptimizationResult(
                    selected_files=[self.project_dir / "main.py"],
                    file_scores=[], budget_allocation=MagicMock(),
                    optimization_strategy="test", confidence_score=0.8,
                    recommendations=[], execution_time=0.1
                )
                
                async def mock_optimize(*args, **kwargs):
                    return optimization_result
                
                mock_engine.optimize_file_selection = mock_optimize
                mock_engine_cls.return_value = mock_engine
                
                mock_export_manager_instance = MagicMock()
                mock_export_manager_instance.export_files.return_value = True
                mock_export_manager.return_value = mock_export_manager_instance
                
                result = self.runner.invoke(cli_main, [
                    'digest',
                    str(self.project_dir),
                    '--format', fmt,
                    '--purpose', 'test'
                ])
                
                assert result.exit_code == 0
                
                # Verify export manager was called with correct format
                export_call = mock_export_manager_instance.export_files.call_args
                assert export_call[1]['format_name'] == fmt
    
    @patch('codeweaver.cli.main.CodeWeaverMCPServer')
    def test_mcp_server_command(self, mock_server_cls):
        """Test MCP server command."""
        mock_server = MagicMock()
        
        async def mock_main():
            pass
        
        mock_server.main = mock_main
        mock_server_cls.return_value = mock_server
        
        result = self.runner.invoke(cli_main, [
            'mcp-server',
            str(self.project_dir)
        ])
        
        assert result.exit_code == 0
        mock_server_cls.assert_called_once_with(root_path=self.project_dir)
    
    def test_mcp_server_with_port_error(self):
        """Test MCP server command with port (should error)."""
        result = self.runner.invoke(cli_main, [
            'mcp-server',
            str(self.project_dir),
            '--port', '8080'
        ])
        
        assert result.exit_code == 1
        assert "HTTP server not yet implemented" in result.output


class TestChunkedCLI:
    """Test the chunked export CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test files
        (self.input_dir / "file1.py").write_text("print('file1')")
        (self.input_dir / "file2.py").write_text("print('file2')")
    
    def teardown_method(self):
        """Clean up test resources."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_chunked_help(self):
        """Test chunked CLI help."""
        result = self.runner.invoke(chunked_cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Export large codebases in manageable chunks" in result.output
        assert "export" in result.output
    
    @patch('codeweaver.cli.chunked_export.create_chunked_export')
    def test_chunked_export_basic(self, mock_create_chunked):
        """Test basic chunked export command."""
        mock_create_chunked.return_value = True
        
        result = self.runner.invoke(chunked_cli, [
            'export',
            str(self.input_dir),
            str(self.output_dir)
        ])
        
        assert result.exit_code == 0
        mock_create_chunked.assert_called_once()
        
        # Check default parameters
        call_args = mock_create_chunked.call_args
        assert call_args[0][0] == Path(self.input_dir)
        assert call_args[0][1] == Path(self.output_dir)
    
    @patch('codeweaver.cli.chunked_export.create_chunked_export')
    def test_chunked_export_with_options(self, mock_create_chunked):
        """Test chunked export with various options."""
        mock_create_chunked.return_value = True
        
        result = self.runner.invoke(chunked_cli, [
            'export',
            str(self.input_dir),
            str(self.output_dir),
            '--format', 'json',
            '--strategy', 'by_size',
            '--max-tokens', '25000',
            '--max-files', '25',
            '--no-cross-refs',
            '--ignore', '*.pyc',
            '--ignore', '*.log',
            '--base-name', 'custom_chunk'
        ])
        
        assert result.exit_code == 0
        
        # Verify parameters were passed correctly
        call_args = mock_create_chunked.call_args
        config = call_args[1]['config']
        
        assert config.max_tokens_per_chunk == 25000
        assert config.max_files_per_chunk == 25
        assert config.generate_cross_references is False
        assert 'custom_chunk' in str(call_args)
    
    @patch('codeweaver.cli.chunked_export.create_chunked_export')
    def test_chunked_export_failure(self, mock_create_chunked):
        """Test chunked export command failure."""
        mock_create_chunked.return_value = False
        
        result = self.runner.invoke(chunked_cli, [
            'export',
            str(self.input_dir),
            str(self.output_dir)
        ])
        
        assert result.exit_code == 1
        assert "Failed to create chunked export" in result.output
    
    def test_chunked_export_strategies(self):
        """Test all supported chunking strategies."""
        strategies = ['by_size', 'by_count', 'by_directory', 'by_importance', 'by_type', 'balanced']
        
        for strategy in strategies:
            with patch('codeweaver.cli.chunked_export.create_chunked_export') as mock_create:
                mock_create.return_value = True
                
                result = self.runner.invoke(chunked_cli, [
                    'export',
                    str(self.input_dir),
                    str(self.output_dir),
                    '--strategy', strategy
                ])
                
                assert result.exit_code == 0
                
                # Verify strategy was set
                call_args = mock_create.call_args
                config = call_args[1]['config']
                assert config.chunk_strategy.value == strategy


class TestEmbeddingCLI:
    """Test the embedding configuration CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_embedding_help(self):
        """Test embedding CLI help."""
        result = self.runner.invoke(embedding_cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Configure embedding providers" in result.output
        assert "set-key" in result.output
    
    @patch('codeweaver.cli.embedding_config.get_embedding_config')
    @patch('codeweaver.cli.embedding_config.create_embedding_service')
    def test_set_key_success(self, mock_create_service, mock_get_config):
        """Test successful API key configuration."""
        # Mock config manager
        mock_config_manager = MagicMock()
        mock_get_config.return_value = mock_config_manager
        
        # Mock embedding service
        mock_service = MagicMock()
        
        # Mock the async test function
        async def mock_get_embeddings(texts):
            return [[0.1, 0.2, 0.3]]  # Return valid embedding
        
        mock_service.get_embeddings = mock_get_embeddings
        mock_create_service.return_value = mock_service
        
        result = self.runner.invoke(embedding_cli, [
            'set-key',
            'openai',
            'test-api-key-123'
        ])
        
        assert result.exit_code == 0
        assert "configured and verified successfully" in result.output
        
        # Verify config manager was called
        mock_config_manager.set_api_key.assert_called_once()
    
    @patch('codeweaver.cli.embedding_config.get_embedding_config')
    @patch('codeweaver.cli.embedding_config.create_embedding_service')
    def test_set_key_verification_failure(self, mock_create_service, mock_get_config):
        """Test API key configuration with verification failure."""
        # Mock config manager
        mock_config_manager = MagicMock()
        mock_get_config.return_value = mock_config_manager
        
        # Mock embedding service that fails verification
        mock_service = MagicMock()
        
        async def mock_get_embeddings(texts):
            return None  # Simulate API failure
        
        mock_service.get_embeddings = mock_get_embeddings
        mock_create_service.return_value = mock_service
        
        result = self.runner.invoke(embedding_cli, [
            'set-key',
            'gemini',
            'invalid-api-key'
        ])
        
        assert result.exit_code == 0  # Command succeeds but shows warning
        assert "configured but verification failed" in result.output
    
    @patch('codeweaver.cli.embedding_config.get_embedding_config')
    def test_set_key_configuration_error(self, mock_get_config):
        """Test API key configuration error."""
        # Mock config manager that raises an exception
        mock_config_manager = MagicMock()
        mock_config_manager.set_api_key.side_effect = Exception("Configuration error")
        mock_get_config.return_value = mock_config_manager
        
        result = self.runner.invoke(embedding_cli, [
            'set-key',
            'openai',
            'test-key'
        ])
        
        assert result.exit_code == 1
        assert "Failed to configure" in result.output
        assert "Configuration error" in result.output
    
    def test_set_key_invalid_provider(self):
        """Test set-key command with invalid provider."""
        result = self.runner.invoke(embedding_cli, [
            'set-key',
            'invalid-provider',
            'test-key'
        ])
        
        assert result.exit_code == 2  # Click validation error
        assert "Invalid value" in result.output
    
    @patch('codeweaver.cli.embedding_config.get_embedding_config')
    @patch('codeweaver.cli.embedding_config.create_embedding_service')
    def test_set_key_providers(self, mock_create_service, mock_get_config):
        """Test set-key command with different providers."""
        providers = ['openai', 'gemini']
        
        for provider in providers:
            # Reset mocks
            mock_get_config.reset_mock()
            mock_create_service.reset_mock()
            
            # Mock config manager
            mock_config_manager = MagicMock()
            mock_get_config.return_value = mock_config_manager
            
            # Mock embedding service
            mock_service = MagicMock()
            
            async def mock_get_embeddings(texts):
                return [[0.1, 0.2, 0.3]]
            
            mock_service.get_embeddings = mock_get_embeddings
            mock_create_service.return_value = mock_service
            
            result = self.runner.invoke(embedding_cli, [
                'set-key',
                provider,
                f'{provider}-test-key'
            ])
            
            assert result.exit_code == 0
            assert f"{provider.title()} API key configured" in result.output


class TestCLIIntegration:
    """Integration tests for CLI components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_command_discovery(self):
        """Test that all commands are properly registered."""
        result = self.runner.invoke(cli_main, ['--help'])
        
        # Main commands
        assert "digest" in result.output
        assert "mcp-server" in result.output
        assert "chunked" in result.output
        assert "embedding" in result.output
    
    def test_subcommand_discovery(self):
        """Test that subcommands are properly registered."""
        # Test chunked subcommands
        result = self.runner.invoke(cli_main, ['chunked', '--help'])
        assert result.exit_code == 0
        assert "export" in result.output
        
        # Test embedding subcommands
        result = self.runner.invoke(cli_main, ['embedding', '--help'])
        assert result.exit_code == 0
        assert "set-key" in result.output
    
    def test_invalid_command(self):
        """Test behavior with invalid commands."""
        result = self.runner.invoke(cli_main, ['invalid-command'])
        assert result.exit_code != 0
        assert "No such command" in result.output
    
    def test_missing_arguments(self):
        """Test behavior with missing required arguments."""
        # Test digest without path
        result = self.runner.invoke(cli_main, ['digest'])
        assert result.exit_code != 0
        assert "Missing argument" in result.output
        
        # Test chunked export without arguments
        result = self.runner.invoke(cli_main, ['chunked', 'export'])
        assert result.exit_code != 0
        assert "Missing argument" in result.output
        
        # Test embedding set-key without arguments
        result = self.runner.invoke(cli_main, ['embedding', 'set-key'])
        assert result.exit_code != 0
        assert "Missing argument" in result.output
    
    def test_nonexistent_path(self):
        """Test behavior with nonexistent file paths."""
        result = self.runner.invoke(cli_main, [
            'digest',
            '/nonexistent/path'
        ])
        
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Path" in result.output
    
    def test_cli_logging_setup(self):
        """Test that CLI sets up logging correctly."""
        import logging
        
        # The CLI should set up basic logging configuration
        logger = logging.getLogger('codeweaver.cli.main')
        assert logger.level <= logging.INFO
    
    @patch('codeweaver.cli.main.asyncio.run')
    def test_async_handling(self, mock_asyncio_run):
        """Test that async operations are handled correctly."""
        # This tests that the CLI properly handles async calls
        with patch('codeweaver.cli.main.OptimizationEngine') as mock_engine_cls, \
             patch('codeweaver.cli.main.CodebaseProcessor') as mock_processor_cls, \
             patch('codeweaver.cli.main.ExportManager') as mock_export_manager:
            
            # Set up mocks
            mock_processor = MagicMock()
            mock_processor.process.return_value = ProcessingResult(
                success=True, files=[Path("test.py")]
            )
            mock_processor_cls.return_value = mock_processor
            
            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine
            
            mock_export_manager_instance = MagicMock()
            mock_export_manager_instance.export_files.return_value = True
            mock_export_manager.return_value = mock_export_manager_instance
            
            # Create a temporary project
            with self.runner.isolated_filesystem():
                Path("test_project").mkdir()
                Path("test_project/main.py").write_text("print('test')")
                
                result = self.runner.invoke(cli_main, [
                    'digest',
                    'test_project',
                    '--purpose', 'test'
                ])
                
                # Verify asyncio.run was called (for the optimization)
                mock_asyncio_run.assert_called()
