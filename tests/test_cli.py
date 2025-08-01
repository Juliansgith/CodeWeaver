import unittest
from unittest.mock import patch
from click.testing import CliRunner
from pathlib import Path

from codeweaver.cli.main import main as cli_main

class TestCli(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def test_cli_main_help(self):
        """Test that the main CLI entrypoint shows help text."""
        result = self.runner.invoke(cli_main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage: main [OPTIONS] COMMAND [ARGS]...", result.output)
        self.assertIn("Intelligent code packaging for AI systems.", result.output)
        # Check for command groups
        self.assertIn("chunked", result.output)
        self.assertIn("embedding", result.output)

    @patch('codeweaver.cli.main.OptimizationEngine')
    def test_digest_preview(self, mock_engine_cls):
        """Test the digest command in preview mode."""
        # Mock the AI engine to avoid heavy processing
        mock_engine_instance = mock_engine_cls.return_value
        
        # Mock the result of optimize_file_selection
        from codeweaver.ai.optimization_engine import OptimizationResult, FileRelevanceScore, BudgetAllocation
        from codeweaver.core.token_budget import BudgetStrategy
        
        # Create a fake project structure
        with self.runner.isolated_filesystem():
            project_dir = Path("test_project")
            project_dir.mkdir()
            (project_dir / "main.py").write_text("import utils\nprint('hello')")
            (project_dir / "utils.py").write_text("def helper(): pass")
            (project_dir / "README.md").write_text("# Test Project")

            # Setup mock return value
            mock_optimization_result = OptimizationResult(
                selected_files=[project_dir / "main.py", project_dir / "utils.py"],
                file_scores=[
                    FileRelevanceScore('test_project/main.py', 0.9, 0.8, 0.7, 0.6, 0.85, []),
                    FileRelevanceScore('test_project/utils.py', 0.8, 0.7, 0.6, 0.5, 0.75, []),
                ],
                budget_allocation=BudgetAllocation([], [], 10000, 500, 9500, 0, 0, BudgetStrategy.BALANCED),
                optimization_strategy="balanced",
                confidence_score=0.9,
                recommendations=[],
                execution_time=0.1
            )
            
            # Since optimize_file_selection is an async method, we need to mock its result
            # inside an async function that we can then run.
            async def mock_optimize(*args, **kwargs):
                return mock_optimization_result
            
            mock_engine_instance.optimize_file_selection = mock_optimize

            # Mock the CodebaseProcessor to return our fake files
            with patch('codeweaver.cli.main.CodebaseProcessor') as mock_processor_cls:
                from codeweaver.core.models import ProcessingResult
                mock_processor_instance = mock_processor_cls.return_value
                mock_processor_instance.process.return_value = ProcessingResult(
                    success=True,
                    files=[project_dir / "main.py", project_dir / "utils.py", project_dir / "README.md"]
                )

                result = self.runner.invoke(cli_main, [
                    'digest',
                    str(project_dir),
                    '--purpose', 'test purpose',
                    '--preview'
                ])

                self.assertEqual(result.exit_code, 0)
                self.assertIn("AI-Selected Files for Digest", result.output)
                self.assertIn("main.py", result.output)
                self.assertIn("utils.py", result.output)
                self.assertNotIn("README.md", result.output) # Assuming it wasn't selected by the mock

if __name__ == '__main__':
    unittest.main()