import unittest
from pathlib import Path
from unittest.mock import MagicMock

from unittest.mock import patch
from codeweaver.core.importance_scorer import FileImportanceScorer, FileType

class TestFileImportanceScorer(unittest.TestCase):

    def setUp(self):
        # Mock the dependency analyzer
        self.mock_dependency_analyzer = MagicMock()
        self.scorer = FileImportanceScorer(self.mock_dependency_analyzer)
        self.project_root = Path("/fake/project")

    def test_classify_file_type(self):
        """Test the file type classification logic."""
        test_cases = {
            "main.py": FileType.ENTRY_POINT,
            "src/index.js": FileType.ENTRY_POINT,
            "app/utils.py": FileType.CORE_LIBRARY,
            "config.json": FileType.CONFIGURATION,
            "package.json": FileType.CONFIGURATION,
            "tests/test_api.py": FileType.TEST,
            "README.md": FileType.DOCUMENTATION,
            "dist/bundle.js": FileType.GENERATED,
            "assets/logo.png": FileType.ASSETS,
            "scripts/deploy.sh": FileType.BUILD_SCRIPT,
            "src/api/user_service.py": FileType.BUSINESS_LOGIC,
            "src/models/user.py": FileType.BUSINESS_LOGIC,
        }

        for path_str, expected_type in test_cases.items():
            with self.subTest(path=path_str):
                file_path = self.project_root / path_str
                # Mock stat().st_size to avoid FileNotFoundError and allow business logic check
                mock_path = MagicMock(spec=Path)
                mock_path.name = Path(path_str).name
                mock_path.parts = Path(path_str).parts
                mock_path.stat.return_value.st_size = 1000 
                
                # Mock file content for business logic check
                with patch("builtins.open", unittest.mock.mock_open(read_data="class UserService: pass")):
                    file_type = self.scorer._classify_file_type(mock_path)
                
                self.assertEqual(file_type, expected_type, f"Failed for path: {path_str}")

    def test_calculate_naming_bonus(self):
        """Test the bonus score for important naming conventions."""
        # High bonus for a core API file
        api_path = self.project_root / "src/api/auth_controller.py"
        bonus = self.scorer._calculate_naming_bonus(api_path)
        self.assertGreater(bonus, 10)

        # Lower bonus for a generic file
        generic_path = self.project_root / "src/components/button.js"
        bonus = self.scorer._calculate_naming_bonus(generic_path)
        self.assertEqual(bonus, 3.0) # Only for being in 'src'

        # No bonus for an unrelated file
        unrelated_path = self.project_root / "data/report.csv"
        bonus = self.scorer._calculate_naming_bonus(unrelated_path)
        self.assertEqual(bonus, 0)

if __name__ == '__main__':
    unittest.main()
