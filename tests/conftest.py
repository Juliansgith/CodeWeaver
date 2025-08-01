import pytest
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

# Only import what we can without causing import errors
try:
    from codeweaver.core.models import ProcessingOptions
except ImportError:
    # Create a mock if import fails
    class ProcessingOptions:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key from environment variables."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        pytest.skip("OpenAI API key not found in environment variables")
    return api_key

@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with sample files."""
    temp_dir = tempfile.mkdtemp()
    project_path = Path(temp_dir) / "test_project"
    project_path.mkdir()
    
    # Create sample Python files
    main_py_content = '''
import os
import sys
from utils import helper_function
from models import User

def main():
    """Main entry point for the application."""
    print("Starting application...")
    user = User("test@example.com")
    result = helper_function(user.email)
    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    main()
'''
    (project_path / "main.py").write_text(main_py_content)
    
    utils_py_content = '''
import re
from typing import Optional

def helper_function(email: str) -> str:
    """Process email and return formatted result."""
    if not is_valid_email(email):
        raise ValueError("Invalid email format")
    return f"Processed: {email.lower()}"

def is_valid_email(email: str) -> bool:
    """Check if email format is valid."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def calculate_metrics(data: list) -> dict:
    """Calculate basic metrics from data."""
    if not data:
        return {"count": 0, "sum": 0, "average": 0}
    
    return {
        "count": len(data),
        "sum": sum(data),
        "average": sum(data) / len(data)
    }
'''
    (project_path / "utils.py").write_text(utils_py_content)
    
    models_py_content = '''
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class User:
    """User model with email and creation timestamp."""
    email: str
    created_at: Optional[datetime] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def deactivate(self):
        """Deactivate the user account."""
        self.is_active = False
    
    def get_domain(self) -> str:
        """Extract domain from email address."""
        return self.email.split('@')[1] if '@' in self.email else ''

@dataclass
class Project:
    """Project model with name and description."""
    name: str
    description: str
    owner: User
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
'''
    (project_path / "models.py").write_text(models_py_content)
    
    config_json_content = '''{
    "app_name": "Test Application",
    "version": "1.0.0",
    "debug": false,
    "database_url": "sqlite:///test.db"
}'''
    (project_path / "config.json").write_text(config_json_content)
    
    readme_content = '''
# Test Project

This is a test project for CodeWeaver testing.

## Features

- User management
- Email validation
- Data processing

## Usage

Run `python main.py` to start the application.
'''
    (project_path / "README.md").write_text(readme_content)
    
    (project_path / "requirements.txt").write_text("pytest>=7.0.0\nrequests>=2.25.0\n")
    
    # Create subdirectories
    (project_path / "tests").mkdir()
    test_main_content = '''
import unittest
from main import main
from models import User

class TestMain(unittest.TestCase):
    def test_main_function(self):
        result = main()
        self.assertIsNotNone(result)
    
    def test_user_creation(self):
        user = User("test@example.com")
        self.assertEqual(user.email, "test@example.com")
        self.assertTrue(user.is_active)
'''
    (project_path / "tests" / "test_main.py").write_text(test_main_content)
    
    yield project_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_processing_options(temp_project_dir):
    """Create sample processing options for testing."""
    return ProcessingOptions(
        input_dir=str(temp_project_dir),
        ignore_patterns=["*.pyc", "__pycache__", ".git", "*.log"],
        size_limit_mb=10.0,
        mode="digest",
        strip_comments=False,
        optimize_whitespace=False,
        intelligent_sampling=False
    )

@pytest.fixture
def mock_dependency_analyzer():
    """Create a mock dependency analyzer for testing."""
    mock = MagicMock()
    mock.analyze_dependencies.return_value = MagicMock(
        centrality_scores={"main.py": 0.8, "utils.py": 0.6, "models.py": 0.7}
    )
    return mock