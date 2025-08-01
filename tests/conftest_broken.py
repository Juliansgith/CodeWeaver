import pytest
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock

from codeweaver.ai.embeddings import OpenAIEmbeddingService, EmbeddingCache
from codeweaver.core.models import ProcessingOptions
from codeweaver.config.embedding_config import EmbeddingConfig, EmbeddingProvider

@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key from environment variables."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        pytest.skip("OpenAI API key not found in environment variables")
    return api_key

@pytest.fixture(scope="session")
def embedding_config(openai_api_key):
    """Create OpenAI embedding configuration."""
    return EmbeddingConfig(
        api_key=openai_api_key,
        model_name="text-embedding-3-small",
        embedding_dimensions=1536,
        max_tokens=8192,
        batch_size=100,
        rate_limit_per_minute=3000
    )

@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with sample files."""
    temp_dir = tempfile.mkdtemp()
    project_path = Path(temp_dir) / "test_project"
    project_path.mkdir()
    
    # Create sample Python files
    (project_path / "main.py").write_text("""
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
""")
    
    (project_path / "utils.py").write_text("""
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
""")
    
    (project_path / "models.py").write_text("""
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
""")
    
    (project_path / "config.json").write_text('''{
    "app_name": "Test Application",
    "version": "1.0.0",
    "debug": false,
    "database_url": "sqlite:///test.db"
}''')
    
    (project_path / "README.md").write_text("""
# Test Project

This is a test project for CodeWeaver testing.

## Features

- User management
- Email validation
- Data processing

## Usage

Run `python main.py` to start the application.
""")
    
    (project_path / "requirements.txt").write_text("pytest>=7.0.0\nrequests>=2.25.0\n")
    
    # Create subdirectories
    (project_path / "tests").mkdir()
    (project_path / "tests" / "test_main.py").write_text("""
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
""")
    
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
def embedding_service(embedding_config):
    """Create OpenAI embedding service instance."""
    # Use a temporary cache directory
    temp_cache = Path(tempfile.mkdtemp()) / "embeddings_cache"
    service = OpenAIEmbeddingService(embedding_config, temp_cache)
    yield service
    # Cleanup cache
    shutil.rmtree(temp_cache, ignore_errors=True)

@pytest.fixture
def mock_dependency_analyzer():
    """Create a mock dependency analyzer for testing."""
    mock = MagicMock()
    mock.analyze_dependencies.return_value = MagicMock(
        centrality_scores={"main.py": 0.8, "utils.py": 0.6, "models.py": 0.7}
    )
    return mock
