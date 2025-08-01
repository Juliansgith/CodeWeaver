"""
Setup script for CodeWeaver - Intelligent Code Packaging for AI Systems
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "CodeWeaver - Intelligent Code Packaging for AI Systems"

# Read version from package
version = "1.0.0"
try:
    with open(Path(__file__).parent / "codeweaver" / "__version__.py", "r") as f:
        exec(f.read())
        version = __version__
except FileNotFoundError:
    pass

# Core requirements (always installed)
core_requirements = [
    "google-generativeai>=0.3.0",
    "openai>=1.0.0",
    "numpy>=1.21.0",
    "click>=8.0.0",
    "requests>=2.25.0",
    "PyYAML>=6.0",
    "python-dotenv>=1.0.0",
    "chardet>=4.0.0",
    "pygments>=2.10.0",
    "GitPython>=3.1.0",
    "psutil>=5.9.0",
]

# Optional feature requirements
extras_require = {
    # Web interface for visual template editor
    "web": [
        "aiohttp>=3.8.0",
        "aiohttp-cors>=0.7.0",
        "websockets>=10.0",
        "jinja2>=3.1.0",
        "aiofiles>=22.0.0",
    ],
    
    # Advanced export formats
    "export": [
        "reportlab>=3.6.0",
        "Pillow>=9.0.0",
        "markdown>=3.4.0",
        "beautifulsoup4>=4.11.0",
        "lxml>=4.6.0",
        "py7zr>=0.20.0",
    ],
    
    # Data analysis and visualization
    "analytics": [
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "textdistance>=4.5.0",
    ],
    
    # Advanced AI/ML features
    "ai": [
        "transformers>=4.21.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.13.0",
        "spacy>=3.4.0",
        "nltk>=3.7.0",
    ],
    
    # Cloud storage integration
    "cloud": [
        "boto3>=1.26.0",
        "google-cloud-storage>=2.5.0",
        "azure-storage-blob>=12.14.0",
    ],
    
    # Enhanced database support
    "database": [
        "sqlalchemy>=1.4.0",
        "redis>=4.0.0",
    ],
    
    # Performance monitoring
    "monitoring": [
        "sentry-sdk>=1.9.0",
        "structlog>=22.0.0",
        "colorlog>=6.7.0",
        "memory-profiler>=0.60.0",
    ],
    
    # Development tools
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0",
        "isort>=5.10.0",
        "bandit>=1.7.0",
        "pre-commit>=2.20.0",
        "sphinx>=5.0.0",
    ],
}

# Common combinations
extras_require["all"] = list(set(sum(extras_require.values(), [])))
extras_require["full"] = extras_require["web"] + extras_require["export"] + extras_require["analytics"]

setup(
    name="codeweaver",
    version=version,
    author="CodeWeaver Development Team",
    author_email="contact@codeweaver.dev",
    description="Intelligent Code Packaging for AI Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/codeweaver",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/codeweaver/issues",
        "Documentation": "https://codeweaver.readthedocs.io/",
        "Source Code": "https://github.com/your-org/codeweaver",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Documentation",
        "Topic :: Text Processing :: Markup :: Markdown",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "codeweaver=codeweaver.cli.main:main",
            "cw=codeweaver.cli.main:main",  # Short alias
        ],
    },
    include_package_data=True,
    package_data={
        "codeweaver": [
            "templates/*.json",
            "gui/static/*",
            "config/*.yaml",
            "config/*.json",
        ],
    },
    keywords=[
        "code analysis",
        "ai",
        "machine learning",
        "documentation",
        "templates",
        "codebase",
        "packaging",
        "embeddings",
        "semantic search",
        "context selection",
    ],
    zip_safe=False,
)