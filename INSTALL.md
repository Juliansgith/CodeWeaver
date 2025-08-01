# CodeWeaver Installation Guide

This guide covers various installation methods and configurations for CodeWeaver.

## Quick Start

### Basic Installation (Core Features Only)
```bash
pip install -r requirements-core.txt
```

### Full Installation (All Features)
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
pip install -r requirements-dev.txt
```

## Installation Options

### 1. Minimal Installation (Core Features)
Perfect for basic code analysis and CLI usage:
```bash
pip install -r requirements-core.txt
```

**Includes:**
- Core CLI functionality
- Basic export formats (Markdown, JSON)
- Template system
- Embedding support (Gemini, OpenAI)
- Git integration

### 2. Web Interface Installation
Adds the visual template editor:
```bash
pip install -r requirements-web.txt
```

**Additional features:**
- Visual template editor with drag-and-drop
- Web-based GUI at localhost:8081
- Real-time WebSocket updates
- Project analysis interface

### 3. Advanced Export Installation
Adds support for advanced export formats:
```bash
pip install -r requirements-export.txt
```

**Additional features:**
- PDF export with syntax highlighting
- Enhanced HTML exports
- ZIP archives with metadata
- 7-Zip compression support
- Image processing for exports

### 4. Analytics Installation
Adds data analysis and visualization:
```bash
pip install -r requirements-analytics.txt
```

**Additional features:**
- Project statistics and metrics
- Code complexity analysis
- Interactive charts and graphs
- Data export capabilities
- Advanced similarity metrics

### 5. Development Installation
For contributors and developers:
```bash
pip install -r requirements-dev.txt
```

**Additional features:**
- Testing frameworks (pytest)
- Code quality tools (black, flake8, mypy)
- Documentation tools (sphinx)
- Performance profiling
- Pre-commit hooks

## Package Installation

### From PyPI (when published)
```bash
# Basic installation
pip install codeweaver

# With web interface
pip install codeweaver[web]

# With export capabilities
pip install codeweaver[export]

# With analytics
pip install codeweaver[analytics]

# Full installation
pip install codeweaver[full]

# Everything including development tools
pip install codeweaver[all]
```

### From Source
```bash
# Clone the repository
git clone https://github.com/your-org/codeweaver.git
cd codeweaver

# Install in development mode
pip install -e .

# Or install with specific features
pip install -e .[web,export,analytics]
```

## Environment Setup

### 1. Copy Environment Template
```bash
cp .env.example .env
```

### 2. Configure API Keys
Edit `.env` and add your API keys:

```bash
# Gemini API Key (get from https://makersuite.google.com/app/apikey)
CODEWEAVER_GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API Key (get from https://platform.openai.com/api-keys)
CODEWEAVER_OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Optional Configuration
Configure additional settings in `.env`:

```bash
# Default settings
CODEWEAVER_DEFAULT_TOKEN_BUDGET=50000
CODEWEAVER_DEFAULT_EXPORT_FORMAT=markdown

# Logging
CODEWEAVER_LOG_LEVEL=INFO
CODEWEAVER_DEBUG=false

# Web interface
CODEWEAVER_MCP_HOST=localhost
CODEWEAVER_MCP_PORT=8081
```

## Verify Installation

### 1. Test CLI
```bash
codeweaver --help
```

### 2. Test Templates
```bash
codeweaver template list
```

### 3. Test Embedding Configuration
```bash
codeweaver embedding status
```

### 4. Test Web Interface (if installed)
```bash
codeweaver template editor
```
Then open http://localhost:8081 in your browser.

## Feature-Specific Setup

### Embedding Services

#### Gemini Setup
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable:
   ```bash
   export CODEWEAVER_GEMINI_API_KEY="your-api-key"
   ```
3. Test configuration:
   ```bash
   codeweaver embedding test gemini
   ```

#### OpenAI Setup
1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set environment variable:
   ```bash
   export CODEWEAVER_OPENAI_API_KEY="your-api-key"
   ```
3. Test configuration:
   ```bash
   codeweaver embedding test openai
   ```

### PDF Export Setup
PDF export requires reportlab:
```bash
pip install reportlab>=3.6.0
```

### Advanced AI Features
For enhanced AI capabilities:
```bash
pip install transformers sentence-transformers torch
```

## Docker Installation

### Using Docker Compose
```yaml
version: '3.8'
services:
  codeweaver:
    build: .
    ports:
      - "8081:8081"
    volumes:
      - .:/workspace
      - ~/.codeweaver:/root/.codeweaver
    environment:
      - CODEWEAVER_GEMINI_API_KEY=${GEMINI_API_KEY}
      - CODEWEAVER_OPENAI_API_KEY=${OPENAI_API_KEY}
    command: codeweaver template editor --host 0.0.0.0
```

### Building Docker Image
```bash
docker build -t codeweaver .
docker run -p 8081:8081 -v $(pwd):/workspace codeweaver
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'aiohttp'
Install web dependencies:
```bash
pip install aiohttp aiohttp-cors websockets
```

#### PDF export not working
Install reportlab:
```bash
pip install reportlab Pillow
```

#### Template editor not starting
1. Check if port 8081 is available
2. Install web dependencies
3. Check firewall settings

#### Embedding tests failing
1. Verify API keys are set correctly
2. Check internet connection
3. Verify API key permissions

### Getting Help

1. **Documentation**: Check the full documentation
2. **Issues**: Report bugs on GitHub
3. **Discussions**: Ask questions in GitHub Discussions
4. **CLI Help**: Use `codeweaver --help` for command help

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 1GB free space
- **Network**: Internet connection for AI API calls

### Platform Support

- **Linux**: Full support
- **macOS**: Full support  
- **Windows**: Full support (use WSL for best experience)

## Performance Optimization

### For Large Codebases
```bash
# Increase token limits
export CODEWEAVER_DEFAULT_TOKEN_BUDGET=100000

# Enable performance logging
export CODEWEAVER_PERFORMANCE_LOGGING=true

# Use chunked exports
codeweaver chunked export ./large-project ./output --strategy balanced
```

### For Better Performance
- Use SSD storage
- Increase available RAM
- Configure appropriate token budgets
- Use appropriate chunking strategies for large projects