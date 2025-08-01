#!/usr/bin/env python3
"""
CodeWeaver - Entry point for the application
"""

import sys
from pathlib import Path

# Add the project root to the Python path to allow running from the root directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from codeweaver.web.server import start_web_server

def main():
    """Launches the CodeWeaver Web UI."""
    print("Starting CodeWeaver Web Interface...")
    start_web_server()

if __name__ == "__main__":
    main()