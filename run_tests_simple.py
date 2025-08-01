#!/usr/bin/env python3
"""
Simple CodeWeaver Test Runner

This script runs the comprehensive test suite for CodeWeaver.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if pytest is installed."""
    try:
        import pytest
        print("pytest is available")
        return True
    except ImportError:
        print("pytest not found. Install with: pip install pytest")
        return False

def check_api_keys():
    """Check if API keys are configured for integration tests."""
    openai_key = os.getenv('OPENAI_API_KEY')
    print(f"OpenAI API Key: {'Found' if openai_key else 'Not found'}")
    return bool(openai_key)

def run_basic_tests():
    """Run basic unit tests without integration."""
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Simple pytest command
    cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def main():
    """Main entry point."""
    print("CodeWeaver Test Suite")
    print("=" * 40)
    
    # Check basic requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check API keys (but don't require them for basic tests)
    has_api_keys = check_api_keys()
    
    if not has_api_keys:
        print("Note: Integration tests will be skipped (no API key)")
    
    # Run tests
    print("\nRunning tests...")
    success = run_basic_tests()
    
    if success:
        print("\nTests completed successfully!")
    else:
        print("\nSome tests failed or had errors. Check output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()