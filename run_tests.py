#!/usr/bin/env python3
"""
CodeWeaver Test Runner

This script runs the comprehensive test suite for CodeWeaver with proper
configuration and real API integration options.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if test requirements are installed."""
    required_packages = [
        'pytest', 'pytest-asyncio', 'pytest-cov', 'pytest-mock',
        'pytest-aiohttp', 'aioresponses'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required test packages: {', '.join(missing_packages)}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr install from requirements file:")
        print("pip install -r tests/requirements-test.txt")
        return False
    
    print("✅ All test requirements are installed")
    return True

def check_api_keys():
    """Check if API keys are configured for integration tests."""
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    
    print("\n🔑 API Key Status:")
    print(f"  OpenAI: {'✅ Configured' if openai_key else '❌ Not found'}")
    print(f"  Gemini: {'✅ Configured' if gemini_key else '❌ Not found'}")
    
    if not openai_key and not gemini_key:
        print("\n⚠️  No API keys found. Integration tests will be skipped.")
        print("To run integration tests, set one of:")
        print("  export OPENAI_API_KEY=your_openai_key")
        print("  export GEMINI_API_KEY=your_gemini_key")
        return False
    
    return True

def run_tests(test_type="all", integration=False, verbose=False):
    """Run the test suite."""
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test selection
    if test_type == "unit":
        cmd.extend(["-m", "not integration"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
        cmd.append("--run-integration")
    elif test_type == "core":
        cmd.append("tests/test_core_models.py")
    elif test_type == "ai":
        cmd.append("tests/test_ai_embeddings.py")
    elif test_type == "cli":
        cmd.append("tests/test_cli_comprehensive.py")
    elif test_type == "export":
        cmd.append("tests/test_export_functions.py")
    elif test_type == "web":
        cmd.append("tests/test_web_interface.py")
    elif test_type == "all":
        cmd.append("tests/")
    
    # Add integration flag if requested
    if integration:
        cmd.append("--run-integration")
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-q")
    
    # Add coverage
    cmd.extend(["--cov=codeweaver", "--cov-report=term-missing"])
    
    print(f"\n🧪 Running tests: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CodeWeaver tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "core", "ai", "cli", "export", "web"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--integration", 
        action="store_true",
        help="Include integration tests (requires API keys)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check requirements and API keys, don't run tests"
    )
    
    args = parser.parse_args()
    
    print("CodeWeaver Test Suite")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check API keys
    has_api_keys = check_api_keys()
    
    if args.check_only:
        print("\n✅ Environment check complete")
        return
    
    # Warn about integration tests
    if args.integration and not has_api_keys:
        print("\n❌ Cannot run integration tests without API keys")
        sys.exit(1)
    
    if args.type == "integration" and not has_api_keys:
        print("\n❌ Cannot run integration tests without API keys")
        sys.exit(1)
    
    # Run tests
    success = run_tests(
        test_type=args.type,
        integration=args.integration,
        verbose=args.verbose
    )
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
