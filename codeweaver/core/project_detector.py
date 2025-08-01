from pathlib import Path
from typing import List, Dict, Any, Optional
import json

class ProjectDetector:
    """Detects project type and framework based on file structure and content."""

    def detect_project_type(self, project_root: Path) -> Optional[str]:
        """Detects the primary project type."""
        if (project_root / 'package.json').exists():
            return 'node'
        if (project_root / 'pom.xml').exists():
            return 'java_maven'
        if (project_root / 'build.gradle').exists():
            return 'java_gradle'
        if (project_root / 'requirements.txt').exists():
            return 'python'
        if (project_root / 'Gemfile').exists():
            return 'ruby'
        if (project_root / 'go.mod').exists():
            return 'go'
        if (project_root / 'Cargo.toml').exists():
            return 'rust'
        if any(f.suffix == '.csproj' for f in project_root.iterdir()):
            return 'dotnet'
        return None

    def detect_framework(self, project_root: Path, project_type: Optional[str]) -> Optional[str]:
        """Detects the framework used in the project."""
        if project_type == 'node':
            with open(project_root / 'package.json') as f:
                package_json = json.load(f)
                dependencies = package_json.get('dependencies', {})
                if 'react' in dependencies:
                    return 'react'
                if 'vue' in dependencies:
                    return 'vue'
                if '@angular/core' in dependencies:
                    return 'angular'
                if 'express' in dependencies:
                    return 'express'
        if project_type == 'python':
            with open(project_root / 'requirements.txt') as f:
                requirements = f.read()
                if 'django' in requirements:
                    return 'django'
                if 'flask' in requirements:
                    return 'flask'
        return None
