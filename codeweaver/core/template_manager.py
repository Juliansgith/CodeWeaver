from pathlib import Path
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .json_utils import safe_json_dumps, convert_for_json

class ProjectType(Enum):
    """Supported project types for smart templates."""
    FRONTEND_REACT = "frontend_react"
    FRONTEND_VUE = "frontend_vue"
    FRONTEND_ANGULAR = "frontend_angular"
    FRONTEND_VANILLA = "frontend_vanilla"
    BACKEND_NODEJS = "backend_nodejs"
    BACKEND_PYTHON = "backend_python"
    BACKEND_JAVA = "backend_java"
    BACKEND_DOTNET = "backend_dotnet"
    BACKEND_GO = "backend_go"
    BACKEND_RUST = "backend_rust"
    MOBILE_REACT_NATIVE = "mobile_react_native"
    MOBILE_FLUTTER = "mobile_flutter"
    MOBILE_IONIC = "mobile_ionic"
    MOBILE_NATIVE_IOS = "mobile_native_ios"
    MOBILE_NATIVE_ANDROID = "mobile_native_android"
    FULLSTACK_MERN = "fullstack_mern"
    FULLSTACK_DJANGO = "fullstack_django"
    FULLSTACK_RAILS = "fullstack_rails"
    MICROSERVICES = "microservices"
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    DEVOPS = "devops"
    GAME_UNITY = "game_unity"
    GAME_UNREAL = "game_unreal"
    DESKTOP_ELECTRON = "desktop_electron"
    CLI_TOOL = "cli_tool"
    LIBRARY = "library"
    API_REST = "api_rest"
    API_GRAPHQL = "api_graphql"


@dataclass
class TemplateConfig:
    """Configuration for a smart template."""
    name: str
    description: str
    project_type: ProjectType
    ignore_patterns: List[str]
    priority_files: List[str]  # Files to always include
    entry_points: List[str]    # Main entry point files
    config_files: List[str]    # Configuration files
    test_patterns: List[str]   # Test file patterns
    build_artifacts: List[str] # Build/generated files to ignore
    documentation_files: List[str] # Documentation files
    tags: List[str]           # Tags for categorization
    file_type_weights: Dict[str, float]  # File extension importance weights
    directory_weights: Dict[str, float]  # Directory importance weights
    created_at: str = ""
    version: str = "1.0.0"
    author: str = "CodeWeaver"
    usage_stats: Dict[str, Any] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.usage_stats is None:
            self.usage_stats = {"usage_count": 0, "last_used": None, "feedback_score": 0.0}


class SmartTemplateLibrary:
    """Advanced template library with intelligent project detection and optimization."""

    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.templates_dir.mkdir(exist_ok=True)
        self.custom_templates_dir = templates_dir / "custom"
        self.custom_templates_dir.mkdir(exist_ok=True)
        
        # Initialize built-in templates
        self._initialize_builtin_templates()
        
        # Load existing templates
        self.templates: Dict[str, TemplateConfig] = {}
        self._load_all_templates()

    def _initialize_builtin_templates(self):
        """Initialize built-in smart templates for common project types."""
        builtin_templates = self._get_builtin_templates()
        
        for template_name, template_config in builtin_templates.items():
            template_path = self.templates_dir / f"{template_name}.json"
            if not template_path.exists():
                self.save_template(template_name, template_config)

    def _get_builtin_templates(self) -> Dict[str, TemplateConfig]:
        """Get all built-in template configurations."""
        return {
            "react_frontend": TemplateConfig(
                name="React Frontend",
                description="Modern React frontend application with TypeScript support",
                project_type=ProjectType.FRONTEND_REACT,
                ignore_patterns=[
                    "node_modules/**", "build/**", "dist/**", ".next/**",
                    "*.log", "*.cache", ".env.local", ".env.*.local",
                    "coverage/**", ".nyc_output/**", "*.tsbuildinfo",
                    "public/static/**", "storybook-static/**"
                ],
                priority_files=[
                    "package.json", "tsconfig.json", "src/App.tsx", "src/index.tsx",
                    "src/App.jsx", "src/index.jsx", "README.md"
                ],
                entry_points=["src/index.tsx", "src/index.jsx", "src/main.tsx", "src/main.jsx"],
                config_files=[
                    "package.json", "tsconfig.json", "webpack.config.js", "vite.config.ts",
                    ".eslintrc.*", ".prettierrc.*", "tailwind.config.*", "next.config.js"
                ],
                test_patterns=["**/*.test.*", "**/*.spec.*", "**/__tests__/**", "**/tests/**"],
                build_artifacts=["build/**", "dist/**", ".next/**", "out/**"],
                documentation_files=["README.md", "CHANGELOG.md", "docs/**", "*.md"],
                tags=["frontend", "react", "typescript", "javascript", "web"],
                file_type_weights={
                    ".tsx": 1.0, ".jsx": 1.0, ".ts": 0.9, ".js": 0.8,
                    ".css": 0.7, ".scss": 0.7, ".less": 0.7,
                    ".json": 0.6, ".md": 0.5
                },
                directory_weights={
                    "src": 1.0, "components": 0.9, "pages": 0.9, "hooks": 0.8,
                    "utils": 0.7, "styles": 0.6, "assets": 0.4, "public": 0.3
                }
            ),
            
            "python_backend": TemplateConfig(
                name="Python Backend",
                description="Python backend API with FastAPI/Django/Flask",
                project_type=ProjectType.BACKEND_PYTHON,
                ignore_patterns=[
                    "__pycache__/**", "*.pyc", "*.pyo", "*.pyd", ".Python",
                    "build/**", "develop-eggs/**", "dist/**", "downloads/**",
                    "eggs/**", ".eggs/**", "lib/**", "lib64/**", "parts/**",
                    "sdist/**", "var/**", "wheels/**", ".venv/**", "venv/**",
                    ".env", ".coverage", "htmlcov/**", ".pytest_cache/**",
                    "*.log", "celerybeat-schedule", "db.sqlite3", "media/**"
                ],
                priority_files=[
                    "requirements.txt", "setup.py", "pyproject.toml", "main.py",
                    "app.py", "manage.py", "wsgi.py", "asgi.py", "README.md"
                ],
                entry_points=["main.py", "app.py", "manage.py", "wsgi.py", "asgi.py"],
                config_files=[
                    "requirements.txt", "setup.py", "pyproject.toml", "tox.ini",
                    ".flake8", "mypy.ini", "pytest.ini", "Dockerfile", "docker-compose.yml"
                ],
                test_patterns=["test_*.py", "*_test.py", "tests/**", "**/test/**"],
                build_artifacts=["build/**", "dist/**", "*.egg-info/**"],
                documentation_files=["README.md", "CHANGELOG.md", "docs/**", "*.rst"],
                tags=["backend", "python", "api", "web", "server"],
                file_type_weights={
                    ".py": 1.0, ".pyx": 0.9, ".pyi": 0.8,
                    ".txt": 0.6, ".toml": 0.7, ".yml": 0.7, ".yaml": 0.7,
                    ".json": 0.6, ".md": 0.5, ".rst": 0.5
                },
                directory_weights={
                    "app": 1.0, "src": 1.0, "api": 0.9, "models": 0.9,
                    "views": 0.8, "serializers": 0.8, "utils": 0.7,
                    "migrations": 0.6, "static": 0.4, "templates": 0.5
                }
            ),
            
            "nodejs_backend": TemplateConfig(
                name="Node.js Backend",
                description="Node.js backend API with Express/NestJS",
                project_type=ProjectType.BACKEND_NODEJS,
                ignore_patterns=[
                    "node_modules/**", "npm-debug.log*", "yarn-debug.log*",
                    "yarn-error.log*", "lerna-debug.log*", ".pnpm-debug.log*",
                    "dist/**", "build/**", "coverage/**", ".nyc_output/**",
                    ".env", ".env.local", ".env.*.local", "*.tsbuildinfo",
                    "logs/**", "*.log", "pids/**", "*.pid", "*.seed", "*.pid.lock"
                ],
                priority_files=[
                    "package.json", "tsconfig.json", "server.js", "index.js",
                    "server.ts", "index.ts", "app.js", "app.ts", "README.md"
                ],
                entry_points=["index.js", "server.js", "app.js", "index.ts", "server.ts", "app.ts"],
                config_files=[
                    "package.json", "tsconfig.json", "jest.config.js", ".eslintrc.*",
                    ".prettierrc.*", "nodemon.json", "Dockerfile", "docker-compose.yml"
                ],
                test_patterns=["**/*.test.*", "**/*.spec.*", "test/**", "tests/**"],
                build_artifacts=["dist/**", "build/**", "lib/**"],
                documentation_files=["README.md", "CHANGELOG.md", "docs/**", "API.md"],
                tags=["backend", "nodejs", "typescript", "javascript", "api", "server"],
                file_type_weights={
                    ".ts": 1.0, ".js": 0.9, ".json": 0.7,
                    ".yml": 0.6, ".yaml": 0.6, ".md": 0.5
                },
                directory_weights={
                    "src": 1.0, "controllers": 0.9, "routes": 0.9, "models": 0.9,
                    "middleware": 0.8, "services": 0.8, "utils": 0.7,
                    "config": 0.7, "public": 0.4, "uploads": 0.3
                }
            ),
            
            "react_native_mobile": TemplateConfig(
                name="React Native Mobile",
                description="Cross-platform mobile app with React Native",
                project_type=ProjectType.MOBILE_REACT_NATIVE,
                ignore_patterns=[
                    "node_modules/**", "ios/build/**", "android/build/**",
                    "android/app/build/**", "*.log", ".metro-cache/**",
                    "ios/Pods/**", "ios/*.xcworkspace/**", "android/.gradle/**",
                    ".expo/**", "expo-env.d.ts", "web-build/**", "dist/**"
                ],
                priority_files=[
                    "package.json", "App.tsx", "App.jsx", "index.js",
                    "metro.config.js", "app.json", "eas.json", "README.md"
                ],
                entry_points=["index.js", "App.tsx", "App.jsx"],
                config_files=[
                    "package.json", "metro.config.js", "babel.config.js",
                    "app.json", "eas.json", "tsconfig.json", ".eslintrc.*"
                ],
                test_patterns=["**/*.test.*", "**/*.spec.*", "__tests__/**"],
                build_artifacts=["ios/build/**", "android/build/**", "web-build/**"],
                documentation_files=["README.md", "CHANGELOG.md", "docs/**"],
                tags=["mobile", "react-native", "cross-platform", "ios", "android"],
                file_type_weights={
                    ".tsx": 1.0, ".jsx": 1.0, ".ts": 0.9, ".js": 0.8,
                    ".json": 0.6, ".md": 0.5
                },
                directory_weights={
                    "src": 1.0, "components": 0.9, "screens": 0.9, "navigation": 0.8,
                    "services": 0.8, "utils": 0.7, "assets": 0.5, "ios": 0.6, "android": 0.6
                }
            ),
            
            "flutter_mobile": TemplateConfig(
                name="Flutter Mobile",
                description="Cross-platform mobile app with Flutter/Dart",
                project_type=ProjectType.MOBILE_FLUTTER,
                ignore_patterns=[
                    "build/**", ".dart_tool/**", ".packages", ".pub-cache/**",
                    ".pub/**", "ios/Flutter/flutter_assets/**", "ios/Pods/**",
                    "android/.gradle/**", "android/build/**", "*.log",
                    "*.iml", ".idea/**", ".vscode/**"
                ],
                priority_files=[
                    "pubspec.yaml", "lib/main.dart", "README.md",
                    "android/app/build.gradle", "ios/Runner/Info.plist"
                ],
                entry_points=["lib/main.dart"],
                config_files=[
                    "pubspec.yaml", "analysis_options.yaml", "android/app/build.gradle",
                    "ios/Runner/Info.plist", ".metadata"
                ],
                test_patterns=["test/**", "integration_test/**"],
                build_artifacts=["build/**", ".dart_tool/**"],
                documentation_files=["README.md", "CHANGELOG.md", "doc/**"],
                tags=["mobile", "flutter", "dart", "cross-platform", "ios", "android"],
                file_type_weights={
                    ".dart": 1.0, ".yaml": 0.7, ".yml": 0.7,
                    ".gradle": 0.6, ".plist": 0.6, ".md": 0.5
                },
                directory_weights={
                    "lib": 1.0, "test": 0.8, "integration_test": 0.7,
                    "android": 0.6, "ios": 0.6, "web": 0.5, "assets": 0.4
                }
            ),
            
            "microservices": TemplateConfig(
                name="Microservices Architecture",
                description="Distributed microservices with Docker and orchestration",
                project_type=ProjectType.MICROSERVICES,
                ignore_patterns=[
                    "node_modules/**", "__pycache__/**", "*.pyc", "build/**",
                    "dist/**", "target/**", "logs/**", "*.log",
                    ".env", ".env.local", "coverage/**", ".pytest_cache/**"
                ],
                priority_files=[
                    "docker-compose.yml", "docker-compose.yaml", "Dockerfile",
                    "kubernetes/**", "k8s/**", "helm/**", "README.md"
                ],
                entry_points=["docker-compose.yml", "main.py", "server.js", "app.py"],
                config_files=[
                    "docker-compose.yml", "Dockerfile", "kubernetes/**", "k8s/**",
                    "helm/**", ".env.example", "config/**", "*.yml", "*.yaml"
                ],
                test_patterns=["test/**", "tests/**", "**/*.test.*", "**/*.spec.*"],
                build_artifacts=["build/**", "dist/**", "target/**"],
                documentation_files=["README.md", "docs/**", "API.md", "ARCHITECTURE.md"],
                tags=["microservices", "docker", "kubernetes", "distributed", "api"],
                file_type_weights={
                    ".yml": 0.9, ".yaml": 0.9, ".py": 0.8, ".js": 0.8, ".ts": 0.8,
                    ".json": 0.7, ".md": 0.6
                },
                directory_weights={
                    "services": 1.0, "kubernetes": 0.9, "k8s": 0.9, "helm": 0.8,
                    "docker": 0.8, "config": 0.7, "docs": 0.6
                }
            )
        }

    def _load_all_templates(self):
        """Load all templates from disk."""
        # Load built-in templates
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert project_type string back to enum
                if 'project_type' in data and isinstance(data['project_type'], str):
                    try:
                        data['project_type'] = ProjectType(data['project_type'])
                    except ValueError:
                        data['project_type'] = ProjectType.LIBRARY  # Default fallback
                
                template_config = TemplateConfig(**data)
                self.templates[template_file.stem] = template_config
            except Exception as e:
                print(f"Warning: Failed to load template {template_file}: {e}")
        
        # Load custom templates
        for template_file in self.custom_templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert project_type string back to enum
                if 'project_type' in data and isinstance(data['project_type'], str):
                    try:
                        data['project_type'] = ProjectType(data['project_type'])
                    except ValueError:
                        data['project_type'] = ProjectType.LIBRARY  # Default fallback
                
                template_config = TemplateConfig(**data)
                self.templates[f"custom_{template_file.stem}"] = template_config
            except Exception as e:
                print(f"Warning: Failed to load custom template {template_file}: {e}")

    def get_template(self, name: str) -> Optional[TemplateConfig]:
        """Get a template by name."""
        return self.templates.get(name)

    def save_template(self, name: str, template_config: TemplateConfig, is_custom: bool = False):
        """Save a template to disk."""
        if is_custom:
            template_path = self.custom_templates_dir / f"{name}.json"
            template_key = f"custom_{name}"
        else:
            template_path = self.templates_dir / f"{name}.json"
            template_key = name
        
        template_data = convert_for_json(asdict(template_config))
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(safe_json_dumps(template_data, indent=2, ensure_ascii=False))
        
        # Update in-memory templates
        self.templates[template_key] = template_config

    def get_available_templates(self) -> List[Tuple[str, TemplateConfig]]:
        """Get all available templates with their configurations."""
        return [(name, config) for name, config in self.templates.items()]

    def get_templates_by_type(self, project_type: ProjectType) -> List[Tuple[str, TemplateConfig]]:
        """Get templates filtered by project type."""
        return [(name, config) for name, config in self.templates.items()
                if config.project_type == project_type]

    def get_templates_by_tags(self, tags: List[str]) -> List[Tuple[str, TemplateConfig]]:
        """Get templates that match any of the provided tags."""
        matching_templates = []
        for name, config in self.templates.items():
            if any(tag in config.tags for tag in tags):
                matching_templates.append((name, config))
        return matching_templates

    def detect_project_type(self, project_path: Path) -> List[Tuple[ProjectType, float]]:
        """Intelligently detect project type based on files and structure."""
        detection_scores = {}
        
        # Get all files in project
        all_files = []
        for pattern in ['**/*', '**/.*']:
            try:
                all_files.extend(list(project_path.glob(pattern)))
            except:
                continue
        
        file_paths = [f for f in all_files if f.is_file()]
        file_names = [f.name.lower() for f in file_paths]
        extensions = [f.suffix.lower() for f in file_paths if f.suffix]
        
        # Detection patterns for each project type
        detection_patterns = {
            ProjectType.FRONTEND_REACT: {
                'files': ['package.json', 'src/app.tsx', 'src/app.jsx', 'public/index.html'],
                'dependencies': ['react', 'react-dom'],
                'extensions': ['.tsx', '.jsx'],
                'directories': ['src', 'public', 'components']
            },
            ProjectType.BACKEND_PYTHON: {
                'files': ['requirements.txt', 'main.py', 'app.py', 'manage.py', 'wsgi.py'],
                'dependencies': ['django', 'flask', 'fastapi'],
                'extensions': ['.py'],
                'directories': ['src', 'app', 'models', 'views']
            },
            ProjectType.BACKEND_NODEJS: {
                'files': ['package.json', 'server.js', 'index.js', 'app.js'],
                'dependencies': ['express', 'nestjs', 'koa'],
                'extensions': ['.js', '.ts'],
                'directories': ['src', 'routes', 'controllers', 'middleware']
            },
            ProjectType.MOBILE_REACT_NATIVE: {
                'files': ['package.json', 'app.tsx', 'app.jsx', 'metro.config.js'],
                'dependencies': ['react-native', '@react-native'],
                'extensions': ['.tsx', '.jsx'],
                'directories': ['ios', 'android', 'src', 'components']
            },
            ProjectType.MOBILE_FLUTTER: {
                'files': ['pubspec.yaml', 'lib/main.dart'],
                'dependencies': ['flutter'],
                'extensions': ['.dart'],
                'directories': ['lib', 'android', 'ios', 'test']
            },
            ProjectType.MICROSERVICES: {
                'files': ['docker-compose.yml', 'dockerfile', 'kubernetes'],
                'dependencies': [],
                'extensions': ['.yml', '.yaml'],
                'directories': ['services', 'kubernetes', 'k8s', 'docker']
            }
        }
        
        for project_type, patterns in detection_patterns.items():
            score = 0.0
            
            # Check for key files
            for file_pattern in patterns['files']:
                if any(file_pattern.lower() in name for name in file_names):
                    score += 2.0
            
            # Check for extensions
            for ext in patterns['extensions']:
                if ext in extensions:
                    score += 1.0 * extensions.count(ext) / len(file_paths)
            
            # Check for directories
            dir_names = [d.name.lower() for d in project_path.iterdir() if d.is_dir()]
            for directory in patterns['directories']:
                if directory.lower() in dir_names:
                    score += 1.5
            
            # Check package.json or similar config files for dependencies
            if patterns['dependencies']:
                config_files = [project_path / 'package.json', project_path / 'requirements.txt', 
                              project_path / 'pubspec.yaml', project_path / 'pom.xml']
                for config_file in config_files:
                    if config_file.exists():
                        try:
                            content = config_file.read_text(encoding='utf-8').lower()
                            for dep in patterns['dependencies']:
                                if dep.lower() in content:
                                    score += 1.5
                        except:
                            continue
            
            if score > 0:
                detection_scores[project_type] = score
        
        # Sort by score and return top matches
        sorted_scores = sorted(detection_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores

    def recommend_template(self, project_path: Path) -> Optional[Tuple[str, TemplateConfig, float]]:
        """Recommend the best template for a project."""
        detected_types = self.detect_project_type(project_path)
        
        if not detected_types:
            return None
        
        # Find templates matching the top detected type
        top_type, confidence = detected_types[0]
        matching_templates = self.get_templates_by_type(top_type)
        
        if not matching_templates:
            return None
        
        # Return the first matching template (could be enhanced with more logic)
        template_name, template_config = matching_templates[0]
        return template_name, template_config, confidence

    def create_template_from_project(self, name: str, project_path: Path, 
                                   description: str = "", is_custom: bool = True) -> TemplateConfig:
        """Create a new template from analyzing an existing project."""
        detected_types = self.detect_project_type(project_path)
        project_type = detected_types[0][0] if detected_types else ProjectType.LIBRARY
        
        # Analyze project structure
        ignore_patterns = self._analyze_ignore_patterns(project_path)
        priority_files = self._analyze_priority_files(project_path)
        entry_points = self._analyze_entry_points(project_path)
        config_files = self._analyze_config_files(project_path)
        test_patterns = self._analyze_test_patterns(project_path)
        
        template_config = TemplateConfig(
            name=name,
            description=description or f"Custom template for {name} projects",
            project_type=project_type,
            ignore_patterns=ignore_patterns,
            priority_files=priority_files,
            entry_points=entry_points,
            config_files=config_files,
            test_patterns=test_patterns,
            build_artifacts=["build/**", "dist/**", "target/**"],
            documentation_files=["README.md", "docs/**", "*.md"],
            tags=["custom", project_type.value],
            file_type_weights=self._analyze_file_weights(project_path),
            directory_weights=self._analyze_directory_weights(project_path)
        )
        
        self.save_template(name, template_config, is_custom=is_custom)
        return template_config

    def _analyze_ignore_patterns(self, project_path: Path) -> List[str]:
        """Analyze project to suggest ignore patterns."""
        patterns = set()
        
        # Check existing ignore files
        ignore_files = ['.gitignore', '.dockerignore', '.eslintignore']
        for ignore_file in ignore_files:
            ignore_path = project_path / ignore_file
            if ignore_path.exists():
                try:
                    content = ignore_path.read_text(encoding='utf-8')
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            patterns.add(line)
                except:
                    continue
        
        # Add common patterns based on detected files
        if (project_path / 'node_modules').exists():
            patterns.add('node_modules/**')
        if (project_path / '__pycache__').exists():
            patterns.add('__pycache__/**')
        if (project_path / 'build').exists():
            patterns.add('build/**')
        if (project_path / 'dist').exists():
            patterns.add('dist/**')
        
        return list(patterns)

    def _analyze_priority_files(self, project_path: Path) -> List[str]:
        """Analyze project to identify priority files."""
        priority_files = []
        
        # Common important files
        important_files = [
            'README.md', 'package.json', 'requirements.txt', 'pubspec.yaml',
            'Cargo.toml', 'pom.xml', 'build.gradle', 'tsconfig.json',
            'main.py', 'app.py', 'index.js', 'server.js', 'main.dart'
        ]
        
        for file_name in important_files:
            if (project_path / file_name).exists():
                priority_files.append(file_name)
        
        return priority_files

    def _analyze_entry_points(self, project_path: Path) -> List[str]:
        """Analyze project to identify entry point files."""
        entry_points = []
        
        common_entries = [
            'main.py', 'app.py', 'index.js', 'server.js', 'main.dart',
            'src/index.js', 'src/main.js', 'src/app.js', 'lib/main.dart'
        ]
        
        for entry in common_entries:
            if (project_path / entry).exists():
                entry_points.append(entry)
        
        return entry_points

    def _analyze_config_files(self, project_path: Path) -> List[str]:
        """Analyze project to identify configuration files."""
        config_files = []
        
        config_patterns = [
            '*.json', '*.yml', '*.yaml', '*.toml', '*.ini', '*.cfg',
            'Dockerfile', 'docker-compose.*', '.*rc', '.*ignore'
        ]
        
        for pattern in config_patterns:
            try:
                matches = list(project_path.glob(pattern))
                config_files.extend([str(m.relative_to(project_path)) for m in matches if m.is_file()])
            except:
                continue
        
        return config_files

    def _analyze_test_patterns(self, project_path: Path) -> List[str]:
        """Analyze project to identify test file patterns."""
        patterns = set()
        
        # Look for existing test files
        test_patterns = ['**/test_*.py', '**/*_test.py', '**/tests/**', '**/test/**',
                        '**/*.test.js', '**/*.spec.js', '**/*.test.ts', '**/*.spec.ts']
        
        for pattern in test_patterns:
            try:
                matches = list(project_path.glob(pattern))
                if matches:
                    patterns.add(pattern)
            except:
                continue
        
        return list(patterns) if patterns else ['test/**', 'tests/**']

    def _analyze_file_weights(self, project_path: Path) -> Dict[str, float]:
        """Analyze project to determine file type importance weights."""
        weights = {}
        
        # Count extensions
        extensions = {}
        try:
            for file_path in project_path.rglob('*'):
                if file_path.is_file() and file_path.suffix:
                    ext = file_path.suffix.lower()
                    extensions[ext] = extensions.get(ext, 0) + 1
        except:
            pass
        
        # Assign weights based on frequency and importance
        total_files = sum(extensions.values()) if extensions else 1
        for ext, count in extensions.items():
            if ext in ['.py', '.js', '.ts', '.tsx', '.jsx', '.dart', '.java', '.go', '.rs']:
                weights[ext] = min(1.0, 0.5 + (count / total_files))
            elif ext in ['.json', '.yml', '.yaml', '.toml']:
                weights[ext] = 0.7
            elif ext in ['.md', '.txt', '.rst']:
                weights[ext] = 0.5
            else:
                weights[ext] = 0.4
        
        return weights

    def _analyze_directory_weights(self, project_path: Path) -> Dict[str, float]:
        """Analyze project to determine directory importance weights."""
        weights = {}
        
        # Standard directory importance
        important_dirs = {
            'src': 1.0, 'lib': 1.0, 'app': 1.0,
            'components': 0.9, 'pages': 0.9, 'views': 0.9,
            'models': 0.9, 'controllers': 0.9, 'services': 0.8,
            'utils': 0.7, 'helpers': 0.7, 'config': 0.7,
            'tests': 0.6, 'test': 0.6, 'docs': 0.5,
            'assets': 0.4, 'static': 0.4, 'public': 0.3
        }
        
        try:
            for dir_path in project_path.iterdir():
                if dir_path.is_dir():
                    dir_name = dir_path.name.lower()
                    if dir_name in important_dirs:
                        weights[dir_name] = important_dirs[dir_name]
                    else:
                        weights[dir_name] = 0.5
        except:
            pass
        
        return weights

    def update_template_usage(self, template_name: str, feedback_score: Optional[float] = None):
        """Update template usage statistics."""
        if template_name in self.templates:
            template = self.templates[template_name]
            template.usage_stats['usage_count'] += 1
            template.usage_stats['last_used'] = datetime.now().isoformat()
            
            if feedback_score is not None:
                current_score = template.usage_stats.get('feedback_score', 0.0)
                usage_count = template.usage_stats['usage_count']
                # Moving average of feedback scores
                template.usage_stats['feedback_score'] = ((current_score * (usage_count - 1)) + feedback_score) / usage_count
            
            # Save updated template
            is_custom = template_name.startswith('custom_')
            actual_name = template_name.replace('custom_', '') if is_custom else template_name
            self.save_template(actual_name, template, is_custom=is_custom)

    def get_popular_templates(self, limit: int = 10) -> List[Tuple[str, TemplateConfig]]:
        """Get most popular templates based on usage statistics."""
        templates_with_usage = [(name, config) for name, config in self.templates.items()]
        templates_with_usage.sort(key=lambda x: x[1].usage_stats['usage_count'], reverse=True)
        return templates_with_usage[:limit]

    def search_templates(self, query: str) -> List[Tuple[str, TemplateConfig, float]]:
        """Search templates by name, description, or tags."""
        results = []
        query_lower = query.lower()
        
        for name, config in self.templates.items():
            score = 0.0
            
            # Name match
            if query_lower in name.lower():
                score += 3.0
            
            # Description match
            if query_lower in config.description.lower():
                score += 2.0
            
            # Tags match
            for tag in config.tags:
                if query_lower in tag.lower():
                    score += 1.0
            
            if score > 0:
                results.append((name, config, score))
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        return results


# Backward compatibility
class TemplateManager(SmartTemplateLibrary):
    """Legacy template manager - now redirects to SmartTemplateLibrary."""
    
    def __init__(self, templates_dir: Path):
        super().__init__(templates_dir)
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Legacy method - returns template as dict."""
        template = super().get_template(name)
        return asdict(template) if template else None
    
    def save_template(self, name: str, template_data: Dict[str, Any]):
        """Legacy method - accepts dict input."""
        if isinstance(template_data, dict):
            # Convert old format to new format
            template_config = TemplateConfig(
                name=template_data.get('name', name),
                description=template_data.get('description', f'Template for {name} projects'),
                project_type=ProjectType.LIBRARY,  # Default
                ignore_patterns=template_data.get('ignore_patterns', []),
                priority_files=[],
                entry_points=[],
                config_files=[],
                test_patterns=[],
                build_artifacts=[],
                documentation_files=[],
                tags=['legacy'],
                file_type_weights={},
                directory_weights={}
            )
            super().save_template(name, template_config, is_custom=True)
        else:
            super().save_template(name, template_data)
    
    def get_available_templates(self) -> List[str]:
        """Legacy method - returns just template names."""
        return [name for name, _ in super().get_available_templates()]
