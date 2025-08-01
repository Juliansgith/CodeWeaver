"""
Advanced cross-file relationship detection for automatic dependency inclusion.
Goes beyond simple imports to detect semantic and structural relationships.
"""

import asyncio
import re
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from enum import Enum

from .dependency_analyzer import DependencyGraphAnalyzer, ImportInfo, DependencyNode
from ..ai.embeddings import GeminiEmbeddingService, calculate_cosine_similarity


class RelationshipType(Enum):
    """Types of cross-file relationships."""
    IMPORT_DEPENDENCY = "import_dependency"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SHARED_INTERFACE = "shared_interface"
    DATA_FLOW = "data_flow"
    INHERITANCE = "inheritance"
    COMPOSITION = "composition"
    CONFIG_DEPENDENCY = "config_dependency"
    TEST_RELATIONSHIP = "test_relationship"
    DOCUMENTATION = "documentation"
    SHARED_CONSTANTS = "shared_constants"
    API_CLIENT_SERVER = "api_client_server"


@dataclass
class FileRelationship:
    """Represents a relationship between two files."""
    source_file: str
    target_file: str
    relationship_type: RelationshipType
    strength: float              # 0.0 to 1.0
    evidence: List[str]          # Human-readable evidence for the relationship
    metadata: Dict[str, Any]     # Additional relationship-specific data


@dataclass
class RelationshipGraph:
    """Complete relationship graph for a project."""
    relationships: List[FileRelationship]
    file_clusters: Dict[str, List[str]]  # Groups of related files
    relationship_matrix: Dict[Tuple[str, str], float]  # Overall relationship strength
    recommendations: List[str]   # Recommendations for file inclusion


class CrossFileRelationshipDetector:
    """
    Advanced detector for cross-file relationships beyond simple imports.
    """
    
    def __init__(self, root_path: Path, embedding_service: Optional[GeminiEmbeddingService] = None):
        self.root_path = root_path
        self.embedding_service = embedding_service
        self.dependency_analyzer = DependencyGraphAnalyzer(root_path)
        
        # Pattern libraries for different relationship types
        self.interface_patterns = self._build_interface_patterns()
        self.data_flow_patterns = self._build_data_flow_patterns()
        self.test_patterns = self._build_test_patterns()
        self.config_patterns = self._build_config_patterns()
        
        # Cached analysis results
        self._file_contents_cache: Dict[str, str] = {}
        self._ast_cache: Dict[str, ast.AST] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
    
    def _build_interface_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for detecting shared interfaces/contracts."""
        return {
            'python': [
                r'class\s+(\w+)\s*\([^)]*Protocol[^)]*\)',  # Protocol classes
                r'class\s+(\w+)\s*\([^)]*ABC[^)]*\)',       # Abstract base classes
                r'@abstractmethod',                          # Abstract methods
                r'def\s+(\w+)\s*\([^)]*\)\s*->\s*(\w+)',    # Type annotations
                r'(\w+)\s*:\s*TypeVar'                       # Type variables
            ],
            'typescript': [
                r'interface\s+(\w+)',                       # Interfaces
                r'type\s+(\w+)\s*=',                        # Type definitions
                r'abstract\s+class\s+(\w+)',               # Abstract classes
                r'export\s+interface\s+(\w+)',             # Exported interfaces
                r'extends\s+(\w+)',                         # Interface/class extension
            ],
            'java': [
                r'interface\s+(\w+)',                       # Interfaces
                r'abstract\s+class\s+(\w+)',               # Abstract classes
                r'implements\s+(\w+(?:,\s*\w+)*)',         # Interface implementation
                r'extends\s+(\w+)',                         # Class extension
                r'@FunctionalInterface'                     # Functional interfaces
            ],
            'csharp': [
                r'interface\s+I(\w+)',                      # Interfaces (I prefix)
                r'abstract\s+class\s+(\w+)',               # Abstract classes
                r':\s*I(\w+)',                              # Interface implementation
                r':\s*(\w+)(?:,|\s*{)',                     # Base class
                r'\[Serializable\]'                         # Serializable attribute
            ]
        }
    
    def _build_data_flow_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for detecting data flow relationships."""
        return {
            'python': [
                r'(\w+)\.save\(',                           # Database saves
                r'(\w+)\.load\(',                           # Database loads
                r'return\s+(\w+)\(',                        # Return function calls
                r'yield\s+(\w+)',                           # Generator yields
                r'json\.loads?\(',                          # JSON operations
                r'pickle\.loads?\(',                        # Pickle operations
                r'with\s+open\([^)]*[\'"]([^\'\"]+)[\'"]', # File operations
            ],
            'javascript': [
                r'fetch\([\'"]([^\'\"]+)[\'"]',              # API calls
                r'axios\.(?:get|post|put|delete)\(',        # Axios HTTP calls
                r'localStorage\.(?:get|set)Item\(',         # Local storage
                r'JSON\.(?:parse|stringify)\(',             # JSON operations
                r'require\([\'"]([^\'\"]+\.json)[\'"]',     # JSON requires
                r'import.*from\s+[\'"]([^\'\"]+\.json)[\'"]' # JSON imports
            ],
            'java': [
                r'@RequestMapping\([^)]*[\'"]([^\'\"]+)[\'"]', # Spring endpoints
                r'@GetMapping\([^)]*[\'"]([^\'\"]+)[\'"]',      # GET endpoints
                r'@PostMapping\([^)]*[\'"]([^\'\"]+)[\'"]',     # POST endpoints
                r'ObjectMapper\(\)\.readValue\(',              # JSON deserialization
                r'Jackson.*\.writeValueAsString\(',            # JSON serialization
            ]
        }
    
    def _build_test_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for detecting test relationships."""
        return {
            'python': [
                r'from\s+([^.\s]+)(?:\.[^.\s]+)*\s+import',  # Import under test
                r'import\s+([^.\s]+)(?:\.[^.\s]+)*',         # Module under test
                r'@pytest\.fixture.*def\s+(\w+)',           # Pytest fixtures
                r'def\s+test_(\w+)',                         # Test functions
                r'assert\s+(\w+)\.',                         # Assertions on objects
                r'mock\.patch\([\'"]([^\'\"]+)[\'"]'         # Mocked modules
            ],
            'javascript': [
                r'require\([\'"]([^\'\"]+)[\'"]',             # Required modules
                r'import.*from\s+[\'"]([^\'\"]+)[\'"]',       # ES6 imports
                r'describe\([\'"]([^\'\"]+)[\'"]',            # Test suites
                r'it\([\'"]([^\'\"]+)[\'"]',                  # Test cases
                r'jest\.mock\([\'"]([^\'\"]+)[\'"]',          # Jest mocks
                r'expect\((\w+)\)',                          # Expectations
            ],
            'java': [
                r'import\s+([^;]+);',                        # Imports
                r'@Test.*public\s+void\s+test(\w+)',         # JUnit tests
                r'@Mock.*(\w+)',                             # Mockito mocks
                r'verify\((\w+)\)',                          # Mock verifications
                r'when\((\w+)\)',                            # Mock stubbing
            ]
        }
    
    def _build_config_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for detecting configuration dependencies."""
        return {
            'general': [
                r'config\.[\'"]?(\w+)[\'"]?',                # Config access
                r'settings\.[\'"]?(\w+)[\'"]?',              # Settings access
                r'env\.[\'"]?(\w+)[\'"]?',                   # Environment variables
                r'process\.env\.[\'"]?(\w+)[\'"]?',          # Node.js env vars
                r'os\.environ\[[\'"](\w+)[\'"]\]',           # Python env vars
                r'System\.getenv\([\'"](\w+)[\'"]',          # Java env vars
            ]
        }
    
    async def analyze_relationships(self, files: List[Path]) -> RelationshipGraph:
        """Analyze all types of relationships between files."""
        logging.info(f"Analyzing relationships for {len(files)} files")
        
        # First, get import-based dependencies
        await self._analyze_import_dependencies(files)
        
        # Collect all relationships
        relationships = []
        
        # 1. Import dependencies (from dependency analyzer)
        import_relationships = await self._extract_import_relationships(files)
        relationships.extend(import_relationships)
        
        # 2. Semantic similarity (if embedding service available)
        if self.embedding_service:
            semantic_relationships = await self._analyze_semantic_relationships(files)
            relationships.extend(semantic_relationships)
        
        # 3. Shared interfaces and contracts
        interface_relationships = await self._analyze_interface_relationships(files)
        relationships.extend(interface_relationships)
        
        # 4. Data flow relationships
        data_flow_relationships = await self._analyze_data_flow_relationships(files)
        relationships.extend(data_flow_relationships)
        
        # 5. Test relationships
        test_relationships = await self._analyze_test_relationships(files)
        relationships.extend(test_relationships)
        
        # 6. Configuration dependencies
        config_relationships = await self._analyze_config_relationships(files)
        relationships.extend(config_relationships)
        
        # Build relationship matrix
        relationship_matrix = self._build_relationship_matrix(relationships)
        
        # Cluster related files
        file_clusters = self._cluster_related_files(relationships, files)
        
        # Generate recommendations
        recommendations = self._generate_inclusion_recommendations(
            relationships, file_clusters
        )
        
        return RelationshipGraph(
            relationships=relationships,
            file_clusters=file_clusters,
            relationship_matrix=relationship_matrix,
            recommendations=recommendations
        )
    
    async def _analyze_import_dependencies(self, files: List[Path]):
        """Analyze import-based dependencies using the dependency analyzer."""
        try:
            # Run the synchronous dependency analyzer in a thread
            result = await asyncio.to_thread(self.dependency_analyzer.analyze_dependencies, files)
            return result
        except Exception as e:
            logging.warning(f"Failed to analyze import dependencies: {e}")
            return None
    
    async def _extract_import_relationships(self, files: List[Path]) -> List[FileRelationship]:
        """Extract import-based relationships."""
        relationships = []
        
        # Use the dependency graph if available
        if hasattr(self.dependency_analyzer, 'dependency_graph'):
            for file_path, node in self.dependency_analyzer.dependency_graph.items():
                for imported_file in node.imports_from:
                    relationships.append(FileRelationship(
                        source_file=file_path,
                        target_file=imported_file,
                        relationship_type=RelationshipType.IMPORT_DEPENDENCY,
                        strength=1.0,  # Import relationships are strong
                        evidence=[f"Imports from {imported_file}"],
                        metadata={"import_count": 1}
                    ))
        
        return relationships
    
    async def _analyze_semantic_relationships(self, files: List[Path]) -> List[FileRelationship]:
        """Analyze semantic similarity between files using embeddings."""
        if not self.embedding_service:
            return []
        
        relationships = []
        
        try:
            # Get content and embeddings for all files
            file_embeddings = {}
            file_contents = {}
            
            for file_path in files:
                try:
                    content = await self._get_file_content(file_path)
                    file_contents[str(file_path)] = content
                    
                    # Get embedding for file content (sample first 2000 chars)
                    sample_content = content[:2000] if len(content) > 2000 else content
                    embeddings = await self.embedding_service.get_embeddings([sample_content])
                    
                    if embeddings:
                        file_embeddings[str(file_path)] = embeddings[0]
                except Exception as e:
                    logging.warning(f"Failed to get embedding for {file_path}: {e}")
                    continue
            
            # Calculate pairwise similarities
            file_paths = list(file_embeddings.keys())
            similarity_threshold = 0.7  # Minimum similarity for relationship
            
            for i, file1 in enumerate(file_paths):
                for j, file2 in enumerate(file_paths[i+1:], i+1):
                    if file1 == file2:
                        continue
                    
                    similarity = calculate_cosine_similarity(
                        file_embeddings[file1], 
                        file_embeddings[file2]
                    )
                    
                    if similarity >= similarity_threshold:
                        evidence = [f"High semantic similarity ({similarity:.2f})"]
                        
                        # Add more specific evidence
                        evidence.extend(self._find_semantic_evidence(
                            file_contents[file1], file_contents[file2]
                        ))
                        
                        relationships.append(FileRelationship(
                            source_file=file1,
                            target_file=file2,
                            relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                            strength=similarity,
                            evidence=evidence,
                            metadata={"cosine_similarity": similarity}
                        ))
            
        except Exception as e:
            logging.error(f"Failed to analyze semantic relationships: {e}")
        
        return relationships
    
    def _find_semantic_evidence(self, content1: str, content2: str) -> List[str]:
        """Find specific evidence for semantic similarity."""
        evidence = []
        
        # Find shared function/class names
        def extract_definitions(content: str) -> Set[str]:
            definitions = set()
            # Python function/class patterns
            for match in re.finditer(r'(?:def|class)\s+(\w+)', content):
                definitions.add(match.group(1))
            # JavaScript function patterns
            for match in re.finditer(r'function\s+(\w+)', content):
                definitions.add(match.group(1))
            # More patterns for other languages...
            return definitions
        
        defs1 = extract_definitions(content1)
        defs2 = extract_definitions(content2)
        shared_defs = defs1.intersection(defs2)
        
        if shared_defs:
            evidence.append(f"Shared definitions: {', '.join(list(shared_defs)[:5])}")
        
        # Find shared constants/variables
        def extract_constants(content: str) -> Set[str]:
            constants = set()
            # ALL_CAPS constants
            for match in re.finditer(r'\b([A-Z][A-Z0-9_]{2,})\b', content):
                constants.add(match.group(1))
            return constants
        
        constants1 = extract_constants(content1)
        constants2 = extract_constants(content2)
        shared_constants = constants1.intersection(constants2)
        
        if shared_constants:
            evidence.append(f"Shared constants: {', '.join(list(shared_constants)[:3])}")
        
        return evidence
    
    async def _analyze_interface_relationships(self, files: List[Path]) -> List[FileRelationship]:
        """Analyze shared interface and contract relationships."""
        relationships = []
        
        # Track interface definitions and implementations
        interfaces = defaultdict(list)  # interface_name -> [files that define it]
        implementations = defaultdict(list)  # interface_name -> [files that implement it]
        
        for file_path in files:
            try:
                content = await self._get_file_content(file_path)
                file_ext = file_path.suffix.lower()
                
                patterns = self.interface_patterns.get(
                    self._get_language_from_extension(file_ext), []
                )
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        interface_name = match.group(1) if match.groups() else match.group(0)
                        
                        if 'interface' in pattern or 'abstract' in pattern:
                            interfaces[interface_name].append(str(file_path))
                        elif 'implements' in pattern or 'extends' in pattern:
                            implementations[interface_name].append(str(file_path))
                
            except Exception as e:
                logging.warning(f"Failed to analyze interfaces in {file_path}: {e}")
                continue
        
        # Create relationships between interface definitions and implementations
        for interface_name, defining_files in interfaces.items():
            implementing_files = implementations.get(interface_name, [])
            
            for def_file in defining_files:
                for impl_file in implementing_files:
                    if def_file != impl_file:
                        relationships.append(FileRelationship(
                            source_file=impl_file,
                            target_file=def_file,
                            relationship_type=RelationshipType.SHARED_INTERFACE,
                            strength=0.8,
                            evidence=[f"Implements interface/contract: {interface_name}"],
                            metadata={"interface_name": interface_name}
                        ))
        
        return relationships
    
    async def _analyze_data_flow_relationships(self, files: List[Path]) -> List[FileRelationship]:
        """Analyze data flow relationships between files."""
        relationships = []
        
        # Track data operations (saves, loads, API calls, etc.)
        data_operations = defaultdict(list)  # operation_type -> [(file, details)]
        
        for file_path in files:
            try:
                content = await self._get_file_content(file_path)
                file_ext = file_path.suffix.lower()
                
                patterns = self.data_flow_patterns.get(
                    self._get_language_from_extension(file_ext), []
                )
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        operation_id = match.group(1) if match.groups() else "data_operation"
                        data_operations[operation_id].append((str(file_path), match.group(0)))
                
            except Exception as e:
                logging.warning(f"Failed to analyze data flow in {file_path}: {e}")
                continue
        
        # Create relationships between files that share data operations
        for operation_id, file_operations in data_operations.items():
            if len(file_operations) > 1:
                # Files that perform the same data operations are related
                for i, (file1, op1) in enumerate(file_operations):
                    for file2, op2 in file_operations[i+1:]:
                        relationships.append(FileRelationship(
                            source_file=file1,
                            target_file=file2,
                            relationship_type=RelationshipType.DATA_FLOW,
                            strength=0.6,
                            evidence=[f"Shared data operation: {operation_id}"],
                            metadata={"operation": operation_id, "patterns": [op1, op2]}
                        ))
        
        return relationships
    
    async def _analyze_test_relationships(self, files: List[Path]) -> List[FileRelationship]:
        """Analyze test-to-source file relationships."""
        relationships = []
        
        test_files = []
        source_files = []
        
        # Separate test files from source files
        for file_path in files:
            file_name = file_path.name.lower()
            if ('test' in file_name or 'spec' in file_name or 
                file_path.parent.name.lower() in ['test', 'tests', 'spec', 'specs']):
                test_files.append(file_path)
            else:
                source_files.append(file_path)
        
        # Analyze each test file to find what it tests
        for test_file in test_files:
            try:
                content = await self._get_file_content(test_file)
                file_ext = test_file.suffix.lower()
                
                patterns = self.test_patterns.get(
                    self._get_language_from_extension(file_ext), []
                )
                
                # Extract imported/referenced modules
                referenced_modules = set()
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        if match.groups():
                            referenced_modules.add(match.group(1))
                
                # Try to match with source files
                for source_file in source_files:
                    source_name = source_file.stem
                    source_path_parts = source_file.parts
                    
                    # Check if test file references this source file
                    relevance_score = 0.0
                    evidence = []
                    
                    # Direct name match
                    if source_name in referenced_modules:
                        relevance_score += 0.8
                        evidence.append(f"Directly imports {source_name}")
                    
                    # Path-based matching
                    for module in referenced_modules:
                        if source_name in module or module in source_name:
                            relevance_score += 0.6
                            evidence.append(f"References related module: {module}")
                    
                    # Test name pattern matching
                    test_name = test_file.stem
                    if source_name in test_name or test_name.replace('test_', '') == source_name:
                        relevance_score += 0.7
                        evidence.append(f"Test name suggests testing {source_name}")
                    
                    if relevance_score > 0.5:
                        relationships.append(FileRelationship(
                            source_file=str(test_file),
                            target_file=str(source_file),
                            relationship_type=RelationshipType.TEST_RELATIONSHIP,
                            strength=min(relevance_score, 1.0),
                            evidence=evidence,
                            metadata={"test_type": "unit_test"}
                        ))
                
            except Exception as e:
                logging.warning(f"Failed to analyze test relationships in {test_file}: {e}")
                continue
        
        return relationships
    
    async def _analyze_config_relationships(self, files: List[Path]) -> List[FileRelationship]:
        """Analyze configuration dependencies."""
        relationships = []
        
        # Find configuration files
        config_files = []
        source_files = []
        
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}
        config_names = {'config', 'settings', 'environment', 'env'}
        
        for file_path in files:
            is_config = (
                file_path.suffix.lower() in config_extensions or
                file_path.stem.lower() in config_names or
                'config' in file_path.name.lower()
            )
            
            if is_config:
                config_files.append(file_path)
            else:
                source_files.append(file_path)
        
        # Analyze source files for config dependencies
        for source_file in source_files:
            try:
                content = await self._get_file_content(source_file)
                
                config_references = set()
                for pattern in self.config_patterns['general']:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        if match.groups():
                            config_references.add(match.group(1))
                
                # Match with actual config files
                for config_file in config_files:
                    strength = 0.0
                    evidence = []
                    
                    # Check if config file is directly referenced
                    config_name = config_file.stem
                    if config_name.lower() in content.lower():
                        strength += 0.7
                        evidence.append(f"References config file: {config_name}")
                    
                    # Check if config values are referenced
                    if config_references:
                        # Try to load config file and check for matching keys
                        try:
                            config_content = await self._get_file_content(config_file)
                            if config_file.suffix.lower() == '.json':
                                config_data = json.loads(config_content)
                                config_keys = self._extract_json_keys(config_data)
                                
                                matching_keys = config_references.intersection(config_keys)
                                if matching_keys:
                                    strength += len(matching_keys) * 0.2
                                    evidence.append(f"Uses config keys: {', '.join(list(matching_keys)[:3])}")
                        except:
                            pass  # Config file might not be valid JSON
                    
                    if strength > 0.4:
                        relationships.append(FileRelationship(
                            source_file=str(source_file),
                            target_file=str(config_file),
                            relationship_type=RelationshipType.CONFIG_DEPENDENCY,
                            strength=min(strength, 1.0),
                            evidence=evidence,
                            metadata={"config_type": config_file.suffix}
                        ))
                
            except Exception as e:
                logging.warning(f"Failed to analyze config relationships in {source_file}: {e}")
                continue
        
        return relationships
    
    def _extract_json_keys(self, data: Any, prefix: str = "") -> Set[str]:
        """Recursively extract all keys from JSON data."""
        keys = set()
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.add(key)
                keys.add(full_key)
                keys.update(self._extract_json_keys(value, full_key))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                keys.update(self._extract_json_keys(item, prefix))
        
        return keys
    
    def _build_relationship_matrix(self, relationships: List[FileRelationship]) -> Dict[Tuple[str, str], float]:
        """Build a matrix of overall relationship strengths between files."""
        matrix = defaultdict(float)
        
        for rel in relationships:
            key = (rel.source_file, rel.target_file)
            # Combine multiple relationship types with weighted sum
            weight = {
                RelationshipType.IMPORT_DEPENDENCY: 1.0,
                RelationshipType.SEMANTIC_SIMILARITY: 0.8,
                RelationshipType.SHARED_INTERFACE: 0.9,
                RelationshipType.DATA_FLOW: 0.7,
                RelationshipType.TEST_RELATIONSHIP: 0.6,
                RelationshipType.CONFIG_DEPENDENCY: 0.5,
            }.get(rel.relationship_type, 0.5)
            
            matrix[key] += rel.strength * weight
        
        # Normalize values to 0-1 range
        if matrix:
            max_strength = max(matrix.values())
            for key in matrix:
                matrix[key] = min(matrix[key] / max_strength, 1.0) if max_strength > 0 else 0.0
        
        return dict(matrix)
    
    def _cluster_related_files(self, relationships: List[FileRelationship], 
                              files: List[Path]) -> Dict[str, List[str]]:
        """Group files into clusters based on their relationships."""
        # Use a simple clustering approach based on relationship strength
        clusters = {}
        file_to_cluster = {}
        cluster_id = 0
        
        # Sort relationships by strength
        sorted_relationships = sorted(relationships, key=lambda r: r.strength, reverse=True)
        
        for rel in sorted_relationships:
            source_cluster = file_to_cluster.get(rel.source_file)
            target_cluster = file_to_cluster.get(rel.target_file)
            
            if source_cluster is None and target_cluster is None:
                # Create new cluster
                cluster_name = f"cluster_{cluster_id}"
                clusters[cluster_name] = [rel.source_file, rel.target_file]
                file_to_cluster[rel.source_file] = cluster_name
                file_to_cluster[rel.target_file] = cluster_name
                cluster_id += 1
            elif source_cluster is None:
                # Add source to target's cluster
                clusters[target_cluster].append(rel.source_file)
                file_to_cluster[rel.source_file] = target_cluster
            elif target_cluster is None:
                # Add target to source's cluster
                clusters[source_cluster].append(rel.target_file)
                file_to_cluster[rel.target_file] = source_cluster
            elif source_cluster != target_cluster:
                # Merge clusters if relationship is strong enough
                if rel.strength > 0.7:
                    # Move all files from target cluster to source cluster
                    for file_path in clusters[target_cluster]:
                        clusters[source_cluster].append(file_path)
                        file_to_cluster[file_path] = source_cluster
                    del clusters[target_cluster]
        
        # Add isolated files to their own clusters
        all_clustered_files = set()
        for cluster_files in clusters.values():
            all_clustered_files.update(cluster_files)
        
        for file_path in files:
            if str(file_path) not in all_clustered_files:
                cluster_name = f"isolated_{file_path.stem}"
                clusters[cluster_name] = [str(file_path)]
        
        return clusters
    
    def _generate_inclusion_recommendations(self, relationships: List[FileRelationship],
                                          file_clusters: Dict[str, List[str]]) -> List[str]:
        """Generate recommendations for automatic file inclusion."""
        recommendations = []
        
        # Recommend including entire clusters for strongly related files
        for cluster_name, cluster_files in file_clusters.items():
            if len(cluster_files) > 1 and len(cluster_files) <= 5:  # Reasonable cluster size
                # Calculate average relationship strength within cluster
                cluster_strengths = []
                for i, file1 in enumerate(cluster_files):
                    for file2 in cluster_files[i+1:]:
                        for rel in relationships:
                            if ((rel.source_file == file1 and rel.target_file == file2) or
                                (rel.source_file == file2 and rel.target_file == file1)):
                                cluster_strengths.append(rel.strength)
                
                if cluster_strengths and sum(cluster_strengths) / len(cluster_strengths) > 0.6:
                    recommendations.append(
                        f"Consider including entire cluster '{cluster_name}' "
                        f"({len(cluster_files)} files) due to strong internal relationships"
                    )
        
        # Recommend including dependencies for import relationships
        import_rels = [r for r in relationships if r.relationship_type == RelationshipType.IMPORT_DEPENDENCY]
        if import_rels:
            recommendations.append(
                f"When including files with import dependencies, consider including "
                f"their imported modules for complete context"
            )
        
        # Recommend including test files with source files
        test_rels = [r for r in relationships if r.relationship_type == RelationshipType.TEST_RELATIONSHIP]
        if test_rels:
            recommendations.append(
                f"When debugging or modifying source files, consider including "
                f"their corresponding test files for better understanding"
            )
        
        return recommendations
    
    async def get_related_files(self, target_files: List[str], 
                               relationship_graph: RelationshipGraph,
                               max_related: int = 10,
                               min_strength: float = 0.5) -> List[Tuple[str, float, List[str]]]:
        """
        Get files related to the target files based on the relationship graph.
        Returns list of (file_path, strength, evidence).
        """
        related_files = defaultdict(lambda: {"strength": 0.0, "evidence": []})
        
        for rel in relationship_graph.relationships:
            if rel.source_file in target_files and rel.strength >= min_strength:
                key = rel.target_file
                related_files[key]["strength"] = max(
                    related_files[key]["strength"], rel.strength
                )
                related_files[key]["evidence"].extend(rel.evidence)
            elif rel.target_file in target_files and rel.strength >= min_strength:
                key = rel.source_file
                related_files[key]["strength"] = max(
                    related_files[key]["strength"], rel.strength
                )
                related_files[key]["evidence"].extend(rel.evidence)
        
        # Sort by strength and return top results
        sorted_related = sorted(
            related_files.items(), 
            key=lambda x: x[1]["strength"], 
            reverse=True
        )
        
        return [
            (file_path, data["strength"], data["evidence"][:3])  # Limit evidence
            for file_path, data in sorted_related[:max_related]
        ]
    
    async def _get_file_content(self, file_path: Path) -> str:
        """Get file content with caching."""
        file_key = str(file_path)
        if file_key not in self._file_contents_cache:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    self._file_contents_cache[file_key] = f.read()
            except Exception as e:
                logging.warning(f"Failed to read {file_path}: {e}")
                self._file_contents_cache[file_key] = ""
        
        return self._file_contents_cache[file_key]
    
    def _get_language_from_extension(self, extension: str) -> str:
        """Map file extension to language name."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.php': 'php',
            '.rb': 'ruby'
        }
        return extension_map.get(extension, 'general')