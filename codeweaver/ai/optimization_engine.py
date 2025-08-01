import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import time
from collections import defaultdict, Counter
import re
import google.generativeai as genai

from ..core.json_utils import safe_json_dumps, convert_for_json

from .embeddings import SemanticCodeSearch, CodeChunk, SemanticMatch, BaseEmbeddingService
from .feedback import FeedbackLoop
from .factory import ai_factory
from ..core.importance_scorer import FileImportanceScorer, FileImportanceInfo, FileType
from ..core.token_budget import SmartTokenBudget, BudgetAllocation, BudgetStrategy
from ..core.dependency_analyzer import DependencyGraphAnalyzer
from ..core.project_detector import ProjectDetector
from ..core.template_manager import TemplateManager, ProjectType
from ..core.relationship_detector import CrossFileRelationshipDetector, RelationshipType
from ..config.llm_config import get_llm_config, LLMProvider, ModelCapability, get_reranking_model

class TaskType(Enum):
    """Different types of coding tasks that require different optimization strategies."""
    DEBUG = "debug"
    UNDERSTAND_FLOW = "understand_flow"
    REVIEW_CODE = "review_code"
    ADD_FEATURE = "add_feature"
    REFACTOR = "refactor"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    API_DESIGN = "api_design"
    GENERAL = "general"

@dataclass
class PurposeAnalysis:
    """Analysis of user's stated purpose/goal."""
    task_type: TaskType
    keywords: List[str]
    technical_domains: List[str]
    file_patterns: List[str]
    confidence_score: float
    reasoning: str

@dataclass
class OptimizationContext:
    """Context for optimization decisions."""
    purpose_analysis: PurposeAnalysis
    project_metadata: Dict[str, Any]
    user_preferences: Dict[str, Any]
    historical_patterns: Optional[Dict[str, Any]] = None

@dataclass
class FileRelevanceScore:
    """Comprehensive relevance score for a file."""
    file_path: str
    base_importance: float
    semantic_relevance: float
    pattern_match: float
    dependency_relevance: float
    combined_score: float
    reasoning: List[str]

@dataclass
class OptimizationResult:
    """Result of optimization engine."""
    selected_files: List[Path]
    file_scores: List[FileRelevanceScore]
    budget_allocation: BudgetAllocation
    optimization_strategy: str
    confidence_score: float
    recommendations: List[str]
    execution_time: float

class PurposeAnalyzer:
    """Analyzes user-stated purposes and converts them to actionable optimization parameters."""
    
    def __init__(self, embedding_service: Optional[BaseEmbeddingService] = None):
        self._embedding_service = embedding_service
        self.task_patterns = self._build_task_patterns()
        self.domain_keywords = self._build_domain_keywords()
    
    @property
    def embedding_service(self) -> Optional[BaseEmbeddingService]:
        """Lazy load the embedding service from the factory."""
        if self._embedding_service is None:
            self._embedding_service = ai_factory.get_embedding_service()
        return self._embedding_service

    def _build_task_patterns(self) -> Dict[TaskType, List[str]]:
        return {
            TaskType.DEBUG: [r'\b(debug|fix|bug|error|issue|problem|broken|crash|exception)\b', r'\b(not working|failing|wrong|incorrect)\b', r'\b(trace|troubleshoot|diagnose)\b'],
            TaskType.UNDERSTAND_FLOW: [r'\b(understand|comprehend|learn|explore|analyze)\b', r'\b(how does|how it works|flow|process|workflow)\b', r'\b(data flow|control flow|execution flow)\b'],
            TaskType.REVIEW_CODE: [r'\b(review|audit|check|examine|inspect)\b', r'\b(code review|pull request|pr|merge)\b', r'\b(quality|standards|best practices)\b'],
            TaskType.ADD_FEATURE: [r'\b(add|implement|create|build|develop)\b', r'\b(feature|functionality|capability)\b', r'\b(new|extend|enhance)\b'],
            TaskType.REFACTOR: [r'\b(refactor|restructure|reorganize|cleanup)\b', r'\b(improve|optimize|simplify|modernize)\b', r'\b(technical debt|code quality)\b'],
            TaskType.SECURITY_AUDIT: [r'\b(security|secure|vulnerability|exploit)\b', r'\b(auth|authorization|authentication|permission)\b', r'\b(sanitize|validate|xss|sql injection)\b'],
            TaskType.PERFORMANCE_OPTIMIZATION: [r'\b(performance|optimize|speed|faster|slow)\b', r'\b(bottleneck|profile|benchmark|cache)\b', r'\b(memory|cpu|latency|throughput)\b'],
            TaskType.DOCUMENTATION: [r'\b(document|docs|documentation|readme)\b', r'\b(explain|describe|comment|annotate)\b', r'\b(api docs|user guide|tutorial)\b'],
            TaskType.TESTING: [r'\b(test|testing|unit test|integration test)\b', r'\b(coverage|mock|stub|fixture)\b', r'\b(tdd|bdd|qa|quality assurance)\b'],
            TaskType.API_DESIGN: [r'\b(api|endpoint|route|service)\b', r'\b(rest|graphql|rpc|microservice)\b', r'\b(schema|contract|interface)\b']
        }
    
    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        return {
            "authentication": ["auth", "login", "logout", "session", "token", "jwt", "oauth", "password", "credential", "user", "account", "signin", "signup"],
            "database": ["db", "database", "sql", "query", "model", "schema", "migration", "orm", "connection", "transaction", "index", "table", "collection"],
            "api": ["api", "endpoint", "route", "controller", "service", "handler", "request", "response", "middleware", "cors", "rest", "graphql"],
            "frontend": ["component", "react", "vue", "angular", "ui", "interface", "render", "state", "props", "template", "style", "css"],
            "backend": ["server", "service", "worker", "job", "queue", "cache", "processor", "handler", "middleware", "pipeline"],
            "payment": ["payment", "billing", "invoice", "charge", "subscription", "stripe", "paypal", "transaction", "credit", "price"],
            "notification": ["notification", "email", "sms", "push", "alert", "message", "mail", "send", "notify", "communication"],
            "file_management": ["file", "upload", "download", "storage", "s3", "blob", "attachment", "document", "media", "asset"],
            "logging": ["log", "logger", "audit", "trace", "monitor", "metric", "analytics", "telemetry", "debug", "error"],
            "security": ["security", "encrypt", "decrypt", "hash", "permission", "role", "access", "validate", "sanitize", "xss", "csrf", "injection"]
        }
    
    async def analyze_purpose(self, purpose_text: str) -> PurposeAnalysis:
        purpose_lower = purpose_text.lower()
        task_type, task_confidence = self._detect_task_type(purpose_lower)
        keywords = self._extract_keywords(purpose_text)
        domains = self._identify_domains(purpose_lower, keywords)
        patterns = self._generate_file_patterns(domains, keywords)
        confidence = min(task_confidence + 0.2, 1.0) if domains else task_confidence
        reasoning = self._generate_reasoning(task_type, domains, keywords, confidence)
        return PurposeAnalysis(task_type=task_type, keywords=keywords, technical_domains=domains, file_patterns=patterns, confidence_score=confidence, reasoning=reasoning)
    
    def _detect_task_type(self, purpose_text: str) -> Tuple[TaskType, float]:
        scores = {}
        for task_type, patterns in self.task_patterns.items():
            score = sum(len(re.findall(pattern, purpose_text, re.IGNORECASE)) for pattern in patterns)
            if score > 0: scores[task_type] = score / len(patterns)
        if not scores: return TaskType.GENERAL, 0.3
        best_task = max(scores, key=scores.get)
        confidence = min(scores[best_task] * 0.8, 0.9)
        return best_task, confidence
    
    def _extract_keywords(self, purpose_text: str) -> List[str]:
        words = re.findall(r'\b\w{3,}\b', purpose_text.lower())
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'have', 'has', 'had', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'do', 'does', 'did', 'get', 'got', 'make', 'made', 'take', 'took', 'come', 'came', 'go', 'went', 'see', 'saw', 'look', 'looked', 'find', 'found', 'give', 'gave', 'work', 'works', 'worked', 'need', 'needs', 'needed', 'want', 'wants', 'wanted', 'like', 'likes', 'liked', 'know', 'knows', 'knew', 'think', 'thinks', 'thought', 'say', 'says', 'said', 'tell', 'tells', 'told', 'ask', 'asks', 'asked', 'try', 'tries', 'tried', 'use', 'uses', 'used', 'help', 'helps', 'helped'}
        keywords = [word for word in words if word not in stop_words and len(word) >= 3]
        return [word for word, _ in Counter(keywords).most_common(10)]
    
    def _identify_domains(self, purpose_text: str, keywords: List[str]) -> List[str]:
        domains = []
        all_terms = purpose_text + ' ' + ' '.join(keywords)
        for domain, domain_keywords in self.domain_keywords.items():
            if sum(1 for keyword in domain_keywords if keyword.lower() in all_terms) >= 1:
                domains.append(domain)
        return domains[:5]
    
    def _generate_file_patterns(self, domains: List[str], keywords: List[str]) -> List[str]:
        patterns = []
        domain_patterns = {
            "authentication": ["*auth*", "*login*", "*session*", "*user*"], "database": ["*model*", "*db*", "*migration*", "*schema*"],
            "api": ["*api*", "*route*", "*controller*", "*endpoint*"], "frontend": ["*component*", "*view*", "*template*", "*ui*"],
            "backend": ["*service*", "*handler*", "*worker*", "*job*"], "payment": ["*payment*", "*billing*", "*invoice*", "*stripe*"],
            "notification": ["*notification*", "*email*", "*message*", "*mail*"], "file_management": ["*file*", "*upload*", "*storage*", "*media*"],
            "logging": ["*log*", "*audit*", "*monitor*", "*metric*"], "security": ["*security*", "*permission*", "*role*", "*access*"]
        }
        for domain in domains: patterns.extend(domain_patterns.get(domain, []))
        for keyword in keywords:
            if len(keyword) >= 4: patterns.append(f"*{keyword}*")
        return list(set(patterns))[:15]
    
    def _generate_reasoning(self, task_type: TaskType, domains: List[str], keywords: List[str], confidence: float) -> str:
        reasoning = f"Detected task type: {task_type.value.replace('_', ' ').title()}"
        if confidence < 0.5: reasoning += " (low confidence)"
        elif confidence > 0.8: reasoning += " (high confidence)"
        if domains: reasoning += f". Technical domains: {', '.join(domains)}"
        if keywords: reasoning += f". Key terms: {', '.join(keywords[:5])}"
        return reasoning

class OptimizationEngine:
    """
    Advanced optimization engine that combines multiple intelligence sources
    to make optimal file selection decisions based on user purpose.
    """
    
    def __init__(self, project_root: Path, 
                 embedding_service: Optional[BaseEmbeddingService] = None,
                 cache_dir: Optional[Path] = None):
        self.project_root = project_root
        self._embedding_service = embedding_service
        self._semantic_search: Optional[SemanticCodeSearch] = None
        
        self.dependency_analyzer = DependencyGraphAnalyzer(self.project_root)
        self.importance_scorer = FileImportanceScorer(self.dependency_analyzer)
        self.token_budget = SmartTokenBudget(self.importance_scorer)
        
        self.purpose_analyzer = PurposeAnalyzer(self._embedding_service)
        self.feedback_loop = FeedbackLoop(cache_dir / 'feedback.db' if cache_dir else Path.home() / '.codeweaver' / 'feedback.db')
        self.template_manager = TemplateManager(cache_dir / 'templates' if cache_dir else Path.home() / '.codeweaver' / 'templates')
        self.relationship_detector = CrossFileRelationshipDetector(project_root, self._embedding_service)
        
        self.default_weights = {
            'importance_weight': 0.3, 'semantic_weight': 0.4,
            'pattern_weight': 0.2, 'dependency_weight': 0.1, 'feedback_weight': 0.2
        }
        
        self.task_specific_weights = {
            TaskType.DEBUG: {'importance_weight': 0.2, 'semantic_weight': 0.5, 'pattern_weight': 0.2, 'dependency_weight': 0.1},
            TaskType.UNDERSTAND_FLOW: {'importance_weight': 0.4, 'semantic_weight': 0.3, 'pattern_weight': 0.1, 'dependency_weight': 0.2},
            TaskType.ADD_FEATURE: {'importance_weight': 0.3, 'semantic_weight': 0.3, 'pattern_weight': 0.3, 'dependency_weight': 0.1},
            TaskType.SECURITY_AUDIT: {'importance_weight': 0.2, 'semantic_weight': 0.4, 'pattern_weight': 0.3, 'dependency_weight': 0.1}
        }

    @property
    def embedding_service(self) -> Optional[BaseEmbeddingService]:
        if self._embedding_service is None:
            self._embedding_service = ai_factory.get_embedding_service()
        return self._embedding_service

    @property
    def semantic_search(self) -> Optional[SemanticCodeSearch]:
        if self._semantic_search is None and self.embedding_service:
            self._semantic_search = SemanticCodeSearch(self.embedding_service)
        return self._semantic_search
    
    async def optimize_file_selection(self, 
                                    purpose: str,
                                    available_files: List[Path],
                                    token_budget: int = 100000,
                                    user_preferences: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        start_time = time.time()
        
        purpose_analysis = await self.purpose_analyzer.analyze_purpose(purpose)
        historical_patterns = self.feedback_loop.get_historical_patterns(purpose)
        
        context = OptimizationContext(
            purpose_analysis=purpose_analysis,
            project_metadata=await self._analyze_project_metadata(available_files),
            user_preferences=user_preferences or {},
            historical_patterns=historical_patterns
        )
        
        if self.semantic_search and not self.semantic_search.code_chunks:
            await self._index_files_for_search(available_files)
        
        file_scores = await self._calculate_file_relevance_scores(available_files, context)

        if self.embedding_service and self.embedding_service.config.api_key:
            file_scores = await self._rerank_with_llm(file_scores, purpose)
        
        file_scores = await self._enhance_with_relationships(file_scores, available_files, context)
        
        selected_files, budget_allocation = await self._select_files_with_budget(
            file_scores, token_budget, context
        )
        
        recommendations = self._generate_recommendations(context, file_scores, selected_files)
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            selected_files=selected_files,
            file_scores=file_scores,
            budget_allocation=budget_allocation,
            optimization_strategy=self._get_strategy_description(context),
            confidence_score=purpose_analysis.confidence_score,
            recommendations=recommendations,
            execution_time=execution_time
        )
    
    async def _analyze_project_metadata(self, files: List[Path]) -> Dict[str, Any]:
        """Analyze project structure and characteristics."""
        file_extensions = defaultdict(int)
        total_size = 0

        for file_path in files:
            try:
                stat = file_path.stat()
                total_size += stat.st_size
                ext = file_path.suffix.lower()
                file_extensions[ext] += 1
            except:
                continue

        detector = ProjectDetector()
        project_type = detector.detect_project_type(self.project_root)
        framework = detector.detect_framework(self.project_root, project_type)

        return {
            'total_files': len(files),
            'total_size_bytes': total_size,
            'file_extensions': dict(file_extensions),
            'project_type': project_type.value if isinstance(project_type, Enum) else project_type,
            'framework': framework,
            'primary_language': self._get_primary_language(file_extensions)
        }
    
    def _get_primary_language(self, extensions: Dict[str, int]) -> str:
        lang_map = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.java': 'java', '.cs': 'csharp', '.go': 'go', '.rs': 'rust', '.cpp': 'cpp', '.c': 'c'}
        code_extensions = {ext: count for ext, count in extensions.items() if ext in lang_map}
        if not code_extensions: return 'unknown'
        primary_ext = max(code_extensions, key=code_extensions.get)
        return lang_map[primary_ext]
    
    async def _index_files_for_search(self, files: List[Path]):
        if not self.semantic_search:
            logging.warning("Semantic search is not available. Skipping file indexing.")
            return
        try:
            stats = await self.semantic_search.index_files(files)
            logging.info(f"Indexed {stats['indexed_files']} files with {stats['total_chunks']} chunks")
        except Exception as e:
            logging.warning(f"Failed to index files for semantic search: {e}")
    
    async def _calculate_file_relevance_scores(self, files: List[Path], context: OptimizationContext) -> List[FileRelevanceScore]:
        scores = []
        
        importance_info = await self._get_importance_scores(files)
        importance_map = {info.relative_path: info for info in importance_info}
        
        semantic_scores = await self._get_semantic_scores(files, context.purpose_analysis)
        pattern_scores = self._get_pattern_match_scores(files, context.purpose_analysis)
        dependency_scores = await self._get_dependency_scores(files, context)
        
        weights = self.task_specific_weights.get(context.purpose_analysis.task_type, self.default_weights)
        
        for file_path in files:
            relative_path = str(file_path.relative_to(self.project_root)).replace("\\", "/")
            
            base_importance = importance_map.get(relative_path, FileImportanceInfo(file_path, relative_path, FileType.UNKNOWN, 0.0)).importance_score
            semantic_relevance = semantic_scores.get(str(file_path), 0.0)
            pattern_match = pattern_scores.get(str(file_path), 0.0)
            dependency_relevance = dependency_scores.get(str(file_path), 0.0)
            
            feedback_score = 0.0
            if context.historical_patterns and context.historical_patterns.get('file_actions'):
                actions = context.historical_patterns['file_actions'].get(str(file_path), {'added': 0, 'removed': 0})
                feedback_score = (actions['added'] - actions['removed']) / (actions['added'] + actions['removed'] + 1)

            combined_score = (
                weights['importance_weight'] * base_importance +
                weights['semantic_weight'] * semantic_relevance +
                weights['pattern_weight'] * pattern_match +
                weights['dependency_weight'] * dependency_relevance +
                weights.get('feedback_weight', 0.2) * feedback_score
            )
            
            reasoning = self._generate_score_reasoning(base_importance, semantic_relevance, pattern_match, dependency_relevance, weights)
            
            scores.append(FileRelevanceScore(
                file_path=str(file_path),
                base_importance=base_importance,
                semantic_relevance=semantic_relevance,
                pattern_match=pattern_match,
                dependency_relevance=dependency_relevance,
                combined_score=combined_score,
                reasoning=reasoning
            ))
        
        return sorted(scores, key=lambda x: x.combined_score, reverse=True)
    
    async def _get_importance_scores(self, files: List[Path]) -> List[FileImportanceInfo]:
        try:
            return await asyncio.to_thread(self.importance_scorer.score_files, files, self.project_root)
        except Exception as e:
            logging.warning(f"Failed to get importance scores: {e}")
            return [FileImportanceInfo(f, str(f.relative_to(self.project_root)).replace("\\", "/"), FileType.UNKNOWN, 0.5) for f in files]
    
    async def _get_semantic_scores(self, files: List[Path], purpose_analysis: PurposeAnalysis) -> Dict[str, float]:
        scores = {}
        if not self.semantic_search: return scores
        
        try:
            query = ' '.join([purpose_analysis.task_type.value.replace('_', ' ')] + purpose_analysis.keywords[:5] + purpose_analysis.technical_domains)
            matches = await self.semantic_search.semantic_search(query, top_k=len(files), min_similarity=0.1)
            for match in matches:
                scores[match.file_path] = match.similarity_score
        except Exception as e:
            logging.warning(f"Failed to get semantic scores: {e}")
        
        return scores
    
    def _get_pattern_match_scores(self, files: List[Path], purpose_analysis: PurposeAnalysis) -> Dict[str, float]:
        scores = {}
        patterns = purpose_analysis.file_patterns
        if not patterns: return scores
        
        for file_path in files:
            file_path_str, file_name = str(file_path).lower(), file_path.name.lower()
            score = 0.0
            for pattern in patterns:
                pattern_clean = pattern.replace('*', '').lower()
                if pattern_clean in file_name: score += 0.8
                elif pattern_clean in file_path_str: score += 0.6
                for keyword in purpose_analysis.keywords:
                    if keyword.lower() in file_name: score += 0.4
                    elif keyword.lower() in file_path_str: score += 0.2
            scores[str(file_path)] = min(score, 1.0)
        
        return scores
    
    async def _get_dependency_scores(self, files: List[Path], context: OptimizationContext) -> Dict[str, float]:
        scores = {}
        try:
            analysis = await asyncio.to_thread(self.dependency_analyzer.analyze_dependencies, files)
            if analysis:
                # analysis is a Dict[str, DependencyNode]
                for file_path, dependency_node in analysis.items():
                    scores[str(self.project_root / file_path)] = dependency_node.centrality_score
        except Exception as e:
            logging.warning(f"Failed to get dependency scores: {e}")
        return scores
    
    def _generate_score_reasoning(self, base_importance: float, semantic_relevance: float, pattern_match: float, dependency_relevance: float, weights: Dict[str, float]) -> List[str]:
        reasoning = []
        if base_importance > 0.7: reasoning.append(f"High base importance ({base_importance:.2f})")
        elif base_importance > 0.4: reasoning.append(f"Medium base importance ({base_importance:.2f})")
        if semantic_relevance > 0.6: reasoning.append(f"Strong semantic match ({semantic_relevance:.2f})")
        elif semantic_relevance > 0.3: reasoning.append(f"Moderate semantic match ({semantic_relevance:.2f})")
        if pattern_match > 0.5: reasoning.append(f"File name/path matches patterns ({pattern_match:.2f})")
        if dependency_relevance > 0.5: reasoning.append(f"High dependency centrality ({dependency_relevance:.2f})")
        if not reasoning: reasoning.append("Low relevance across all factors")
        return reasoning
    
    async def _select_files_with_budget(self, file_scores: List[FileRelevanceScore], token_budget: int, context: OptimizationContext) -> Tuple[List[Path], BudgetAllocation]:
        strategy = self._choose_allocation_strategy(context.purpose_analysis.task_type)
        
        importance_info = []
        for score in file_scores:
            info = FileImportanceInfo(
                path=Path(score.file_path),
                relative_path=str(Path(score.file_path).relative_to(self.project_root)).replace("\\", "/"),
                importance_score=score.combined_score,
                file_type=FileType.OTHER, tokens=1000, language="", suggestions=[]
            )
            importance_info.append(info)
        
        budget_allocation = await asyncio.to_thread(
            self.token_budget.allocate_budget, importance_info, token_budget, strategy
        )
        selected_paths = [info.path for info in budget_allocation.selected_files]
        return selected_paths, budget_allocation
    
    def _choose_allocation_strategy(self, task_type: TaskType) -> BudgetStrategy:
        strategy_map = {
            TaskType.DEBUG: BudgetStrategy.IMPORTANCE_FIRST, TaskType.UNDERSTAND_FLOW: BudgetStrategy.COVERAGE_FIRST,
            TaskType.REVIEW_CODE: BudgetStrategy.BALANCED, TaskType.ADD_FEATURE: BudgetStrategy.SMART_SAMPLING,
            TaskType.REFACTOR: BudgetStrategy.COVERAGE_FIRST, TaskType.SECURITY_AUDIT: BudgetStrategy.IMPORTANCE_FIRST,
            TaskType.PERFORMANCE_OPTIMIZATION: BudgetStrategy.EFFICIENCY_FIRST, TaskType.DOCUMENTATION: BudgetStrategy.COVERAGE_FIRST,
            TaskType.TESTING: BudgetStrategy.BALANCED, TaskType.API_DESIGN: BudgetStrategy.IMPORTANCE_FIRST,
            TaskType.GENERAL: BudgetStrategy.BALANCED
        }
        return strategy_map.get(task_type, BudgetStrategy.BALANCED)
    
    def _generate_recommendations(self, context: OptimizationContext, file_scores: List[FileRelevanceScore], selected_files: List[Path]) -> List[str]:
        recommendations = []
        task_type = context.purpose_analysis.task_type
        
        if task_type == TaskType.DEBUG: recommendations.extend(["Focus on error handling and logging files", "Include test files to understand expected behavior"])
        elif task_type == TaskType.UNDERSTAND_FLOW: recommendations.extend(["Start with entry point files and main controllers", "Follow dependency chains to understand data flow"])
        elif task_type == TaskType.SECURITY_AUDIT: recommendations.extend(["Prioritize authentication and authorization code", "Review input validation and sanitization"])
        
        if context.purpose_analysis.confidence_score < 0.5: recommendations.append("Consider refining your purpose description for better results")
        
        high_score_files = [score for score in file_scores if score.combined_score > 0.7]
        if len(high_score_files) > len(selected_files):
            recommendations.append(f"Consider increasing token budget to include {len(high_score_files) - len(selected_files)} more highly relevant files")
        
        domains = context.purpose_analysis.technical_domains
        if "database" in domains: recommendations.append("Include model and migration files for complete context")
        if "api" in domains: recommendations.append("Include route definitions and middleware")
        
        return recommendations[:5]
    
    def _get_strategy_description(self, context: OptimizationContext) -> str:
        task_type, confidence = context.purpose_analysis.task_type, context.purpose_analysis.confidence_score
        desc = f"{task_type.value.replace('_', ' ').title()}-optimized selection"
        if confidence > 0.8: desc += " with high confidence"
        elif confidence < 0.5: desc += " with low confidence"
        if context.purpose_analysis.technical_domains:
            desc += f" focusing on {', '.join(context.purpose_analysis.technical_domains[:3])}"
        return desc

    async def _rerank_with_llm(self, files: List[FileRelevanceScore], purpose: str, top_n: int = 10) -> List[FileRelevanceScore]:
        """Re-rank files using LLM with centralized model configuration."""
        llm_config = get_llm_config()
        
        # Get available providers
        available_providers = llm_config.get_available_providers()
        if not available_providers:
            logging.warning("LLM re-ranking skipped: No API keys configured.")
            return files
        
        # Try each available provider for re-ranking
        for provider in available_providers:
            try:
                # Get the preferred re-ranking model for this provider
                reranking_model = get_reranking_model(provider)
                if not reranking_model:
                    continue
                
                api_key = llm_config.get_api_key(provider)
                if not api_key:
                    continue
                
                top_files = files[:top_n]
                prompt = self._build_reranking_prompt(top_files, purpose)
                
                if provider == LLMProvider.OPENAI:
                    reranked = await self._rerank_with_openai(api_key, reranking_model, prompt, top_files)
                elif provider == LLMProvider.ANTHROPIC:
                    reranked = await self._rerank_with_anthropic(api_key, reranking_model, prompt, top_files)
                elif provider == LLMProvider.GEMINI:
                    reranked = await self._rerank_with_gemini(api_key, reranking_model, prompt, top_files)
                else:
                    continue
                
                if reranked:
                    logging.info(f"Successfully re-ranked files using {provider.value} {reranking_model}")
                    return reranked + files[top_n:]
                    
            except Exception as e:
                logging.error(f"Error during {provider.value} re-ranking: {e}")
                continue
        
        logging.warning("LLM re-ranking failed for all providers.")
        return files

    async def _rerank_with_openai(self, api_key: str, model: str, prompt: str, 
                                 top_files: List[FileRelevanceScore]) -> Optional[List[FileRelevanceScore]]:
        """Re-rank using OpenAI models."""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a code analysis assistant. Respond only with a JSON array of file paths ranked by relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return self._parse_reranking_response(response.choices[0].message.content, top_files)
        except Exception as e:
            logging.error(f"OpenAI re-ranking failed: {e}")
            return None

    async def _rerank_with_anthropic(self, api_key: str, model: str, prompt: str,
                                   top_files: List[FileRelevanceScore]) -> Optional[List[FileRelevanceScore]]:
        """Re-rank using Anthropic models."""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=api_key)
            
            response = await client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{
                    "role": "user", 
                    "content": f"You are a code analysis assistant. Respond only with a JSON array of file paths ranked by relevance.\n\n{prompt}"
                }]
            )
            
            return self._parse_reranking_response(response.content[0].text, top_files)
        except Exception as e:
            logging.error(f"Anthropic re-ranking failed: {e}")
            return None

    async def _rerank_with_gemini(self, api_key: str, model: str, prompt: str,
                                top_files: List[FileRelevanceScore]) -> Optional[List[FileRelevanceScore]]:
        """Re-rank using Gemini models."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            genai_model = genai.GenerativeModel(model)
            
            response = await genai_model.generate_content_async(prompt)
            return self._parse_reranking_response(response.text, top_files)
        except Exception as e:
            logging.error(f"Gemini re-ranking failed: {e}")
            return None

    def _build_reranking_prompt(self, files: List[FileRelevanceScore], purpose: str) -> str:
        prompt = f"""Given the user's purpose: '{purpose}', re-rank the following files by their relevance. 
        Provide a JSON list of file paths in the new order of importance. Do not include any other text in your response.

        Files to re-rank:
        """
        for i, file_score in enumerate(files):
            try:
                with open(file_score.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(2000)
                prompt += f"""
                --- File {i+1}: {file_score.file_path} ---
                {content}
                ---
                """
            except FileNotFoundError:
                continue
        return prompt

    def _parse_reranking_response(self, response_text: str, original_files: List[FileRelevanceScore]) -> List[FileRelevanceScore]:
        try:
            reranked_paths = json.loads(response_text)
            reranked_files = []
            original_files_map = {f.file_path: f for f in original_files}
            for path in reranked_paths:
                if path in original_files_map:
                    reranked_files.append(original_files_map[path])
            return reranked_files
        except (json.JSONDecodeError, TypeError):
            logging.error(f"Failed to parse LLM re-ranking response: {response_text}")
            return original_files

    def log_user_feedback(self, purpose: str, selected_files: List[Path], user_feedback: Dict[str, Any]):
        self.feedback_loop.log_interaction(purpose, selected_files, user_feedback)
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        return {
            'embedding_service_stats': self.semantic_search.get_search_stats() if self.semantic_search else {},
            'cache_stats': self.embedding_service.cache.get_cache_stats() if self.embedding_service else {},
            'supported_task_types': [t.value for t in TaskType],
            'supported_domains': list(self.purpose_analyzer.domain_keywords.keys())
        }
    
    async def _enhance_with_relationships(self, file_scores: List[FileRelevanceScore], available_files: List[Path], context: OptimizationContext) -> List[FileRelevanceScore]:
        try:
            relationship_graph = await self.relationship_detector.analyze_relationships(available_files)
            high_scoring_files = [score.file_path for score in file_scores if score.combined_score > 0.6][:10]
            if not high_scoring_files: return file_scores
            
            related_files_info = await self.relationship_detector.get_related_files(
                target_files=high_scoring_files, relationship_graph=relationship_graph, max_related=20, min_strength=0.4
            )
            
            score_map = {score.file_path: score for score in file_scores}
            relationship_bonuses = {}
            
            for related_file, strength, evidence in related_files_info:
                if related_file in score_map:
                    base_bonus = strength * 0.3
                    for rel in relationship_graph.relationships:
                        if (rel.source_file == related_file and rel.target_file in high_scoring_files) or (rel.target_file == related_file and rel.source_file in high_scoring_files):
                            type_multiplier = {
                                RelationshipType.IMPORT_DEPENDENCY: 1.0, RelationshipType.SHARED_INTERFACE: 0.9,
                                RelationshipType.TEST_RELATIONSHIP: 0.8, RelationshipType.SEMANTIC_SIMILARITY: 0.7,
                                RelationshipType.DATA_FLOW: 0.6, RelationshipType.CONFIG_DEPENDENCY: 0.5,
                            }.get(rel.relationship_type, 0.5)
                            bonus = base_bonus * type_multiplier
                            relationship_bonuses[related_file] = max(relationship_bonuses.get(related_file, 0), bonus)
            
            enhanced_scores = []
            for score in file_scores:
                bonus = relationship_bonuses.get(score.file_path, 0)
                if bonus > 0:
                    new_combined_score = min(score.combined_score + bonus, 1.0)
                    new_reasoning = list(score.reasoning) + [f"Relationship bonus: +{bonus:.2f}"]
                    enhanced_scores.append(FileRelevanceScore(
                        file_path=score.file_path, base_importance=score.base_importance,
                        semantic_relevance=score.semantic_relevance, pattern_match=score.pattern_match,
                        dependency_relevance=score.dependency_relevance + bonus,
                        combined_score=new_combined_score, reasoning=new_reasoning
                    ))
                else:
                    enhanced_scores.append(score)
            
            enhanced_scores.sort(key=lambda x: x.combined_score, reverse=True)
            logging.info(f"Enhanced {len(relationship_bonuses)} files with relationship bonuses")
            return enhanced_scores
            
        except Exception as e:
            logging.warning(f"Failed to enhance with relationships: {e}")
            return file_scores