import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, parse_qs
import hashlib
import time

from ..core.dependency_analyzer import DependencyGraphAnalyzer
from ..core.importance_scorer import FileImportanceScorer, FileImportanceInfo
from ..core.token_budget import SmartTokenBudget, BudgetStrategy, BudgetConstraints
from ..core.processor import CodebaseProcessor
from ..core.models import ProcessingOptions
from ..core.tokenizer import TokenEstimator, LLMProvider
from ..ai.conversation_tracker import ConversationTracker
from ..ai.embeddings import BaseEmbeddingService
from ..ai.factory import ai_factory
from ..ai.realtime_context import RealTimeContextManager

@dataclass
class MCPRequest:
    """Represents an MCP request."""
    jsonrpc: str
    method: str
    params: Dict[str, Any]
    id: Optional[Union[str, int]] = None

@dataclass
class MCPResponse:
    """Represents an MCP response."""
    jsonrpc: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

@dataclass
class MCPResource:
    """Represents an MCP resource."""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None

class CodeWeaverMCPServer:
    """
    Model Context Protocol (MCP) server for CodeWeaver.
    Provides intelligent context selection based on user queries and semantic understanding.
    """
    
    def __init__(self, root_path: Optional[Path] = None, 
                 embedding_service: Optional[BaseEmbeddingService] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CodeWeaverMCPServer...")
        
        self.root_path = root_path or Path.cwd()
        self.logger.info(f"  - Project root path: {self.root_path}")
        
        self.dependency_analyzer = DependencyGraphAnalyzer(self.root_path)
        self.logger.info("  - DependencyAnalyzer initialized.")
        
        self.importance_scorer = FileImportanceScorer(self.dependency_analyzer)
        self.logger.info("  - FileImportanceScorer initialized.")
        
        self.token_budget = SmartTokenBudget(self.importance_scorer)
        self.logger.info("  - SmartTokenBudget initialized.")
        
        self.processor = CodebaseProcessor()
        self.logger.info("  - CodebaseProcessor initialized.")
        
        self._embedding_service = embedding_service
        self.logger.info("  - Attempting to get embedding service from factory...")
        if self.embedding_service:
            self.logger.info("  - Embedding service acquired.")
        else:
            self.logger.warning("  - No embedding service available. Semantic features will be disabled.")

        db_path = self.root_path / '.codeweaver'
        db_path.mkdir(exist_ok=True)
        self.conversation_tracker = ConversationTracker(
            db_path=db_path / 'conversations.db',
            embedding_service=self._embedding_service
        )
        self.logger.info("  - ConversationTracker initialized.")
        
        self.realtime_manager = RealTimeContextManager(
            self.conversation_tracker, self._embedding_service
        )
        self.logger.info("  - RealTimeContextManager initialized.")
        
        self._file_cache: Dict[str, Any] = {}
        self._dependency_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 300
        
        self.methods = {
            'initialize': self._handle_initialize,
            'resources/list': self._handle_resources_list,
            'resources/read': self._handle_resources_read,
            'tools/list': self._handle_tools_list,
            'tools/call': self._handle_tools_call,
            'realtime/subscribe': self._handle_realtime_subscribe,
            'realtime/unsubscribe': self._handle_realtime_unsubscribe,
            'realtime/status': self._handle_realtime_status,
        }
        self.logger.info("CodeWeaverMCPServer initialization complete.")

    @property
    def embedding_service(self) -> Optional[BaseEmbeddingService]:
        """Lazy load the embedding service from the factory."""
        if self._embedding_service is None:
            self._embedding_service = ai_factory.get_embedding_service()
        return self._embedding_service

    async def run_stdio_loop(self):
        """Run the main STDIO loop for the MCP server in a cross-platform way."""
        self.logger.info("MCP STDIO loop started. Waiting for input.")
        loop = asyncio.get_running_loop()

        while True:
            try:
                # Run the blocking sys.stdin.readline in a separate thread
                request_data = await asyncio.to_thread(sys.stdin.readline)
                
                if not request_data:
                    self.logger.info("STDIN closed, shutting down MCP loop.")
                    break # EOF
                
                request_data = request_data.strip()
                if request_data:
                    response_data = await self.handle_request(request_data)
                    # sys.stdout is thread-safe
                    print(response_data, flush=True)

            except (BrokenPipeError, EOFError):
                self.logger.info("Pipe closed, shutting down MCP loop.")
                break
            except Exception as e:
                self.logger.error(f"Error in MCP STDIO loop: {e}")
                break
        self.logger.info("MCP STDIO loop finished.")

    async def handle_request(self, request_data: str) -> str:
        """Handle incoming MCP request."""
        try:
            request_json = json.loads(request_data)
            request = MCPRequest(**request_json)
            
            if request.method in self.methods:
                result = await self.methods[request.method](request.params)
                response = MCPResponse(
                    jsonrpc="2.0",
                    result=result,
                    id=request.id
                )
            else:
                response = MCPResponse(
                    jsonrpc="2.0",
                    error={
                        "code": -32601,
                        "message": f"Method not found: {request.method}"
                    },
                    id=request.id
                )
            
            return json.dumps(asdict(response), default=str)
            
        except Exception as e:
            self.logger.error(f"Error handling MCP request: {e}")
            error_response = MCPResponse(
                jsonrpc="2.0",
                error={
                    "code": -32000,
                    "message": str(e)
                }
            )
            return json.dumps(asdict(error_response), default=str)
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "resources": {
                    "subscribe": True,
                    "listChanged": True
                },
                "tools": {}
            },
            "serverInfo": {
                "name": "codeweaver-mcp-server",
                "version": "2.0.0"
            }
        }
    
    async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available resources."""
        resources = [
            MCPResource(
                uri="codeweaver://project/smart-context",
                name="Smart Context Selection",
                description="Intelligently select most relevant files based on query",
                mimeType="application/json"
            ),
            MCPResource(
                uri="codeweaver://project/dependency-graph",
                name="Dependency Graph",
                description="Project dependency analysis and graph",
                mimeType="application/json"
            ),
            MCPResource(
                uri="codeweaver://project/importance-analysis",
                name="File Importance Analysis",
                description="Analysis of file importance scores",
                mimeType="application/json"
            ),
            MCPResource(
                uri="codeweaver://project/token-budget",
                name="Token Budget Optimization",
                description="Optimal token budget allocation",
                mimeType="application/json"
            )
        ]
        
        return {
            "resources": [asdict(resource) for resource in resources]
        }
    
    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a specific resource."""
        uri = params.get("uri", "")
        parsed_uri = urlparse(uri)
        
        if parsed_uri.scheme != "codeweaver" or parsed_uri.netloc != "project":
            raise ValueError(f"Unsupported URI: {uri}")
        
        path = parsed_uri.path.lstrip("/")
        query_params = parse_qs(parsed_uri.query)
        
        if path == "smart-context":
            return await self._handle_smart_context_selection(query_params)
        elif path == "dependency-graph":
            return await self._handle_dependency_graph(query_params)
        elif path == "importance-analysis":
            return await self._handle_importance_analysis(query_params)
        elif path == "token-budget":
            return await self._handle_token_budget(query_params)
        else:
            raise ValueError(f"Unknown resource path: {path}")
    
    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available tools."""
        return {"tools": []}
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls."""
        return {"content": []}
    
    async def _handle_realtime_subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle real-time context update subscriptions."""
        client_id = params.get("client_id")
        session_id = params.get("session_id")
        filters = params.get("filters", {})
        
        if not client_id or not session_id:
            raise ValueError("client_id and session_id are required")
        
        # Create callback for sending updates (this would need to be handled by the transport layer)
        def update_callback(update):
            # In a real implementation, this would send the update via WebSocket or similar
            self.logger.info(f"Context update for client {client_id}: {update.update_type}")
        
        success = await self.realtime_manager.subscribe_client(
            client_id=client_id,
            session_id=session_id,
            callback=update_callback,
            filters=filters
        )
        
        return {
            "subscribed": success,
            "client_id": client_id,
            "session_id": session_id,
            "active_clients": self.realtime_manager.get_client_count(session_id)
        }
    
    async def _handle_realtime_unsubscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle real-time context update unsubscriptions."""
        client_id = params.get("client_id")
        
        if not client_id:
            raise ValueError("client_id is required")
        
        self.realtime_manager.unsubscribe_client(client_id)
        
        return {
            "unsubscribed": True,
            "client_id": client_id
        }
    
    async def _handle_realtime_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time context manager status."""
        return self.realtime_manager.get_manager_stats()
    
    async def _handle_smart_context_selection(self, query_params: Dict[str, List[str]]) -> Dict[str, Any]:
        """Handle smart context selection based on query with conversation awareness."""
        query = query_params.get("query", [""])[0]
        budget = int(query_params.get("budget", ["50000"])[0])
        strategy = query_params.get("strategy", ["balanced"])[0]
        session_id = query_params.get("session_id", [None])[0]
        max_files = int(query_params.get("max_files", ["20"])[0])
        
        # Start or continue conversation session
        if not session_id:
            session_id = self.conversation_tracker.start_session(str(self.root_path))
        elif not self.conversation_tracker.current_session:
            self.conversation_tracker.start_session(str(self.root_path), session_id)
        
        # Add user query to conversation
        self.conversation_tracker.add_message(
            role="user",
            content=query,
            message_type="context_request"
        )
        
        # Get all project files
        project_files = await self._get_project_files()
        
        if not project_files:
            return {
                "contents": [{
                    "uri": "codeweaver://project/smart-context",
                    "mimeType": "application/json",
                    "text": json.dumps({
                        "error": "No files found in project",
                        "selected_files": [],
                        "total_tokens": 0,
                        "session_id": session_id
                    })
                }]
            }
        
        # Get conversation-aware context adaptation
        file_paths = [str(f) for f in project_files]
        adaptation = await self.conversation_tracker.get_context_adaptation(
            user_query=query,
            available_files=file_paths,
            max_files=max_files
        )
        
        # Apply token budget constraints to recommended files
        recommended_paths = [Path(f) for f in adaptation.recommended_files]
        scored_files = await self._analyze_file_importance(recommended_paths)
        
        # Allocate token budget
        budget_strategy = self._parse_budget_strategy(strategy)
        allocation = self.token_budget.allocate_budget(
            scored_files, budget, budget_strategy
        )
        
        # Prepare response with conversation context
        selected_files_info = []
        for file_info in allocation.selected_files:
            confidence = adaptation.confidence_scores.get(str(file_info.path), 0.5)
            selected_files_info.append({
                "path": file_info.relative_path,
                "importance_score": file_info.importance_score,
                "conversation_relevance": confidence,
                "tokens": file_info.tokens,
                "file_type": file_info.file_type.value,
                "efficiency_ratio": file_info.efficiency_ratio
            })
        
        # Add conversation context to response and trigger real-time updates
        assistant_message = self.conversation_tracker.add_message(
            role="assistant",
            content=f"Selected {len(selected_files_info)} files based on conversation context",
            message_type="context_response",
            context_files=[f["path"] for f in selected_files_info]
        )
        
        # Trigger real-time context updates
        await self.realtime_manager.on_context_changed(session_id, adaptation)
        
        context_data = {
            "selected_files": selected_files_info,
            "total_files_available": len(file_paths),
            "files_selected": len(allocation.selected_files),
            "total_tokens": allocation.used_tokens,
            "budget_utilization": allocation.budget_utilization,
            "efficiency_score": allocation.efficiency_score,
            "coverage_score": allocation.coverage_score,
            "strategy_used": allocation.strategy_used.value,
            "conversation_strategy": adaptation.adaptation_strategy,
            "conversation_relevance": adaptation.conversation_relevance,
            "adaptation_reasoning": adaptation.reasoning[:5],  # Top 5 reasons
            "query": query,
            "session_id": session_id,
            "file_contents": await self._get_file_contents(allocation.selected_files)
        }
        
        return {
            "contents": [{
                "uri": "codeweaver://project/smart-context", 
                "mimeType": "application/json",
                "text": json.dumps(context_data, indent=2)
            }]
        }
    
    async def _handle_dependency_graph(self, query_params: Dict[str, List[str]]) -> Dict[str, Any]:
        """Handle dependency graph analysis."""
        project_files = await self._get_project_files()
        
        if not project_files:
            return {
                "contents": [{
                    "uri": "codeweaver://project/dependency-graph",
                    "mimeType": "application/json", 
                    "text": json.dumps({"error": "No files found"})
                }]
            }
        
        # Get or create dependency analysis
        if not self._is_cache_valid():
            dependency_graph = self.dependency_analyzer.analyze_dependencies(project_files)
            self._dependency_cache = {
                "nodes": {},
                "stats": self.dependency_analyzer.get_dependency_stats(),
                "central_files": [],
                "entry_points": [],
                "circular_dependencies": self.dependency_analyzer.find_circular_dependencies()
            }
            
            # Convert nodes to serializable format
            for path, node in dependency_graph.items():
                self._dependency_cache["nodes"][path] = {
                    "relative_path": node.relative_path,
                    "imports_count": len(node.imports_from),
                    "imported_by_count": len(node.imported_by),
                    "centrality_score": node.centrality_score,
                    "imports_from": list(node.imports_from),
                    "imported_by": list(node.imported_by)
                }
            
            # Get most central files
            central_files = self.dependency_analyzer.get_most_central_files(20)
            self._dependency_cache["central_files"] = [
                {
                    "path": node.relative_path,
                    "centrality_score": node.centrality_score,
                    "imports_count": len(node.imports_from),
                    "imported_by_count": len(node.imported_by)
                }
                for node in central_files
            ]
            
            # Get entry points
            entry_points = self.dependency_analyzer.get_entry_points()
            self._dependency_cache["entry_points"] = [
                {
                    "path": node.relative_path,
                    "imports_count": len(node.imports_from),
                    "imported_by_count": len(node.imported_by)
                }
                for node in entry_points
            ]
            
            self._cache_timestamp = time.time()
        
        return {
            "contents": [{
                "uri": "codeweaver://project/dependency-graph",
                "mimeType": "application/json",
                "text": json.dumps(self._dependency_cache, indent=2)
            }]
        }
    
    async def _handle_importance_analysis(self, query_params: Dict[str, List[str]]) -> Dict[str, Any]:
        """Handle file importance analysis."""
        project_files = await self._get_project_files()
        
        if not project_files:
            return {
                "contents": [{
                    "uri": "codeweaver://project/importance-analysis",
                    "mimeType": "application/json",
                    "text": json.dumps({"error": "No files found"})
                }]
            }
        
        scored_files = await self._analyze_file_importance(project_files)
        
        # Prepare analysis data
        files_by_type = self.importance_scorer.get_files_by_type(scored_files)
        most_efficient = self.importance_scorer.get_most_efficient_files(scored_files, 20)
        ignore_suggestions = self.importance_scorer.suggest_ignore_patterns(scored_files)
        summary = self.importance_scorer.get_importance_summary(scored_files)
        
        analysis_data = {
            "summary": summary,
            "files_by_importance": [
                {
                    "path": f.relative_path,
                    "importance_score": f.importance_score,
                    "file_type": f.file_type.value,
                    "tokens": f.tokens,
                    "efficiency_ratio": f.efficiency_ratio,
                    "factors": {
                        "base_type_score": f.factors.base_type_score,
                        "centrality_score": f.factors.centrality_score,
                        "size_factor": f.factors.size_factor,
                        "content_complexity_factor": f.factors.content_complexity_factor,
                        "naming_convention_bonus": f.factors.naming_convention_bonus
                    }
                }
                for f in scored_files[:50]  # Top 50 files
            ],
            "files_by_type": {
                file_type.value: len(files)
                for file_type, files in files_by_type.items()
            },
            "most_efficient_files": [
                {
                    "path": f.relative_path,
                    "efficiency_ratio": f.efficiency_ratio,
                    "importance_score": f.importance_score,
                    "tokens": f.tokens
                }
                for f in most_efficient
            ],
            "ignore_pattern_suggestions": ignore_suggestions
        }
        
        return {
            "contents": [{
                "uri": "codeweaver://project/importance-analysis",
                "mimeType": "application/json",
                "text": json.dumps(analysis_data, indent=2)
            }]
        }
    
    async def _handle_token_budget(self, query_params: Dict[str, List[str]]) -> Dict[str, Any]:
        """Handle token budget optimization."""
        budget = int(query_params.get("budget", ["50000"])[0])
        llm_provider = query_params.get("provider", ["claude"])[0]
        model = query_params.get("model", ["claude-3.5-sonnet"])[0]
        
        project_files = await self._get_project_files()
        scored_files = await self._analyze_file_importance(project_files)
        
        # Compare different strategies
        provider_enum = LLMProvider(llm_provider.lower())
        strategy_comparison = self.token_budget.compare_strategies(scored_files, budget)
        
        # Get LLM-specific budget suggestion
        suggested_budget, optimal_allocation = self.token_budget.suggest_budget_for_llm(
            scored_files, provider_enum, model
        )
        
        budget_data = {
            "requested_budget": budget,
            "suggested_budget": suggested_budget,
            "llm_provider": llm_provider,
            "model": model,
            "strategy_comparison": {
                strategy.value: {
                    "files_selected": len(allocation.selected_files),
                    "tokens_used": allocation.used_tokens,
                    "budget_utilization": allocation.budget_utilization,
                    "efficiency_score": allocation.efficiency_score,
                    "coverage_score": allocation.coverage_score
                }
                for strategy, allocation in strategy_comparison.items()
            },
            "optimal_allocation": {
                "files_selected": len(optimal_allocation.selected_files),
                "tokens_used": optimal_allocation.used_tokens,
                "budget_utilization": optimal_allocation.budget_utilization,
                "efficiency_score": optimal_allocation.efficiency_score,
                "coverage_score": optimal_allocation.coverage_score,
                "selected_files": [
                    {
                        "path": f.relative_path,
                        "importance_score": f.importance_score,
                        "tokens": f.tokens
                    }
                    for f in optimal_allocation.selected_files[:20]  # Top 20
                ]
            }
        }
        
        return {
            "contents": [{
                "uri": "codeweaver://project/token-budget",
                "mimeType": "application/json",
                "text": json.dumps(budget_data, indent=2)
            }]
        }
    
    async def _get_project_files(self) -> List[Path]:
        """Get list of project files."""
        cache_key = "project_files"
        
        if cache_key in self._file_cache and self._is_cache_valid():
            return self._file_cache[cache_key]
        
        project_files = []
        
        # Use CodebaseProcessor to get files with default ignore patterns
        options = ProcessingOptions(
            input_dir=str(self.root_path),
            ignore_patterns=[
                "*.pyc", "__pycache__", ".git", "node_modules", 
                "*.min.js", "*.bundle.js", "dist", "build"
            ],
            size_limit_mb=10.0,
            mode='preview'
        )
        
        result = self.processor.process(options)
        if result.success and result.files:
            project_files = result.files
        
        self._file_cache[cache_key] = project_files
        return project_files
    
    async def _analyze_file_importance(self, project_files: List[Path]) -> List[FileImportanceInfo]:
        """Analyze file importance with caching."""
        cache_key = f"importance_{self._hash_file_list(project_files)}"
        
        if cache_key in self._file_cache and self._is_cache_valid():
            return self._file_cache[cache_key]
        
        # Get token counts for files
        token_info = {}
        for file_path in project_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                estimates = TokenEstimator.estimate_tokens(content, LLMProvider.CLAUDE)
                token_info[str(file_path.relative_to(self.root_path)).replace("\\", "/")] = estimates.get("claude-3.5-sonnet", 0)
            except Exception:
                continue
        
        scored_files = self.importance_scorer.score_files(
            project_files, self.root_path, token_info
        )
        
        self._file_cache[cache_key] = scored_files
        return scored_files
    
    async def _filter_files_by_query(self, scored_files: List, query: str) -> List:
        """Filter files based on semantic query matching."""
        # Simple keyword-based filtering for now
        # In a full implementation, this would use embeddings
        query_keywords = query.lower().split()
        
        filtered_files = []
        for file_info in scored_files:
            file_path_lower = file_info.relative_path.lower()
            
            # Check if any query keyword matches file path or likely content
            relevance_score = 0
            for keyword in query_keywords:
                if keyword in file_path_lower:
                    relevance_score += 2
                
                # Boost score for files that likely contain relevant code
                if any(pattern in file_path_lower for pattern in [
                    keyword, f"{keyword}s", f"{keyword}_", f"_{keyword}"
                ]):
                    relevance_score += 1
            
            # Include files with any relevance, but boost their importance
            if relevance_score > 0:
                file_info.importance_score += relevance_score * 10
                filtered_files.append(file_info)
            elif not query_keywords or len(query_keywords) == 0:
                # Include all files if no meaningful query
                filtered_files.append(file_info)
        
        # Re-sort by updated importance scores
        return sorted(filtered_files, key=lambda f: f.importance_score, reverse=True)
    
    async def _get_file_contents(self, selected_files: List[FileImportanceInfo]) -> Dict[str, str]:
        """Get contents of selected files."""
        contents = {}
        
        for file_info in selected_files[:10]:  # Limit to prevent huge responses
            try:
                with open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                    contents[file_info.relative_path] = f.read()
            except Exception:
                contents[file_info.relative_path] = "[Could not read file]"
        
        return contents
    
    def _parse_budget_strategy(self, strategy_str: str) -> BudgetStrategy:
        """Parse budget strategy from string."""
        strategy_map = {
            "importance": BudgetStrategy.IMPORTANCE_FIRST,
            "efficiency": BudgetStrategy.EFFICIENCY_FIRST, 
            "balanced": BudgetStrategy.BALANCED,
            "coverage": BudgetStrategy.COVERAGE_FIRST,
            "smart": BudgetStrategy.SMART_SAMPLING
        }
        return strategy_map.get(strategy_str.lower(), BudgetStrategy.BALANCED)
    
    def _hash_file_list(self, files: List[Path]) -> str:
        """Create hash of file list for caching."""
        file_paths = sorted(str(f) for f in files)
        return hashlib.md5("".join(file_paths).encode()).hexdigest()[:16]
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        return time.time() - self._cache_timestamp < self._cache_ttl

async def main():
    """Main function to run the MCP server."""
    import sys
    
    root_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    server = CodeWeaverMCPServer(root_path)
    
    # Simple STDIO-based MCP server
    while True:
        try:
            line = input()
            if not line:
                break
            
            response = await server.handle_request(line)
            print(response)
            
        except EOFError:
            break
        except Exception as e:
            logging.error(f"Server error: {e}")
            break

if __name__ == "__main__":
    asyncio.run(main())