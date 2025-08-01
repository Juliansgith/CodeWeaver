import asyncio
import json
import logging
from pathlib import Path
import webbrowser
import threading
import time
import traceback
from typing import Dict, Any, List

from aiohttp import web, WSMsgType

from ..core.processor import CodebaseProcessor
from ..core.models import ProcessingOptions
from ..ai.optimization_engine import OptimizationEngine
from ..export.formats import ExportManager
from ..export.chunked_exporter import ChunkConfiguration, ChunkStrategy, create_chunked_export
from ..config.embedding_config import get_embedding_config, EmbeddingProvider
from ..config.settings import SettingsManager
from ..mcp.server import CodeWeaverMCPServer
from ..core.json_utils import safe_json_dumps, serialize_optimization_result, convert_for_json
from ..ai.cost_tracker import get_cost_tracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@web.middleware
async def error_middleware(request, handler):
    """AIOHTTP middleware to catch exceptions and provide detailed logs."""
    try:
        response = await handler(request)
        return response
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Unhandled exception for {request.method} {request.path}:\n{error_trace}")
        
        if 'websockets' in request.app:
            error_payload = {"type": "error", "data": f"Server Error: {e.__class__.__name__}: {e}"}
            for ws in request.app['websockets']:
                await ws.send_str(json.dumps(error_payload))

        return web.json_response({"error": "An internal server error occurred.", "details": str(e)}, status=500)

class WebServer:
    """The main web server for the CodeWeaver GUI."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application(middlewares=[error_middleware])
        self.app['websockets'] = set()
        self.mcp_server_task = None
        self.mcp_server_instance: CodeWeaverMCPServer | None = None
        self.settings = SettingsManager()
        self._setup_routes()

    @property
    def websockets(self):
        return self.app['websockets']

    def _setup_routes(self):
        self.app.router.add_get('/', self.handle_index)
        static_path = Path(__file__).parent / 'static'
        self.app.router.add_static('/static/', path=static_path, name='static')
        self.app.router.add_post('/api/digest', self.api_digest)
        self.app.router.add_post('/api/chunked-export', self.api_chunked_export)
        self.app.router.add_get('/api/embeddings/status', self.api_get_embedding_status)
        self.app.router.add_post('/api/embeddings/configure', self.api_configure_embedding)
        self.app.router.add_get('/api/mcp/status', self.api_get_mcp_status)
        self.app.router.add_post('/api/mcp/start', self.api_start_mcp)
        self.app.router.add_post('/api/mcp/stop', self.api_stop_mcp)
        self.app.router.add_post('/api/browse', self.api_browse)
        self.app.router.add_get('/api/recent-projects', self.api_get_recent_projects)
        self.app.router.add_post('/api/recent-projects', self.api_add_recent_project)
        self.app.router.add_get('/api/costs/current', self.api_get_current_costs)
        self.app.router.add_get('/api/costs/summary', self.api_get_cost_summary)
        self.app.router.add_post('/api/costs/start-session', self.api_start_cost_session)
        self.app.router.add_post('/api/costs/end-session', self.api_end_cost_session)
        self.app.router.add_get('/ws', self.handle_websocket)

    async def handle_index(self, request):
        index_path = Path(__file__).parent / 'static' / 'index.html'
        return web.FileResponse(index_path)

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        logger.info("WebSocket client connected.")
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT and msg.data == 'close':
                    await ws.close()
        finally:
            self.websockets.remove(ws)
            logger.info("WebSocket client disconnected.")
        return ws

    async def _send_ws_message(self, message_type: str, data: Any):
        if not self.websockets: return
        payload = safe_json_dumps({"type": message_type, "data": data})
        for ws in list(self.websockets):
            try:
                await ws.send_str(payload)
            except ConnectionResetError:
                self.websockets.remove(ws)

    async def _log(self, message: str):
        logger.info(message)
        await self._send_ws_message("log", message)

    async def api_digest(self, request):
        data = await request.json()
        project_path = Path(data['path'])
        purpose = data['purpose']
        budget = int(data.get('budget', 200000))
        
        await self._log(f"Received digest request for '{project_path}' with purpose: '{purpose}'")

        async def run_optimization():
            try:
                # Start cost tracking session
                cost_tracker = get_cost_tracker()
                session_id = f"digest_{int(time.time())}"
                cost_tracker.start_session(session_id, str(project_path))
                await self._send_ws_message("cost_session_started", {"session_id": session_id})
                
                engine = OptimizationEngine(project_path)
                await self._log("Discovering project files...")
                processor = CodebaseProcessor()
                # Use comprehensive default ignore patterns
                default_ignore_patterns = [
                    # Python
                    "__pycache__/**", "*.pyc", "*.pyo", "*.pyd", ".Python",
                    "build/**", "develop-eggs/**", "dist/**", "downloads/**",
                    "eggs/**", ".eggs/**", "lib/**", "lib64/**", "parts/**",
                    "sdist/**", "var/**", "wheels/**", 
                    ".venv/**", "venv/**", "env/**", ".env/**",
                    ".coverage", "htmlcov/**", ".pytest_cache/**",
                    "*.log", "celerybeat-schedule", "db.sqlite3", "media/**",
                    
                    # Node.js
                    "node_modules/**", "npm-debug.log*", "yarn-debug.log*",
                    "yarn-error.log*", "lerna-debug.log*", ".pnpm-debug.log*",
                    "*.tsbuildinfo",
                    
                    # General build artifacts
                    "build/**", "dist/**", "out/**", "target/**",
                    "*.min.js", "*.bundle.js", "*.map",
                    
                    # IDE and editor files
                    ".vscode/**", ".idea/**", "*.swp", "*.swo", "*~",
                    ".DS_Store", "Thumbs.db",
                    
                    # Version control
                    ".git/**", ".svn/**", ".hg/**",
                    
                    # Temporary files
                    "tmp/**", "temp/**", "*.tmp", "*.temp",
                    
                    # Documentation build
                    "docs/_build/**", "site/**",
                    
                    # Package manager cache
                    ".npm/**", ".yarn/**", ".pnpm-store/**",
                    
                    # Test coverage
                    "coverage/**", ".nyc_output/**",
                    
                    # Compiled binaries
                    "*.exe", "*.dll", "*.so", "*.dylib", "*.a", "*.lib",
                ]
                options = ProcessingOptions(str(project_path), default_ignore_patterns, 10.0, 'preview')
                file_result = processor.process(options)
                if not file_result.success or not file_result.files:
                    raise Exception("No processable files found.")
                
                await self._log(f"Found {len(file_result.files)} files. Optimizing selection...")
                result = await engine.optimize_file_selection(
                    purpose=purpose, available_files=file_result.files, token_budget=budget
                )
                
                await self._log(f"AI selected {len(result.selected_files)} files. Generating output...")
                output_path = project_path / "codebase_digest.md"
                export_manager = ExportManager()
                
                # Import ExportOptions
                from ..export.formats import ExportOptions
                export_options = ExportOptions()
                
                success = export_manager.export_files(
                    files=result.selected_files, 
                    project_path=project_path,
                    output_path=output_path, 
                    format_name='markdown',
                    options=export_options,
                    budget_allocation=result.budget_allocation,
                    file_importance_info=result.file_scores
                )
                
                if not success:
                    raise Exception("Failed to export files")
                
                await self._log(f"Digest complete! Saved to {output_path}")
                
                # End cost tracking session and send final costs
                cost_summary = cost_tracker.end_session()
                if cost_summary:
                    await self._send_ws_message("cost_summary", {
                        "total_cost": cost_summary.total_cost,
                        "total_requests": cost_summary.total_requests,
                        "total_tokens": cost_summary.total_input_tokens + cost_summary.total_output_tokens
                    })
                
                await self._send_ws_message("digest_complete", {
                    "path": str(output_path),
                    "optimization_result": serialize_optimization_result(result)
                })
            except Exception as e:
                await self._log(f"Error during digest process: {e}")
                # End session on error too
                cost_tracker = get_cost_tracker()
                cost_tracker.end_session()

        asyncio.create_task(run_optimization())
        return web.json_response({"status": "processing_started"})

    async def api_chunked_export(self, request):
        await self._log("Chunked export feature is not fully implemented in the web UI yet.")
        return web.json_response({"status": "not_implemented"}, status=501)

    async def api_get_embedding_status(self, request):
        config_manager = get_embedding_config()
        status = {}
        for provider, has_key in config_manager.get_available_providers().items():
            config = config_manager.get_config(provider)
            status[provider.value] = {"configured": has_key, "model": config.model_name}
        return web.json_response(status)

    async def api_configure_embedding(self, request):
        data = await request.json()
        provider = EmbeddingProvider(data['provider'])
        api_key = data['apiKey']
        config_manager = get_embedding_config()
        config_manager.set_api_key(provider, api_key)
        await self._log(f"API Key for {provider.value} has been configured.")
        return web.json_response({"status": "ok"})

    def _get_mcp_status_data(self):
        is_running = self.mcp_server_task is not None and not self.mcp_server_task.done()
        project_path = str(self.mcp_server_instance.root_path) if self.mcp_server_instance else None
        return {"running": is_running, "project_path": project_path}

    async def _broadcast_mcp_status(self):
        status_data = self._get_mcp_status_data()
        await self._send_ws_message("mcp_status_update", status_data)

    async def api_get_mcp_status(self, request):
        return web.json_response(self._get_mcp_status_data())

    async def api_start_mcp(self, request):
        if self.mcp_server_task and not self.mcp_server_task.done():
            return web.json_response({"error": "MCP server is already running."}, status=400)
        
        data = await request.json()
        project_path = Path(data['path'])
        if not project_path.exists() or not project_path.is_dir():
            return web.json_response({"error": "Invalid project path."}, status=400)
        
        await self._log(f"Starting REAL MCP server for project: {project_path}...")
        self.mcp_server_instance = CodeWeaverMCPServer(root_path=project_path)
        
        def mcp_thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Call the correct method to run the server loop
                loop.run_until_complete(self.mcp_server_instance.run_stdio_loop())
            finally:
                loop.close()

        mcp_thread = threading.Thread(target=mcp_thread_target, daemon=True)
        mcp_thread.start()
        
        async def monitor_task():
            await self._log("MCP Server thread started. Listening on STDIO for AI assistant connections.")
            while mcp_thread.is_alive():
                await asyncio.sleep(5)
            await self._log("MCP Server thread has stopped.")
            self.mcp_server_task = None
            self.mcp_server_instance = None
            await self._broadcast_mcp_status()

        self.mcp_server_task = asyncio.create_task(monitor_task())
        await self._log("MCP Server is now active.")
        
        await self._broadcast_mcp_status()
        
        return web.json_response({"status": "started"})

    async def api_stop_mcp(self, request):
        if not self.mcp_server_task or self.mcp_server_task.done():
            return web.json_response({"error": "MCP server is not running."}, status=400)
        
        self.mcp_server_task.cancel()
        try:
            await self.mcp_server_task
        except asyncio.CancelledError:
            pass
        
        await self._log("MCP Server monitor stopped. The underlying server thread will exit with the application.")
        
        self.mcp_server_task = None
        self.mcp_server_instance = None
        await self._broadcast_mcp_status()

        return web.json_response({"status": "stopped"})

    async def api_browse(self, request):
        data = await request.json()
        req_path_str = data.get('path')
        base_path = Path(req_path_str).resolve() if req_path_str else Path.home()
        home_dir = Path.home()

        if home_dir not in base_path.parents and base_path != home_dir:
            return web.json_response({"error": "Access denied."}, status=403)
        if not base_path.is_dir():
            return web.json_response({"error": "Path is not a valid directory."}, status=400)

        items = [{"name": item.name, "type": "directory", "path": str(item)}
                 for item in sorted(base_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                 if item.is_dir()]
        
        parent_path = str(base_path.parent) if base_path != home_dir else None
        return web.json_response({"current_path": str(base_path), "parent_path": parent_path, "items": items})

    async def api_get_recent_projects(self, request):
        return web.json_response(self.settings.recent_projects)

    async def api_add_recent_project(self, request):
        data = await request.json()
        path = data['path']
        name = Path(path).name
        self.settings.add_recent_project(project_path=path, project_name=name)
        self.settings.save_settings()
        return web.json_response({"status": "ok"})

    async def api_get_current_costs(self, request):
        """Get current session cost stats."""
        cost_tracker = get_cost_tracker()
        stats = cost_tracker.get_current_session_stats()
        return web.json_response(stats)

    async def api_get_cost_summary(self, request):
        """Get cost summary for a time period."""
        hours = int(request.query.get('hours', 24))
        cost_tracker = get_cost_tracker()
        summary = cost_tracker.get_period_summary(hours)
        
        return web.json_response({
            "total_cost": summary.total_cost,
            "total_requests": summary.total_requests,
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "by_provider": summary.by_provider,
            "by_operation": summary.by_operation,
            "by_model": summary.by_model,
            "period_hours": hours
        })

    async def api_start_cost_session(self, request):
        """Start a new cost tracking session."""
        data = await request.json()
        session_id = data.get('session_id', f"session_{int(time.time())}")
        project_path = data.get('project_path')
        
        cost_tracker = get_cost_tracker()
        cost_tracker.start_session(session_id, project_path)
        
        return web.json_response({
            "status": "started",
            "session_id": session_id
        })

    async def api_end_cost_session(self, request):
        """End current cost tracking session."""
        cost_tracker = get_cost_tracker()
        summary = cost_tracker.end_session()
        
        if summary:
            return web.json_response({
                "status": "ended",
                "summary": {
                    "total_cost": summary.total_cost,
                    "total_requests": summary.total_requests,
                    "total_tokens": summary.total_input_tokens + summary.total_output_tokens,
                    "by_provider": summary.by_provider,
                    "by_operation": summary.by_operation
                }
            })
        else:
            return web.json_response({"status": "no_active_session"})

    def run(self):
        url = f"http://{self.host}:{self.port}"
        logger.info(f"Starting CodeWeaver Web UI at {url}")
        threading.Timer(1, lambda: webbrowser.open(url)).start()
        web.run_app(self.app, host=self.host, port=self.port)

def start_web_server():
    server = WebServer()
    server.run()