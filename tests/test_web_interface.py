import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from aiohttp import web, WSMsgType
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from aiohttp.web_ws import WSMsgType

from codeweaver.web.server import WebServer, error_middleware
from codeweaver.core.models import ProcessingResult, ProcessingStats
from codeweaver.ai.optimization_engine import OptimizationResult, FileRelevanceScore
from codeweaver.core.token_budget import BudgetAllocation, BudgetStrategy
from codeweaver.export.chunked_exporter import ChunkConfiguration, ChunkStrategy
from codeweaver.config.embedding_config import EmbeddingProvider


class TestWebServer(AioHTTPTestCase):
    """Test the main web server functionality."""
    
    async def get_application(self):
        """Create application for testing."""
        server = WebServer()
        return server.app
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_project = self.temp_dir / "test_project"
        self.test_project.mkdir()
        
        # Create test files
        (self.test_project / "main.py").write_text("""
def main():
    print("Hello from main!")
    return "success"

if __name__ == "__main__":
    main()
""")
        
        (self.test_project / "utils.py").write_text("""
def helper_function(x, y):
    return x + y

def validate_input(data):
    return isinstance(data, (str, int, float))
""")
        
        (self.test_project / "config.json").write_text('{
    "app_name": "Test App",
    "debug": true
}')
    
    def tearDown(self):
        """Clean up test fixtures."""
        super().tearDown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest_run_loop
    async def test_index_page(self):
        """Test serving the index page."""
        # Mock the static file path
        with patch('pathlib.Path.exists', return_value=True), \
             patch('aiohttp.web.FileResponse') as mock_response:
            
            mock_response.return_value = web.Response(text="<html>Test</html>", content_type="text/html")
            
            resp = await self.client.request("GET", "/")
            
            assert resp.status == 200
    
    @unittest_run_loop
    async def test_websocket_connection(self):
        """Test WebSocket connection handling."""
        async with self.client.ws_connect("/ws") as ws:
            # WebSocket should connect successfully
            assert not ws.closed
            
            # Test closing the connection
            await ws.send_str("close")
            
            # Wait for close
            await asyncio.sleep(0.1)
    
    @unittest_run_loop
    async def test_websocket_message_broadcast(self):
        """Test WebSocket message broadcasting."""
        # Connect multiple WebSocket clients
        ws1 = await self.client.ws_connect("/ws")
        ws2 = await self.client.ws_connect("/ws")
        
        try:
            # Get the server instance to send a broadcast message
            server = WebServer()
            server.app = self.app
            server.app['websockets'] = {ws1, ws2}
            
            # Send a broadcast message
            await server._send_ws_message("test", {"message": "hello"})
            
            # Both clients should receive the message
            msg1 = await ws1.receive()
            msg2 = await ws2.receive()
            
            assert msg1.type == WSMsgType.TEXT
            assert msg2.type == WSMsgType.TEXT
            
            data1 = json.loads(msg1.data)
            data2 = json.loads(msg2.data)
            
            assert data1["type"] == "test"
            assert data1["data"]["message"] == "hello"
            assert data2["type"] == "test"
            assert data2["data"]["message"] == "hello"
            
        finally:
            await ws1.close()
            await ws2.close()
    
    @unittest_run_loop
    async def test_api_browse(self):
        """Test the browse API endpoint."""
        request_data = {
            "path": str(self.test_project)
        }
        
        resp = await self.client.post("/api/browse", json=request_data)
        
        assert resp.status == 200
        data = await resp.json()
        
        assert "files" in data
        assert "directories" in data
        
        # Should find our test files
        file_names = [f["name"] for f in data["files"]]
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "config.json" in file_names
    
    @unittest_run_loop
    async def test_api_browse_invalid_path(self):
        """Test browse API with invalid path."""
        request_data = {
            "path": "/nonexistent/path"
        }
        
        resp = await self.client.post("/api/browse", json=request_data)
        
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data
    
    @unittest_run_loop
    @patch('codeweaver.web.server.OptimizationEngine')
    @patch('codeweaver.web.server.CodebaseProcessor')
    @patch('codeweaver.web.server.ExportManager')
    async def test_api_digest(self, mock_export_manager, mock_processor_cls, mock_engine_cls):
        """Test the digest API endpoint."""
        # Set up mocks
        mock_processor = MagicMock()
        mock_processor.process.return_value = ProcessingResult(
            success=True,
            files=[self.test_project / "main.py", self.test_project / "utils.py"]
        )
        mock_processor_cls.return_value = mock_processor
        
        mock_engine = MagicMock()
        optimization_result = OptimizationResult(
            selected_files=[self.test_project / "main.py"],
            file_scores=[
                FileRelevanceScore(
                    file_path=str(self.test_project / "main.py"),
                    base_importance=0.8, semantic_relevance=0.7, pattern_match=0.6,
                    dependency_relevance=0.5, combined_score=0.75, reasoning=[]
                )
            ],
            budget_allocation=BudgetAllocation(
                selected_files=[], filtered_files=[], budget_used=5000,
                budget_remaining=45000, total_budget=50000, 
                over_budget_files=0, under_budget_threshold=0, strategy=BudgetStrategy.BALANCED
            ),
            optimization_strategy="test", confidence_score=0.85,
            recommendations=[], execution_time=0.1
        )
        
        async def mock_optimize(*args, **kwargs):
            return optimization_result
        
        mock_engine.optimize_file_selection = mock_optimize
        mock_engine_cls.return_value = mock_engine
        
        mock_export_manager_instance = MagicMock()
        mock_export_manager_instance.export_files.return_value = True
        mock_export_manager.return_value = mock_export_manager_instance
        
        # Test digest request
        request_data = {
            "project_path": str(self.test_project),
            "purpose": "test purpose",
            "format": "markdown",
            "budget": 50000,
            "options": {
                "strip_comments": False,
                "optimize_whitespace": False
            }
        }
        
        resp = await self.client.post("/api/digest", json=request_data)
        
        assert resp.status == 200
        data = await resp.json()
        
        assert data["success"] is True
        assert "output_path" in data
        assert "stats" in data
        assert "optimization_result" in data
    
    @unittest_run_loop
    async def test_api_digest_missing_path(self):
        """Test digest API with missing project path."""
        request_data = {
            "purpose": "test purpose",
            "format": "markdown"
        }
        
        resp = await self.client.post("/api/digest", json=request_data)
        
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data
    
    @unittest_run_loop
    @patch('codeweaver.web.server.create_chunked_export')
    async def test_api_chunked_export(self, mock_create_chunked):
        """Test the chunked export API endpoint."""
        mock_create_chunked.return_value = True
        
        request_data = {
            "input_dir": str(self.test_project),
            "output_dir": str(self.temp_dir / "output"),
            "config": {
                "chunk_strategy": "balanced",
                "max_tokens_per_chunk": 25000,
                "max_files_per_chunk": 25,
                "generate_cross_references": True
            },
            "export_format": "markdown",
            "base_name": "test_chunk"
        }
        
        resp = await self.client.post("/api/chunked-export", json=request_data)
        
        assert resp.status == 200
        data = await resp.json()
        
        assert data["success"] is True
        mock_create_chunked.assert_called_once()
    
    @unittest_run_loop
    async def test_api_chunked_export_failure(self):
        """Test chunked export API failure."""
        with patch('codeweaver.web.server.create_chunked_export', return_value=False):
            request_data = {
                "input_dir": str(self.test_project),
                "output_dir": str(self.temp_dir / "output"),
                "config": {},
                "export_format": "markdown"
            }
            
            resp = await self.client.post("/api/chunked-export", json=request_data)
            
            assert resp.status == 500
            data = await resp.json()
            assert data["success"] is False
    
    @unittest_run_loop
    @patch('codeweaver.web.server.get_embedding_config')
    async def test_api_get_embedding_status(self, mock_get_config):
        """Test getting embedding status."""
        mock_config_manager = MagicMock()
        mock_config_manager.get_available_providers.return_value = {
            EmbeddingProvider.OPENAI: True,
            EmbeddingProvider.GEMINI: False
        }
        mock_get_config.return_value = mock_config_manager
        
        resp = await self.client.get("/api/embeddings/status")
        
        assert resp.status == 200
        data = await resp.json()
        
        assert "providers" in data
        assert data["providers"]["openai"] is True
        assert data["providers"]["gemini"] is False
    
    @unittest_run_loop
    @patch('codeweaver.web.server.get_embedding_config')
    async def test_api_configure_embedding(self, mock_get_config):
        """Test configuring embedding provider."""
        mock_config_manager = MagicMock()
        mock_get_config.return_value = mock_config_manager
        
        request_data = {
            "provider": "openai",
            "api_key": "test-api-key-123"
        }
        
        resp = await self.client.post("/api/embeddings/configure", json=request_data)
        
        assert resp.status == 200
        data = await resp.json()
        
        assert data["success"] is True
        mock_config_manager.set_api_key.assert_called_once()
    
    @unittest_run_loop
    async def test_api_get_recent_projects(self):
        """Test getting recent projects."""
        with patch.object(self.app['server'] if 'server' in self.app else MagicMock(), 'settings') as mock_settings:
            mock_settings.get_recent_projects.return_value = [
                {
                    "path": str(self.test_project),
                    "name": "test_project",
                    "last_accessed": "2023-01-01T00:00:00Z"
                }
            ]
            
            resp = await self.client.get("/api/recent-projects")
            
            assert resp.status == 200
            data = await resp.json()
            
            assert "projects" in data
            assert len(data["projects"]) == 1
            assert data["projects"][0]["name"] == "test_project"
    
    @unittest_run_loop
    async def test_api_add_recent_project(self):
        """Test adding a project to recent projects."""
        request_data = {
            "path": str(self.test_project),
            "name": "New Test Project"
        }
        
        with patch.object(self.app.get('server', MagicMock()), 'settings') as mock_settings:
            resp = await self.client.post("/api/recent-projects", json=request_data)
            
            assert resp.status == 200
            data = await resp.json()
            
            assert data["success"] is True
    
    @unittest_run_loop
    async def test_api_mcp_status(self):
        """Test getting MCP server status."""
        resp = await self.client.get("/api/mcp/status")
        
        assert resp.status == 200
        data = await resp.json()
        
        assert "running" in data
        # Initially should not be running
        assert data["running"] is False
    
    @unittest_run_loop
    @patch('codeweaver.web.server.CodeWeaverMCPServer')
    async def test_api_start_mcp(self, mock_mcp_server_cls):
        """Test starting MCP server."""
        mock_server = MagicMock()
        
        async def mock_main():
            await asyncio.sleep(0.1)  # Simulate server running
        
        mock_server.main = mock_main
        mock_mcp_server_cls.return_value = mock_server
        
        request_data = {
            "project_path": str(self.test_project)
        }
        
        resp = await self.client.post("/api/mcp/start", json=request_data)
        
        assert resp.status == 200
        data = await resp.json()
        
        assert data["success"] is True
        assert "message" in data
    
    @unittest_run_loop
    async def test_api_stop_mcp(self):
        """Test stopping MCP server."""
        # First, simulate that a server is running
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        
        # Patch the server to have a running MCP task
        with patch.object(self.app.get('server', MagicMock()), 'mcp_server_task', mock_task):
            resp = await self.client.post("/api/mcp/stop")
            
            assert resp.status == 200
            data = await resp.json()
            
            assert data["success"] is True


class TestErrorMiddleware:
    """Test the error middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_error_middleware_success(self):
        """Test middleware with successful request."""
        # Mock a successful handler
        async def success_handler(request):
            return web.Response(text="success")
        
        # Mock request
        request = MagicMock()
        request.method = "GET"
        request.path = "/test"
        
        # Test middleware
        response = await error_middleware(request, success_handler)
        
        assert response.text == "success"
    
    @pytest.mark.asyncio
    async def test_error_middleware_exception(self):
        """Test middleware with handler exception."""
        # Mock a failing handler
        async def failing_handler(request):
            raise ValueError("Test error")
        
        # Mock request and app
        request = MagicMock()
        request.method = "POST"
        request.path = "/api/test"
        request.app = {'websockets': []}
        
        # Test middleware
        response = await error_middleware(request, failing_handler)
        
        assert response.status == 500
        assert response.content_type == "application/json"
    
    @pytest.mark.asyncio
    async def test_error_middleware_with_websockets(self):
        """Test middleware error handling with WebSocket broadcasting."""
        # Mock WebSocket
        mock_ws = AsyncMock()
        
        # Mock failing handler
        async def failing_handler(request):
            raise RuntimeError("Server crashed")
        
        # Mock request with WebSocket
        request = MagicMock()
        request.method = "GET"
        request.path = "/api/crash"
        request.app = {'websockets': [mock_ws]}
        
        # Test middleware
        response = await error_middleware(request, failing_handler)
        
        assert response.status == 500
        
        # Verify WebSocket was notified
        mock_ws.send_str.assert_called_once()
        error_message = mock_ws.send_str.call_args[0][0]
        error_data = json.loads(error_message)
        
        assert error_data["type"] == "error"
        assert "RuntimeError" in error_data["data"]


class TestWebServerIntegration:
    """Integration tests for WebServer components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.server = WebServer(host="localhost", port=8080)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_server_initialization(self):
        """Test WebServer initialization."""
        assert self.server.host == "localhost"
        assert self.server.port == 8080
        assert self.server.app is not None
        assert 'websockets' in self.server.app
        assert len(self.server.websockets) == 0
    
    def test_routes_setup(self):
        """Test that all routes are properly configured."""
        routes = [str(route.resource) for route in self.server.app.router.routes()]
        
        expected_routes = [
            "/",
            "/api/digest",
            "/api/chunked-export",
            "/api/embeddings/status",
            "/api/embeddings/configure",
            "/api/mcp/status",
            "/api/mcp/start",
            "/api/mcp/stop",
            "/api/browse",
            "/api/recent-projects",
            "/ws"
        ]
        
        # Check that expected routes exist (may not be exact match due to aiohttp internals)
        for expected_route in expected_routes:
            # Look for routes that contain our expected path
            found = any(expected_route in route for route in routes)
            assert found, f"Route {expected_route} not found in {routes}"
    
    def test_websocket_management(self):
        """Test WebSocket connection management."""
        # Mock WebSocket
        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        
        # Add WebSockets
        self.server.websockets.add(mock_ws1)
        self.server.websockets.add(mock_ws2)
        
        assert len(self.server.websockets) == 2
        
        # Remove WebSocket
        self.server.websockets.remove(mock_ws1)
        
        assert len(self.server.websockets) == 1
        assert mock_ws2 in self.server.websockets
    
    @pytest.mark.asyncio
    async def test_send_ws_message_no_connections(self):
        """Test sending WebSocket message with no connections."""
        # Should not raise exception
        await self.server._send_ws_message("test", {"data": "test"})
    
    @pytest.mark.asyncio
    async def test_send_ws_message_with_connections(self):
        """Test sending WebSocket message to connected clients."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        self.server.websockets.add(mock_ws1)
        self.server.websockets.add(mock_ws2)
        
        await self.server._send_ws_message("notification", {"message": "hello"})
        
        # Both WebSockets should receive the message
        mock_ws1.send_str.assert_called_once()
        mock_ws2.send_str.assert_called_once()
        
        # Check message content
        sent_message = mock_ws1.send_str.call_args[0][0]
        message_data = json.loads(sent_message)
        
        assert message_data["type"] == "notification"
        assert message_data["data"]["message"] == "hello"
    
    @pytest.mark.asyncio
    async def test_send_ws_message_connection_error(self):
        """Test WebSocket message handling with connection errors."""
        mock_ws_good = AsyncMock()
        mock_ws_bad = AsyncMock()
        
        # Mock connection error
        mock_ws_bad.send_str.side_effect = ConnectionResetError("Connection lost")
        
        self.server.websockets.add(mock_ws_good)
        self.server.websockets.add(mock_ws_bad)
        
        await self.server._send_ws_message("test", {"data": "test"})
        
        # Good WebSocket should receive message
        mock_ws_good.send_str.assert_called_once()
        
        # Bad WebSocket should be removed from the set
        assert mock_ws_bad not in self.server.websockets
        assert mock_ws_good in self.server.websockets
    
    def test_settings_integration(self):
        """Test settings manager integration."""
        assert self.server.settings is not None
        # Settings should be properly initialized
        assert hasattr(self.server.settings, 'get_recent_projects')
    
    def test_mcp_server_properties(self):
        """Test MCP server related properties."""
        assert self.server.mcp_server_task is None
        assert self.server.mcp_server_instance is None
        
        # These should be settable
        mock_task = MagicMock()
        mock_instance = MagicMock()
        
        self.server.mcp_server_task = mock_task
        self.server.mcp_server_instance = mock_instance
        
        assert self.server.mcp_server_task == mock_task
        assert self.server.mcp_server_instance == mock_instance


class TestWebServerHelpers:
    """Test helper functions and utilities."""
    
    def test_chunk_strategy_conversion(self):
        """Test conversion between string and ChunkStrategy enum."""
        strategy_map = {
            "by_size": ChunkStrategy.BY_SIZE,
            "by_count": ChunkStrategy.BY_COUNT,
            "by_directory": ChunkStrategy.BY_DIRECTORY,
            "by_importance": ChunkStrategy.BY_IMPORTANCE,
            "by_type": ChunkStrategy.BY_TYPE,
            "balanced": ChunkStrategy.BALANCED
        }
        
        for string_strategy, enum_strategy in strategy_map.items():
            # Test that we can convert string to enum
            converted = ChunkStrategy(string_strategy)
            assert converted == enum_strategy
    
    def test_embedding_provider_conversion(self):
        """Test conversion between string and EmbeddingProvider enum."""
        provider_map = {
            "openai": EmbeddingProvider.OPENAI,
            "gemini": EmbeddingProvider.GEMINI
        }
        
        for string_provider, enum_provider in provider_map.items():
            converted = EmbeddingProvider(string_provider)
            assert converted == enum_provider
    
    def test_json_serialization(self):
        """Test JSON serialization of complex objects."""
        # Test that we can serialize common data structures
        test_data = {
            "optimization_result": {
                "selected_files": ["file1.py", "file2.py"],
                "confidence_score": 0.85,
                "recommendations": ["Add more files", "Consider dependencies"]
            },
            "stats": {
                "file_count": 10,
                "total_tokens": 5000,
                "processing_time": 2.5
            }
        }
        
        # Should not raise exception
        json_string = json.dumps(test_data)
        
        # Should be able to parse back
        parsed_data = json.loads(json_string)
        
        assert parsed_data["optimization_result"]["confidence_score"] == 0.85
        assert parsed_data["stats"]["file_count"] == 10


# Test configuration for async tests
pytestmark = pytest.mark.asyncio
