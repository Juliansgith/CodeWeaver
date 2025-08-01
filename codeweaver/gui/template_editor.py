"""
Visual Template Editor - Web-based GUI for creating and editing CodeWeaver templates.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import webbrowser
import threading
import time
from datetime import datetime

try:
    from aiohttp import web, WSMsgType
    from aiohttp.web import Application, Response, WebSocketResponse
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("aiohttp not available. Install with: pip install aiohttp")

from ..core.template_manager import SmartTemplateLibrary, TemplateConfig, ProjectType
from ..core.processor import CodebaseProcessor
from ..core.models import ProcessingOptions
from ..core.json_utils import safe_json_dumps, serialize_template_config


class TemplateEditorServer:
    """Web server for the visual template editor."""
    
    def __init__(self, port: int = 8081, host: str = "localhost"):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for the template editor. Install with: pip install aiohttp")
        
        self.host = host
        self.port = port
        self.app = Application()
        self.websockets: List[WebSocketResponse] = []
        self.template_library = SmartTemplateLibrary(Path.home() / '.codeweaver' / 'templates')
        
        # Setup routes
        self._setup_routes()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_routes(self):
        """Setup web routes."""
        # Static files
        self.app.router.add_get('/', self._serve_index)
        self.app.router.add_get('/editor', self._serve_editor)
        self.app.router.add_static('/static/', path=self._get_static_dir(), name='static')
        
        # API endpoints
        self.app.router.add_get('/api/templates', self._api_get_templates)
        self.app.router.add_get('/api/templates/{template_name}', self._api_get_template)
        self.app.router.add_post('/api/templates', self._api_create_template)
        self.app.router.add_put('/api/templates/{template_name}', self._api_update_template)
        self.app.router.add_delete('/api/templates/{template_name}', self._api_delete_template)
        self.app.router.add_post('/api/detect-project', self._api_detect_project)
        self.app.router.add_post('/api/analyze-project', self._api_analyze_project)
        
        # WebSocket for real-time updates
        self.app.router.add_get('/ws', self._websocket_handler)
    
    def _get_static_dir(self) -> Path:
        """Get the static files directory."""
        return Path(__file__).parent / 'static'
    
    async def _serve_index(self, request) -> Response:
        """Serve the main index page."""
        html_content = self._generate_index_html()
        return Response(text=html_content, content_type='text/html')
    
    async def _serve_editor(self, request) -> Response:
        """Serve the template editor page."""
        html_content = self._generate_editor_html()
        return Response(text=html_content, content_type='text/html')
    
    async def _websocket_handler(self, request):
        """Handle WebSocket connections for real-time updates."""
        ws = WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.append(ws)
        self.logger.info(f"WebSocket client connected. Total clients: {len(self.websockets)}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON message'
                        }))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {ws.exception()}')
        except Exception as e:
            self.logger.error(f"WebSocket handler error: {e}")
        finally:
            if ws in self.websockets:
                self.websockets.remove(ws)
            self.logger.info(f"WebSocket client disconnected. Total clients: {len(self.websockets)}")
        
        return ws
    
    async def _handle_websocket_message(self, ws: WebSocketResponse, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        message_type = data.get('type')
        
        if message_type == 'ping':
            await ws.send_str(json.dumps({'type': 'pong'}))
        elif message_type == 'subscribe':
            await ws.send_str(json.dumps({
                'type': 'subscribed',
                'message': 'Successfully subscribed to template updates'
            }))
        else:
            await ws.send_str(json.dumps({
                'type': 'error',
                'message': f'Unknown message type: {message_type}'
            }))
    
    async def _broadcast_update(self, update_data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients."""
        if not self.websockets:
            return
        
        message = json.dumps(update_data)
        disconnected = []
        
        for ws in self.websockets:
            try:
                await ws.send_str(message)
            except Exception as e:
                self.logger.error(f"Failed to send WebSocket message: {e}")
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            if ws in self.websockets:
                self.websockets.remove(ws)
    
    # API Endpoints
    async def _api_get_templates(self, request) -> Response:
        """Get all available templates."""
        try:
            templates = self.template_library.get_available_templates()
            template_data = []
            
            for name, config in templates:
                template_info = {
                    'name': name,
                    'display_name': config.name,
                    'description': config.description,
                    'project_type': config.project_type.value,
                    'tags': config.tags,
                    'created_at': config.created_at,
                    'usage_stats': config.usage_stats,
                    'is_custom': name.startswith('custom_')
                }
                template_data.append(template_info)
            
            return Response(
                text=json.dumps({'templates': template_data}),
                content_type='application/json'
            )
        except Exception as e:
            return Response(
                text=json.dumps({'error': str(e)}),
                content_type='application/json',
                status=500
            )
    
    async def _api_get_template(self, request) -> Response:
        """Get a specific template."""
        template_name = request.match_info['template_name']
        
        try:
            template = self.template_library.get_template(template_name)
            if not template:
                return Response(
                    text=json.dumps({'error': 'Template not found'}),
                    content_type='application/json',
                    status=404
                )
            
            return Response(
                text=safe_json_dumps({'template': serialize_template_config(template)}),
                content_type='application/json'
            )
        except Exception as e:
            return Response(
                text=json.dumps({'error': str(e)}),
                content_type='application/json',
                status=500
            )
    
    async def _api_create_template(self, request) -> Response:
        """Create a new template."""
        try:
            data = await request.json()
            
            # Validate required fields
            required_fields = ['name', 'description', 'project_type']
            for field in required_fields:
                if field not in data:
                    return Response(
                        text=json.dumps({'error': f'Missing required field: {field}'}),
                        content_type='application/json',
                        status=400
                    )
            
            # Create template configuration
            template_config = TemplateConfig(
                name=data['name'],
                description=data['description'],
                project_type=ProjectType(data['project_type']),
                ignore_patterns=data.get('ignore_patterns', []),
                priority_files=data.get('priority_files', []),
                entry_points=data.get('entry_points', []),
                config_files=data.get('config_files', []),
                test_patterns=data.get('test_patterns', []),
                build_artifacts=data.get('build_artifacts', []),
                documentation_files=data.get('documentation_files', []),
                tags=data.get('tags', []),
                file_type_weights=data.get('file_type_weights', {}),
                directory_weights=data.get('directory_weights', {}),
                created_at=datetime.now().isoformat(),
                author=data.get('author', 'Template Editor')
            )
            
            # Save template
            template_key = data['name'].lower().replace(' ', '_')
            self.template_library.save_template(template_key, template_config, is_custom=True)
            
            # Broadcast update
            await self._broadcast_update({
                'type': 'template_created',
                'template_name': template_key,
                'template_data': asdict(template_config)
            })
            
            return Response(
                text=json.dumps({'success': True, 'template_name': template_key}),
                content_type='application/json'
            )
        except Exception as e:
            return Response(
                text=json.dumps({'error': str(e)}),
                content_type='application/json',
                status=500
            )
    
    async def _api_update_template(self, request) -> Response:
        """Update an existing template."""
        template_name = request.match_info['template_name']
        
        try:
            data = await request.json()
            
            # Get existing template
            existing_template = self.template_library.get_template(template_name)
            if not existing_template:
                return Response(
                    text=json.dumps({'error': 'Template not found'}),
                    content_type='application/json',
                    status=404
                )
            
            # Update fields
            for field, value in data.items():
                if hasattr(existing_template, field):
                    if field == 'project_type' and isinstance(value, str):
                        setattr(existing_template, field, ProjectType(value))
                    else:
                        setattr(existing_template, field, value)
            
            # Save updated template
            is_custom = template_name.startswith('custom_')
            actual_name = template_name.replace('custom_', '') if is_custom else template_name
            self.template_library.save_template(actual_name, existing_template, is_custom=is_custom)
            
            # Broadcast update
            await self._broadcast_update({
                'type': 'template_updated',
                'template_name': template_name,
                'template_data': asdict(existing_template)
            })
            
            return Response(
                text=json.dumps({'success': True}),
                content_type='application/json'
            )
        except Exception as e:
            return Response(
                text=json.dumps({'error': str(e)}),
                content_type='application/json',
                status=500
            )
    
    async def _api_delete_template(self, request) -> Response:
        """Delete a template."""
        template_name = request.match_info['template_name']
        
        try:
            # Only allow deletion of custom templates
            if not template_name.startswith('custom_'):
                return Response(
                    text=json.dumps({'error': 'Cannot delete built-in templates'}),
                    content_type='application/json',
                    status=403
                )
            
            # Delete template file
            template_path = self.template_library.custom_templates_dir / f"{template_name.replace('custom_', '')}.json"
            if template_path.exists():
                template_path.unlink()
            
            # Remove from memory
            if template_name in self.template_library.templates:
                del self.template_library.templates[template_name]
            
            # Broadcast update
            await self._broadcast_update({
                'type': 'template_deleted',
                'template_name': template_name
            })
            
            return Response(
                text=json.dumps({'success': True}),
                content_type='application/json'
            )
        except Exception as e:
            return Response(
                text=json.dumps({'error': str(e)}),
                content_type='application/json',
                status=500
            )
    
    async def _api_detect_project(self, request) -> Response:
        """Detect project type from a given path."""
        try:
            data = await request.json()
            project_path = Path(data.get('path', ''))
            
            if not project_path.exists():
                return Response(
                    text=json.dumps({'error': 'Path does not exist'}),
                    content_type='application/json',
                    status=400
                )
            
            detected_types = self.template_library.detect_project_type(project_path)
            
            return Response(
                text=json.dumps({
                    'detected_types': [(ptype.value, score) for ptype, score in detected_types]
                }),
                content_type='application/json'
            )
        except Exception as e:
            return Response(
                text=json.dumps({'error': str(e)}),
                content_type='application/json',
                status=500
            )
    
    async def _api_analyze_project(self, request) -> Response:
        """Analyze a project to suggest template configuration."""
        try:
            data = await request.json()
            project_path = Path(data.get('path', ''))
            
            if not project_path.exists():
                return Response(
                    text=json.dumps({'error': 'Path does not exist'}),
                    content_type='application/json',
                    status=400
                )
            
            # Analyze project structure
            processor = CodebaseProcessor()
            options = ProcessingOptions(
                input_dir=str(project_path),
                ignore_patterns=["*.pyc", "__pycache__", ".git", "node_modules"],
                size_limit_mb=10.0,
                mode='preview'
            )
            
            result = processor.process(options)
            if not result.success:
                return Response(
                    text=json.dumps({'error': 'Failed to analyze project'}),
                    content_type='application/json',
                    status=500
                )
            
            # Get project analysis
            suggestion = self.template_library.create_template_from_project(
                name="Analyzed Template",
                project_path=project_path,
                description=f"Template created from analysis of {project_path.name}",
                is_custom=False  # Don't save yet
            )
            
            return Response(
                text=json.dumps({
                    'suggestion': asdict(suggestion),
                    'files_analyzed': len(result.files) if result.files else 0
                }),
                content_type='application/json'
            )
        except Exception as e:
            return Response(
                text=json.dumps({'error': str(e)}),
                content_type='application/json',
                status=500
            )
    
    def _generate_index_html(self) -> str:
        """Generate the main index page HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeWeaver Template Editor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .container {
            background: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 600px;
            width: 90%;
        }
        
        .logo {
            font-size: 3em;
            margin-bottom: 20px;
            color: #667eea;
        }
        
        h1 {
            color: #333;
            margin-bottom: 20px;
            font-size: 2.5em;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 40px;
            font-size: 1.2em;
            line-height: 1.6;
        }
        
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .feature {
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            text-align: left;
        }
        
        .feature h3 {
            color: #333;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .feature-icon {
            font-size: 1.2em;
        }
        
        .cta-button {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1em;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .cta-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        .status.connected {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">🧩</div>
        <h1>CodeWeaver Template Editor</h1>
        <p class="subtitle">
            Create, edit, and manage intelligent project templates with our visual drag-and-drop interface.
            Perfect for standardizing code analysis across different project types.
        </p>
        
        <div class="features">
            <div class="feature">
                <h3><span class="feature-icon">🎨</span> Visual Editor</h3>
                <p>Drag-and-drop interface for creating templates without writing JSON.</p>
            </div>
            <div class="feature">
                <h3><span class="feature-icon">🔍</span> Smart Detection</h3>
                <p>Automatically detect project types and suggest template configurations.</p>
            </div>
            <div class="feature">
                <h3><span class="feature-icon">⚡</span> Real-time Preview</h3>
                <p>See template changes applied instantly with live preview functionality.</p>
            </div>
        </div>
        
        <a href="/editor" class="cta-button">Launch Template Editor</a>
        
        <div id="status" class="status" style="display: none;"></div>
    </div>
    
    <script>
        // Test WebSocket connection
        function testConnection() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            const statusEl = document.getElementById('status');
            
            ws.onopen = function() {
                statusEl.textContent = '✅ Server connection established';
                statusEl.className = 'status connected';
                statusEl.style.display = 'block';
                ws.send(JSON.stringify({type: 'ping'}));
            };
            
            ws.onerror = function() {
                statusEl.textContent = '❌ Failed to connect to server';
                statusEl.className = 'status error';
                statusEl.style.display = 'block';
            };
            
            ws.onclose = function() {
                setTimeout(testConnection, 3000); // Retry after 3 seconds
            };
        }
        
        // Test connection when page loads
        window.addEventListener('load', testConnection);
    </script>
</body>
</html>
        """
    
    def _generate_editor_html(self) -> str:
        """Generate the template editor page HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Template Editor - CodeWeaver</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* Header */
        .header {
            background: white;
            border-bottom: 1px solid #e1e5e9;
            padding: 15px 20px;
            display: flex;
            justify-content: between;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .header h1 {
            color: #333;
            font-size: 1.5em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .header-actions {
            display: flex;
            gap: 10px;
            margin-left: auto;
        }
        
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background: #5a6fd8;
        }
        
        .btn-secondary {
            background: #f8f9fa;
            color: #333;
            border: 1px solid #e1e5e9;
        }
        
        .btn-secondary:hover {
            background: #e9ecef;
        }
        
        .btn-success {
            background: #28a745;
            color: white;
        }
        
        .btn-success:hover {
            background: #218838;
        }
        
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        
        .btn-danger:hover {
            background: #c82333;
        }
        
        /* Main Layout */
        .main {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        
        /* Sidebar */
        .sidebar {
            width: 300px;
            background: white;
            border-right: 1px solid #e1e5e9;
            display: flex;
            flex-direction: column;
        }
        
        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid #e1e5e9;
        }
        
        .sidebar-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .template-list {
            list-style: none;
        }
        
        .template-item {
            padding: 12px;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s;
            background: white;
        }
        
        .template-item:hover {
            border-color: #667eea;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.1);
        }
        
        .template-item.active {
            border-color: #667eea;
            background: #f8f9ff;
        }
        
        .template-name {
            font-weight: 600;
            color: #333;
            margin-bottom: 4px;
        }
        
        .template-description {
            font-size: 12px;
            color: #666;
            margin-bottom: 4px;
        }
        
        .template-meta {
            display: flex;
            gap: 8px;
            font-size: 11px;
        }
        
        .template-tag {
            background: #e9ecef;
            color: #495057;
            padding: 2px 6px;
            border-radius: 3px;
        }
        
        .template-type {
            background: #667eea;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
        }
        
        /* Editor */
        .editor {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: white;
        }
        
        .editor-header {
            padding: 20px;
            border-bottom: 1px solid #e1e5e9;
            display: flex;
            justify-content: between;
            align-items: center;
        }
        
        .editor-title {
            font-size: 1.3em;
            color: #333;
        }
        
        .editor-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            color: #333;
        }
        
        .form-input, .form-textarea, .form-select {
            width: 100%;
            padding: 10px;
            border: 1px solid #e1e5e9;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.2s;
        }
        
        .form-input:focus, .form-textarea:focus, .form-select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .form-textarea {
            resize: vertical;
            min-height: 80px;
        }
        
        .form-help {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
        
        /* Drag and Drop */
        .drop-zone {
            border: 2px dashed #e1e5e9;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            color: #666;
            background: #f8f9fa;
            margin-bottom: 20px;
            transition: all 0.2s;
        }
        
        .drop-zone.dragover {
            border-color: #667eea;
            background: #f8f9ff;
            color: #667eea;
        }
        
        .patterns-list {
            list-style: none;
            border: 1px solid #e1e5e9;
            border-radius: 6px;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .pattern-item {
            padding: 8px 12px;
            border-bottom: 1px solid #e1e5e9;
            display: flex;
            justify-content: between;
            align-items: center;
        }
        
        .pattern-item:last-child {
            border-bottom: none;
        }
        
        .pattern-text {
            font-family: monospace;
            font-size: 13px;
        }
        
        .remove-pattern {
            background: none;
            border: none;
            color: #dc3545;
            cursor: pointer;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }
        
        .remove-pattern:hover {
            background: #f8d7da;
        }
        
        /* Status and notifications */
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 6px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            transform: translateX(400px);
            transition: transform 0.3s;
        }
        
        .notification.show {
            transform: translateX(0);
        }
        
        .notification.success {
            background: #28a745;
        }
        
        .notification.error {
            background: #dc3545;
        }
        
        .notification.warning {
            background: #ffc107;
            color: #212529;
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        
        .empty-state h3 {
            margin-bottom: 10px;
            color: #333;
        }
        
        /* Loading */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
            color: #666;
        }
        
        .spinner {
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .main {
                flex-direction: column;
            }
            
            .sidebar {
                width: 100%;
                height: 200px;
            }
            
            .header-actions {
                flex-wrap: wrap;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>🧩 Template Editor</h1>
        <div class="header-actions">
            <button class="btn btn-secondary" onclick="analyzeProject()">
                📁 Analyze Project
            </button>
            <button class="btn btn-primary" onclick="createNewTemplate()">
                ➕ New Template
            </button>
            <button class="btn btn-success" onclick="saveTemplate()" id="saveBtn" disabled>
                💾 Save
            </button>
            <a href="/" class="btn btn-secondary">← Back</a>
        </div>
    </header>
    
    <main class="main">
        <aside class="sidebar">
            <div class="sidebar-header">
                <h3>Templates</h3>
            </div>
            <div class="sidebar-content">
                <div id="templateList" class="loading">
                    <div class="spinner"></div>
                    Loading templates...
                </div>
            </div>
        </aside>
        
        <section class="editor">
            <div class="editor-header">
                <h2 class="editor-title" id="editorTitle">Select a template to edit</h2>
            </div>
            <div class="editor-content" id="editorContent">
                <div class="empty-state">
                    <h3>Welcome to the Template Editor</h3>
                    <p>Select an existing template from the sidebar or create a new one to get started.</p>
                </div>
            </div>
        </section>
    </main>
    
    <script>
        let currentTemplate = null;
        let websocket = null;
        let templates = [];
        
        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            loadTemplates();
        });
        
        // WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            websocket = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            websocket.onopen = function() {
                console.log('WebSocket connected');
                websocket.send(JSON.stringify({type: 'subscribe'}));
            };
            
            websocket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            websocket.onclose = function() {
                console.log('WebSocket disconnected');
                setTimeout(initWebSocket, 3000);
            };
            
            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'template_created':
                case 'template_updated':
                case 'template_deleted':
                    loadTemplates();
                    break;
                case 'error':
                    showNotification(data.message, 'error');
                    break;
            }
        }
        
        // Load templates from API
        async function loadTemplates() {
            try {
                const response = await fetch('/api/templates');
                const data = await response.json();
                
                if (data.templates) {
                    templates = data.templates;
                    renderTemplateList();
                } else {
                    throw new Error('Failed to load templates');
                }
            } catch (error) {
                console.error('Error loading templates:', error);
                document.getElementById('templateList').innerHTML = 
                    '<div class="error">Failed to load templates</div>';
            }
        }
        
        function renderTemplateList() {
            const listEl = document.getElementById('templateList');
            
            if (templates.length === 0) {
                listEl.innerHTML = '<div class="empty-state"><p>No templates found</p></div>';
                return;
            }
            
            const html = `
                <ul class="template-list">
                    ${templates.map(template => `
                        <li class="template-item" onclick="selectTemplate('${template.name}')">
                            <div class="template-name">${template.display_name}</div>
                            <div class="template-description">${template.description}</div>
                            <div class="template-meta">
                                <span class="template-type">${template.project_type}</span>
                                ${template.is_custom ? '<span class="template-tag">Custom</span>' : ''}
                                ${template.tags.slice(0, 2).map(tag => `<span class="template-tag">${tag}</span>`).join('')}
                            </div>
                        </li>
                    `).join('')}
                </ul>
            `;
            
            listEl.innerHTML = html;
        }
        
        // Select and load a template
        async function selectTemplate(templateName) {
            try {
                // Update UI
                document.querySelectorAll('.template-item').forEach(item => {
                    item.classList.remove('active');
                });
                event.target.closest('.template-item').classList.add('active');
                
                // Load template data
                const response = await fetch(`/api/templates/${templateName}`);
                const data = await response.json();
                
                if (data.template) {
                    currentTemplate = data.template;
                    renderTemplateEditor();
                    document.getElementById('saveBtn').disabled = false;
                } else {
                    throw new Error('Template not found');
                }
            } catch (error) {
                console.error('Error loading template:', error);
                showNotification('Failed to load template', 'error');
            }
        }
        
        function renderTemplateEditor() {
            if (!currentTemplate) return;
            
            document.getElementById('editorTitle').textContent = currentTemplate.name;
            
            const html = `
                <form id="templateForm" onsubmit="return false;">
                    <div class="form-group">
                        <label class="form-label">Template Name</label>
                        <input type="text" class="form-input" name="name" value="${currentTemplate.name}" onchange="markDirty()">
                        <div class="form-help">Display name for this template</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Description</label>
                        <textarea class="form-textarea" name="description" onchange="markDirty()">${currentTemplate.description}</textarea>
                        <div class="form-help">Brief description of what this template is for</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Project Type</label>
                        <select class="form-select" name="project_type" onchange="markDirty()">
                            ${getProjectTypeOptions(currentTemplate.project_type)}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Tags</label>
                        <input type="text" class="form-input" name="tags" value="${currentTemplate.tags.join(', ')}" onchange="markDirty()">
                        <div class="form-help">Comma-separated tags for categorization</div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Ignore Patterns</label>
                        <div class="drop-zone" ondrop="dropPatterns(event)" ondragover="allowDrop(event)">
                            Drop text files here or add patterns manually
                        </div>
                        <input type="text" class="form-input" placeholder="Add pattern (e.g., *.pyc, node_modules/**)" onkeypress="addPatternOnEnter(event, 'ignore_patterns')">
                        <ul class="patterns-list" id="ignorePatterns">
                            ${currentTemplate.ignore_patterns.map(pattern => `
                                <li class="pattern-item">
                                    <span class="pattern-text">${pattern}</span>
                                    <button type="button" class="remove-pattern" onclick="removePattern('ignore_patterns', '${pattern}')">×</button>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Priority Files</label>
                        <input type="text" class="form-input" placeholder="Add priority file pattern" onkeypress="addPatternOnEnter(event, 'priority_files')">
                        <div class="form-help">Files that should always be included</div>
                        <ul class="patterns-list" id="priorityFiles">
                            ${currentTemplate.priority_files.map(pattern => `
                                <li class="pattern-item">
                                    <span class="pattern-text">${pattern}</span>
                                    <button type="button" class="remove-pattern" onclick="removePattern('priority_files', '${pattern}')">×</button>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Entry Points</label>
                        <input type="text" class="form-input" placeholder="Add entry point file" onkeypress="addPatternOnEnter(event, 'entry_points')">
                        <div class="form-help">Main entry point files for the project</div>
                        <ul class="patterns-list" id="entryPoints">
                            ${currentTemplate.entry_points.map(pattern => `
                                <li class="pattern-item">
                                    <span class="pattern-text">${pattern}</span>
                                    <button type="button" class="remove-pattern" onclick="removePattern('entry_points', '${pattern}')">×</button>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Test Patterns</label>
                        <input type="text" class="form-input" placeholder="Add test file pattern" onkeypress="addPatternOnEnter(event, 'test_patterns')">
                        <div class="form-help">Patterns to identify test files</div>
                        <ul class="patterns-list" id="testPatterns">
                            ${currentTemplate.test_patterns.map(pattern => `
                                <li class="pattern-item">
                                    <span class="pattern-text">${pattern}</span>
                                    <button type="button" class="remove-pattern" onclick="removePattern('test_patterns', '${pattern}')">×</button>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                </form>
            `;
            
            document.getElementById('editorContent').innerHTML = html;
        }
        
        function getProjectTypeOptions(selectedType) {
            const types = [
                'frontend_react', 'frontend_vue', 'frontend_angular', 'frontend_vanilla',
                'backend_nodejs', 'backend_python', 'backend_java', 'backend_dotnet',
                'backend_go', 'backend_rust', 'mobile_react_native', 'mobile_flutter',
                'mobile_ionic', 'mobile_native_ios', 'mobile_native_android',
                'fullstack_mern', 'fullstack_django', 'fullstack_rails',
                'microservices', 'data_science', 'machine_learning', 'devops',
                'game_unity', 'game_unreal', 'desktop_electron', 'cli_tool',
                'library', 'api_rest', 'api_graphql'
            ];
            
            return types.map(type => 
                `<option value="${type}" ${type === selectedType ? 'selected' : ''}>${type.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}</option>`
            ).join('');
        }
        
        // Drag and drop functionality
        function allowDrop(ev) {
            ev.preventDefault();
            ev.target.classList.add('dragover');
        }
        
        function dropPatterns(ev) {
            ev.preventDefault();
            ev.target.classList.remove('dragover');
            
            const files = ev.dataTransfer.files;
            for (let file of files) {
                if (file.type === 'text/plain') {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const content = e.target.result;
                        const patterns = content.split('\\n').filter(line => line.trim() && !line.startsWith('#'));
                        patterns.forEach(pattern => addPattern('ignore_patterns', pattern.trim()));
                    };
                    reader.readAsText(file);
                }
            }
        }
        
        function addPatternOnEnter(event, type) {
            if (event.key === 'Enter') {
                const value = event.target.value.trim();
                if (value) {
                    addPattern(type, value);
                    event.target.value = '';
                }
            }
        }
        
        function addPattern(type, pattern) {
            if (!currentTemplate[type].includes(pattern)) {
                currentTemplate[type].push(pattern);
                renderPatternList(type);
                markDirty();
            }
        }
        
        function removePattern(type, pattern) {
            currentTemplate[type] = currentTemplate[type].filter(p => p !== pattern);
            renderPatternList(type);
            markDirty();
        }
        
        function renderPatternList(type) {
            const containerMap = {
                'ignore_patterns': 'ignorePatterns',
                'priority_files': 'priorityFiles',
                'entry_points': 'entryPoints',
                'test_patterns': 'testPatterns'
            };
            
            const containerId = containerMap[type];
            const container = document.getElementById(containerId);
            
            if (container) {
                container.innerHTML = currentTemplate[type].map(pattern => `
                    <li class="pattern-item">
                        <span class="pattern-text">${pattern}</span>
                        <button type="button" class="remove-pattern" onclick="removePattern('${type}', '${pattern}')">×</button>
                    </li>
                `).join('');
            }
        }
        
        // Form handling
        function markDirty() {
            document.getElementById('saveBtn').style.background = '#28a745';
        }
        
        async function saveTemplate() {
            if (!currentTemplate) return;
            
            try {
                // Get form data
                const form = document.getElementById('templateForm');
                const formData = new FormData(form);
                
                // Update current template
                currentTemplate.name = formData.get('name');
                currentTemplate.description = formData.get('description');
                currentTemplate.project_type = formData.get('project_type');
                currentTemplate.tags = formData.get('tags').split(',').map(tag => tag.trim()).filter(tag => tag);
                
                // Find original template name
                const originalTemplate = templates.find(t => t.name === getCurrentTemplateName());
                const templateName = originalTemplate ? originalTemplate.name : currentTemplate.name.toLowerCase().replace(/\\s+/g, '_');
                
                // Save to server
                const response = await fetch(`/api/templates/${templateName}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(currentTemplate)
                });
                
                if (response.ok) {
                    showNotification('Template saved successfully!', 'success');
                    document.getElementById('saveBtn').style.background = '#667eea';
                    loadTemplates();
                } else {
                    throw new Error('Failed to save template');
                }
            } catch (error) {
                console.error('Error saving template:', error);
                showNotification('Failed to save template', 'error');
            }
        }
        
        function getCurrentTemplateName() {
            const activeItem = document.querySelector('.template-item.active');
            return activeItem ? activeItem.onclick.toString().match(/'([^']+)'/)[1] : null;
        }
        
        // New template creation
        function createNewTemplate() {
            const name = prompt('Enter template name:');
            if (!name) return;
            
            const description = prompt('Enter template description:');
            if (!description) return;
            
            currentTemplate = {
                name: name,
                description: description,
                project_type: 'library',
                ignore_patterns: [],
                priority_files: [],
                entry_points: [],
                config_files: [],
                test_patterns: [],
                build_artifacts: [],
                documentation_files: [],
                tags: [],
                file_type_weights: {},
                directory_weights: {}
            };
            
            renderTemplateEditor();
            document.getElementById('saveBtn').disabled = false;
            markDirty();
        }
        
        // Project analysis
        async function analyzeProject() {
            const path = prompt('Enter project path to analyze:');
            if (!path) return;
            
            try {
                showNotification('Analyzing project...', 'warning');
                
                const response = await fetch('/api/analyze-project', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({path: path})
                });
                
                const data = await response.json();
                
                if (data.suggestion) {
                    currentTemplate = data.suggestion;
                    renderTemplateEditor();
                    document.getElementById('saveBtn').disabled = false;
                    markDirty();
                    showNotification(`Project analyzed! Found ${data.files_analyzed} files.`, 'success');
                } else {
                    throw new Error(data.error || 'Analysis failed');
                }
            } catch (error) {
                console.error('Error analyzing project:', error);
                showNotification('Failed to analyze project: ' + error.message, 'error');
            }
        }
        
        // Utility functions
        function showNotification(message, type = 'success') {
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => notification.classList.add('show'), 100);
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }
    </script>
</body>
</html>
        """
    
    async def start_server(self):
        """Start the web server."""
        try:
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            
            self.logger.info(f"Template Editor Server started at http://{self.host}:{self.port}")
            
            # Open browser automatically
            threading.Thread(target=self._open_browser, daemon=True).start()
            
            # Keep server running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Shutting down server...")
                await runner.cleanup()
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise
    
    def _open_browser(self):
        """Open the browser after a short delay."""
        time.sleep(1)  # Wait for server to start
        webbrowser.open(f"http://{self.host}:{self.port}")


def start_template_editor(port: int = 8081, host: str = "localhost"):
    """Start the visual template editor server."""
    if not AIOHTTP_AVAILABLE:
        print("Error: aiohttp is required for the template editor.")
        print("Install with: pip install aiohttp")
        return False
    
    try:
        server = TemplateEditorServer(port=port, host=host)
        asyncio.run(server.start_server())
        return True
    except KeyboardInterrupt:
        print("\nTemplate editor stopped.")
        return True
    except Exception as e:
        print(f"Failed to start template editor: {e}")
        return False


if __name__ == "__main__":
    start_template_editor()