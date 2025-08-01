import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import base64
from abc import ABC, abstractmethod
import zipfile
import tempfile
import os

from ..core.json_utils import safe_json_dumps, convert_for_json

from ..core.importance_scorer import FileImportanceInfo
from ..core.token_budget import BudgetAllocation


@dataclass
class ExportMetadata:
    """Metadata for exports."""
    project_path: str
    generated_at: str
    generator: str = "CodeWeaver"
    version: str = "1.0.0"
    total_files: int = 0
    total_tokens: int = 0
    export_format: str = ""
    options: Optional[Dict[str, Any]] = None


@dataclass
class FileExportInfo:
    """Information about an exported file."""
    path: str
    content: str
    language: str
    tokens: int
    size_bytes: int
    importance_score: float = 0.0
    file_type: str = "unknown"
    encoding: str = "utf-8"
    content_hash: Optional[str] = None


@dataclass
class ExportOptions:
    """Options for export formatting."""
    include_metadata: bool = True
    include_file_stats: bool = True
    include_content: bool = True
    compress_content: bool = False
    pretty_format: bool = True
    custom_template: Optional[str] = None
    template_variables: Optional[Dict[str, Any]] = None


class BaseExporter(ABC):
    """Base class for all exporters."""
    
    def __init__(self, options: ExportOptions = None):
        self.options = options or ExportOptions()
    
    @abstractmethod
    def export(self, files: List[FileExportInfo], metadata: ExportMetadata, 
               output_path: Path) -> bool:
        """Export files to the specified format."""
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Name of the export format."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for this format."""
        pass
    
    def _prepare_content(self, content: str) -> str:
        """Prepare content for export (compression, encoding, etc.)."""
        if self.options.compress_content:
            import gzip
            compressed = gzip.compress(content.encode('utf-8'))
            return base64.b64encode(compressed).decode('ascii')
        return content
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content."""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


class MarkdownExporter(BaseExporter):
    """Export to Markdown format."""
    
    @property
    def format_name(self) -> str:
        return "Markdown"
    
    @property
    def file_extension(self) -> str:
        return ".md"
    
    def export(self, files: List[FileExportInfo], metadata: ExportMetadata, 
               output_path: Path) -> bool:
        """Export to Markdown format."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Header
                f.write(f"# {Path(metadata.project_path).name} - CodeWeaver Digest\n\n")
                
                # Metadata section
                if self.options.include_metadata:
                    f.write("## Project Information\n\n")
                    f.write(f"- **Project Path:** `{metadata.project_path}`\n")
                    f.write(f"- **Generated:** {metadata.generated_at}\n")
                    f.write(f"- **Generator:** {metadata.generator} v{metadata.version}\n")
                    f.write(f"- **Total Files:** {metadata.total_files}\n")
                    f.write(f"- **Total Tokens:** {metadata.total_tokens:,}\n\n")
                
                # Statistics section
                if self.options.include_file_stats and files:
                    f.write("## File Statistics\n\n")
                    
                    # Language breakdown
                    languages = {}
                    for file_info in files:
                        lang = file_info.language or "unknown"
                        languages[lang] = languages.get(lang, 0) + 1
                    
                    f.write("### Languages\n")
                    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(files)) * 100
                        f.write(f"- **{lang.title()}:** {count} files ({percentage:.1f}%)\n")
                    f.write("\n")
                    
                    # Top files by tokens
                    top_files = sorted(files, key=lambda x: x.tokens, reverse=True)[:10]
                    f.write("### Largest Files (by tokens)\n")
                    for i, file_info in enumerate(top_files, 1):
                        f.write(f"{i}. `{file_info.path}` - {file_info.tokens:,} tokens\n")
                    f.write("\n")
                
                # Table of contents
                f.write("## Table of Contents\n\n")
                for i, file_info in enumerate(files, 1):
                    anchor = file_info.path.replace('/', '').replace('.', '').replace(' ', '-').lower()
                    f.write(f"{i}. [{file_info.path}](#{anchor})\n")
                f.write("\n---\n\n")
                
                # File contents
                if self.options.include_content:
                    f.write("## File Contents\n\n")
                    
                    for file_info in files:
                        anchor = file_info.path.replace('/', '').replace('.', '').replace(' ', '-').lower()
                        f.write(f"### `{file_info.path}` {{#{anchor}}}\n\n")
                        
                        # File metadata
                        f.write(f"**Language:** {file_info.language}  \n")
                        f.write(f"**Tokens:** {file_info.tokens:,}  \n")
                        f.write(f"**Size:** {file_info.size_bytes:,} bytes  \n")
                        if file_info.importance_score > 0:
                            f.write(f"**Importance:** {file_info.importance_score:.2f}  \n")
                        f.write("\n")
                        
                        # Code block
                        language_id = self._get_language_identifier(file_info.language)
                        content = self._prepare_content(file_info.content)
                        
                        if self.options.compress_content:
                            f.write("```base64\n")
                            f.write(content)
                            f.write("\n```\n\n")
                            f.write("*Content is compressed and base64 encoded*\n\n")
                        else:
                            f.write(f"```{language_id}\n")
                            f.write(content)
                            f.write("\n```\n\n")
                        
                        f.write("---\n\n")
            
            return True
            
        except Exception as e:
            print(f"Failed to export to Markdown: {e}")
            return False
    
    def _get_language_identifier(self, language: str) -> str:
        """Get the language identifier for syntax highlighting."""
        language_map = {
            'python': 'python',
            'javascript': 'javascript',
            'typescript': 'typescript',  
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'csharp': 'csharp',
            'go': 'go',
            'rust': 'rust',
            'php': 'php',
            'ruby': 'ruby',
            'swift': 'swift',
            'kotlin': 'kotlin',
            'scala': 'scala',
            'html': 'html',
            'css': 'css',
            'scss': 'scss',
            'xml': 'xml',
            'json': 'json',
            'yaml': 'yaml',
            'markdown': 'markdown',
            'bash': 'bash',
            'sql': 'sql'
        }
        return language_map.get(language.lower(), 'text')


class JSONExporter(BaseExporter):
    """Export to JSON format."""
    
    @property
    def format_name(self) -> str:
        return "JSON"
    
    @property
    def file_extension(self) -> str:
        return ".json"
    
    def export(self, files: List[FileExportInfo], metadata: ExportMetadata, 
               output_path: Path) -> bool:
        """Export to JSON format."""
        try:
            # Build export data
            export_data = {
                "metadata": convert_for_json(asdict(metadata)),
                "files": []
            }
            
            for file_info in files:
                file_data = {
                    "path": file_info.path,
                    "language": file_info.language,
                    "tokens": file_info.tokens,
                    "size_bytes": file_info.size_bytes,
                    "importance_score": file_info.importance_score,
                    "file_type": file_info.file_type,
                    "encoding": file_info.encoding
                }
                
                if self.options.include_content:
                    content = self._prepare_content(file_info.content)
                    file_data["content"] = content
                    file_data["content_compressed"] = self.options.compress_content
                
                if file_info.content_hash:
                    file_data["content_hash"] = file_info.content_hash
                
                export_data["files"].append(file_data)
            
            # Write JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                if self.options.pretty_format:
                    f.write(safe_json_dumps(export_data, indent=2, ensure_ascii=False))
                else:
                    f.write(safe_json_dumps(export_data, separators=(',', ':'), ensure_ascii=False))
            
            return True
            
        except Exception as e:
            print(f"Failed to export to JSON: {e}")
            return False


class XMLExporter(BaseExporter):
    """Export to XML format."""
    
    @property
    def format_name(self) -> str:
        return "XML"
    
    @property
    def file_extension(self) -> str:
        return ".xml"
    
    def export(self, files: List[FileExportInfo], metadata: ExportMetadata, 
               output_path: Path) -> bool:
        """Export to XML format."""
        try:
            # Create root element
            root = ET.Element("codeweaver_digest")
            
            # Metadata
            if self.options.include_metadata:
                metadata_elem = ET.SubElement(root, "metadata")
                for key, value in asdict(metadata).items():
                    if value is not None:
                        elem = ET.SubElement(metadata_elem, key)
                        elem.text = str(value)
            
            # Files
            files_elem = ET.SubElement(root, "files")
            
            for file_info in files:
                file_elem = ET.SubElement(files_elem, "file")
                file_elem.set("path", file_info.path)
                file_elem.set("language", file_info.language or "unknown")
                file_elem.set("tokens", str(file_info.tokens))
                file_elem.set("size_bytes", str(file_info.size_bytes))
                file_elem.set("importance_score", str(file_info.importance_score))
                file_elem.set("file_type", file_info.file_type)
                
                if self.options.include_content:
                    content_elem = ET.SubElement(file_elem, "content")
                    content = self._prepare_content(file_info.content)
                    
                    if self.options.compress_content:
                        content_elem.set("encoding", "base64")
                        content_elem.set("compressed", "true")
                    else:
                        content_elem.set("encoding", file_info.encoding)
                    
                    content_elem.text = content
            
            # Write XML file
            tree = ET.ElementTree(root)
            if self.options.pretty_format:
                self._indent_xml(root)
            
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            
            return True
            
        except Exception as e:
            print(f"Failed to export to XML: {e}")
            return False
    
    def _indent_xml(self, elem, level=0):
        """Add indentation to XML for pretty printing."""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent_xml(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


class HTMLExporter(BaseExporter):
    """Export to HTML format."""
    
    @property
    def format_name(self) -> str:
        return "HTML"
    
    @property
    def file_extension(self) -> str:
        return ".html"
    
    def export(self, files: List[FileExportInfo], metadata: ExportMetadata, 
               output_path: Path) -> bool:
        """Export to HTML format."""
        try:
            html_content = self._generate_html(files, metadata)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
            
        except Exception as e:
            print(f"Failed to export to HTML: {e}")
            return False
    
    def _generate_html(self, files: List[FileExportInfo], metadata: ExportMetadata) -> str:
        """Generate HTML content."""
        project_name = Path(metadata.project_path).name
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - CodeWeaver Digest</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{project_name}</h1>
            <p class="subtitle">CodeWeaver Digest</p>
        </header>
        
        <nav class="metadata">
"""
        
        if self.options.include_metadata:
            html += f"""
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Project Path:</strong> <code>{metadata.project_path}</code>
                </div>
                <div class="metadata-item">
                    <strong>Generated:</strong> {metadata.generated_at}
                </div>
                <div class="metadata-item">
                    <strong>Total Files:</strong> {metadata.total_files}
                </div>
                <div class="metadata-item">
                    <strong>Total Tokens:</strong> {metadata.total_tokens:,}
                </div>
            </div>
"""
        
        html += """
        </nav>
        
        <nav class="toc">
            <h2>Table of Contents</h2>
            <ul>
"""
        
        for i, file_info in enumerate(files, 1):
            file_id = f"file-{i}"
            html += f'                <li><a href="#{file_id}">{file_info.path}</a> <span class="tokens">({file_info.tokens:,} tokens)</span></li>\n'
        
        html += """
            </ul>
        </nav>
        
        <main class="files">
"""
        
        if self.options.include_content:
            for i, file_info in enumerate(files, 1):
                file_id = f"file-{i}"
                language_class = self._get_language_class(file_info.language)
                
                html += f"""
            <section id="{file_id}" class="file">
                <header class="file-header">
                    <h3>{file_info.path}</h3>
                    <div class="file-stats">
                        <span class="language">{file_info.language}</span>
                        <span class="tokens">{file_info.tokens:,} tokens</span>
                        <span class="size">{file_info.size_bytes:,} bytes</span>
"""
                
                if file_info.importance_score > 0:
                    html += f'                        <span class="importance">Importance: {file_info.importance_score:.2f}</span>\n'
                
                html += """
                    </div>
                </header>
                <div class="file-content">
"""
                
                content = self._prepare_content(file_info.content)
                if self.options.compress_content:
                    html += f'                    <pre class="compressed">Content compressed and base64 encoded:\n{content}</pre>\n'
                else:
                    escaped_content = self._escape_html(content)
                    html += f'                    <pre class="code {language_class}"><code>{escaped_content}</code></pre>\n'
                
                html += """
                </div>
            </section>
"""
        
        html += """
        </main>
        
        <footer>
            <p>Generated by CodeWeaver</p>
        </footer>
    </div>
    
    <script>
        // Add syntax highlighting if available
        if (typeof hljs !== 'undefined') {
            hljs.highlightAll();
        }
        
        // Add file navigation
        document.querySelectorAll('.toc a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });
    </script>
</body>
</html>"""
        
        return html
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML export."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        header h1 {
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #7f8c8d;
            font-size: 1.2em;
        }
        
        .metadata {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .metadata-item {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        
        .toc {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .toc h2 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        .toc ul {
            list-style: none;
        }
        
        .toc li {
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .toc a {
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }
        
        .toc a:hover {
            text-decoration: underline;
        }
        
        .tokens {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .file {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            overflow: hidden;
        }
        
        .file-header {
            background: #34495e;
            color: white;
            padding: 20px;
        }
        
        .file-header h3 {
            font-size: 1.3em;
            margin-bottom: 10px;
        }
        
        .file-stats {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .file-stats span {
            background: rgba(255,255,255,0.2);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        .file-content {
            padding: 0;
        }
        
        .code {
            background: #f8f9fa;
            padding: 20px;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 14px;
            line-height: 1.4;
            margin: 0;
        }
        
        .compressed {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        
        code {
            font-family: inherit;
        }
        
        footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 50px;
            padding: 20px;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            header h1 {
                font-size: 2em;
            }
            
            .metadata-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _get_language_class(self, language: str) -> str:
        """Get CSS class for language syntax highlighting."""
        return f"language-{language.lower()}" if language else "language-text"
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))


class CSVExporter(BaseExporter):
    """Export to CSV format (file metadata only)."""
    
    @property
    def format_name(self) -> str:
        return "CSV"
    
    @property
    def file_extension(self) -> str:
        return ".csv"
    
    def export(self, files: List[FileExportInfo], metadata: ExportMetadata, 
               output_path: Path) -> bool:
        """Export to CSV format."""
        try:
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                headers = ['path', 'language', 'tokens', 'size_bytes', 
                          'importance_score', 'file_type']
                if self.options.include_content:
                    headers.append('content_length')
                    headers.append('content_hash')
                
                writer.writerow(headers)
                
                # Data rows
                for file_info in files:
                    row = [
                        file_info.path,
                        file_info.language or 'unknown',
                        file_info.tokens,
                        file_info.size_bytes,
                        file_info.importance_score,
                        file_info.file_type
                    ]
                    
                    if self.options.include_content:
                        row.append(len(file_info.content))
                        row.append(file_info.content_hash or '')
                    
                    writer.writerow(row)
            
            return True
            
        except Exception as e:
            print(f"Failed to export to CSV: {e}")
            return False


class ExportManager:
    """Manages different export formats."""
    
    def __init__(self):
        self.exporters = {
            'markdown': MarkdownExporter,
            'json': JSONExporter,
            'xml': XMLExporter,
            'html': HTMLExporter,
            'csv': CSVExporter,
            'zip': ZipExporter,
            'pdf': PDFExporter
        }
        self.supports_chunking = {  # Formats that support chunked exports
            'markdown', 'json', 'html'
        }
    
    def get_available_formats(self) -> List[str]:
        """Get list of available export formats."""
        return list(self.exporters.keys())
    
    def supports_chunked_export(self, format_name: str) -> bool:
        """Check if a format supports chunked exports."""
        return format_name in self.supports_chunking
    
    def export_files_chunked(self, files: List[Path], project_path: Path,
                           output_dir: Path, format_name: str,
                           chunk_config: Optional[Any] = None,
                           options: ExportOptions = None,
                           budget_allocation: Optional[BudgetAllocation] = None,
                           file_importance_info: Optional[List[FileImportanceInfo]] = None) -> bool:
        """
        Export files using chunked export for large codebases.
        
        Args:
            files: List of file paths to export
            project_path: Root project path
            output_dir: Output directory for chunks
            format_name: Export format name
            chunk_config: Chunked export configuration
            options: Export options
            budget_allocation: Optional budget allocation info
            file_importance_info: Optional file importance information
        """
        if not self.supports_chunked_export(format_name):
            raise ValueError(f"Format {format_name} does not support chunked exports")
        
        try:
            # Import here to avoid circular imports
            from .chunked_exporter import ChunkedExporter, ChunkConfiguration
            
            # Use provided config or create default
            if chunk_config is None:
                chunk_config = ChunkConfiguration(output_format=format_name)
            elif not hasattr(chunk_config, 'output_format'):
                chunk_config.output_format = format_name
            
            options = options or ExportOptions()
            
            # Prepare file information (similar to regular export)
            file_export_infos = []
            importance_map = {}
            
            if file_importance_info:
                importance_map = {info.relative_path: info for info in file_importance_info}
            
            for file_path in files:
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Get relative path
                    try:
                        relative_path = str(file_path.relative_to(project_path)).replace("\\", "/")
                    except ValueError:
                        relative_path = str(file_path)
                    
                    # Get importance info if available
                    importance_info = importance_map.get(relative_path)
                    
                    # Detect language
                    language = self._detect_language(file_path)
                    
                    # Estimate tokens (simple approximation)
                    tokens = len(content.split()) * 1.3
                    
                    # Create file export info
                    file_info = FileExportInfo(
                        path=relative_path,
                        content=content,
                        language=language,
                        tokens=int(tokens),
                        size_bytes=len(content.encode('utf-8')),
                        importance_score=importance_info.importance_score if importance_info else 0.0,
                        file_type=importance_info.file_type.value if importance_info else 'unknown',
                        content_hash=self._calculate_content_hash(content)
                    )
                    
                    file_export_infos.append(file_info)
                    
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
                    continue
            
            # Create metadata
            metadata = ExportMetadata(
                project_path=str(project_path),
                generated_at=datetime.now().isoformat(),
                total_files=len(file_export_infos),
                total_tokens=sum(f.tokens for f in file_export_infos),
                export_format=f"{format_name}_chunked",
                options=asdict(options) if options else None
            )
            
            # Add budget allocation info to metadata if available
            if budget_allocation:
                metadata.options = metadata.options or {}
                metadata.options['budget_info'] = {
                    'total_budget': budget_allocation.total_budget,
                    'used_tokens': budget_allocation.used_tokens,
                    'budget_utilization': budget_allocation.budget_utilization,
                    'efficiency_score': budget_allocation.efficiency_score,
                    'coverage_score': budget_allocation.coverage_score,
                    'strategy_used': budget_allocation.strategy_used.value
                }
            
            # Create chunked exporter and export
            exporter = ChunkedExporter(chunk_config)
            return exporter.export_chunked(file_export_infos, metadata, output_dir)
            
        except Exception as e:
            print(f"Failed to create chunked export: {e}")
            return False
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content."""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def export_files(self, files: List[Path], project_path: Path, 
                    output_path: Path, format_name: str,
                    options: ExportOptions = None,
                    budget_allocation: Optional[BudgetAllocation] = None,
                    file_importance_info: Optional[List[FileImportanceInfo]] = None) -> bool:
        """
        Export files to the specified format.
        
        Args:
            files: List of file paths to export
            project_path: Root project path
            output_path: Output file path
            format_name: Export format name
            options: Export options
            budget_allocation: Optional budget allocation info
            file_importance_info: Optional file importance information
        """
        if format_name not in self.exporters:
            raise ValueError(f"Unsupported format: {format_name}")
        
        options = options or ExportOptions()
        
        # Create exporter
        exporter_class = self.exporters[format_name]
        exporter = exporter_class(options)
        
        # Ensure output path has correct extension
        if not output_path.suffix:
            output_path = output_path.with_suffix(exporter.file_extension)
        
        # Prepare file information
        file_export_infos = []
        importance_map = {}
        
        if file_importance_info:
            importance_map = {info.relative_path: info for info in file_importance_info}
        
        for file_path in files:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Get relative path
                try:
                    relative_path = str(file_path.relative_to(project_path)).replace("\\", "/")
                except ValueError:
                    relative_path = str(file_path)
                
                # Get importance info if available
                importance_info = importance_map.get(relative_path)
                
                # Detect language
                language = self._detect_language(file_path)
                
                # Estimate tokens (simple approximation)
                tokens = len(content.split()) * 1.3
                
                # Create file export info
                file_info = FileExportInfo(
                    path=relative_path,
                    content=content,
                    language=language,
                    tokens=int(tokens),
                    size_bytes=len(content.encode('utf-8')),
                    importance_score=importance_info.importance_score if importance_info else 0.0,
                    file_type=importance_info.file_type.value if importance_info else 'unknown',
                    content_hash=exporter._calculate_content_hash(content)
                )
                
                file_export_infos.append(file_info)
                
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                continue
        
        # Create metadata
        metadata = ExportMetadata(
            project_path=str(project_path),
            generated_at=datetime.now().isoformat(),
            total_files=len(file_export_infos),
            total_tokens=sum(f.tokens for f in file_export_infos),
            export_format=format_name,
            options=asdict(options) if options else None
        )
        
        # Add budget allocation info to metadata if available
        if budget_allocation:
            metadata.options = metadata.options or {}
            metadata.options['budget_info'] = {
                'total_budget': budget_allocation.total_budget,
                'used_tokens': budget_allocation.used_tokens,
                'budget_utilization': budget_allocation.budget_utilization,
                'efficiency_score': budget_allocation.efficiency_score,
                'coverage_score': budget_allocation.coverage_score,
                'strategy_used': budget_allocation.strategy_used.value
            }
        
        # Perform export
        return exporter.export(file_export_infos, metadata, output_path)
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.xml': 'xml',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bash': 'bash'
        }
        
        ext = file_path.suffix.lower()
        return extension_map.get(ext, 'unknown')


class ZipExporter(BaseExporter):
    """Export to ZIP archive format with organized folder structure."""
    
    @property
    def format_name(self) -> str:
        return "ZIP"
    
    @property
    def file_extension(self) -> str:
        return ".zip"
    
    def export(self, files: List[FileExportInfo], metadata: ExportMetadata, 
               output_path: Path) -> bool:
        """Export to ZIP archive format."""
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add metadata file
                if self.options.include_metadata:
                    metadata_content = safe_json_dumps(convert_for_json(asdict(metadata)), indent=2)
                    zipf.writestr("metadata.json", metadata_content)
                
                # Add README file
                readme_content = self._generate_readme(files, metadata)
                zipf.writestr("README.md", readme_content)
                
                # Add files with preserved directory structure
                if self.options.include_content:
                    for file_info in files:
                        # Preserve original directory structure
                        archive_path = f"source/{file_info.path}"
                        
                        # Create content
                        content = self._prepare_content(file_info.content)
                        if self.options.compress_content:
                            # For compressed content, save as .b64 file
                            archive_path += ".b64"
                            zipf.writestr(archive_path, content)
                        else:
                            zipf.writestr(archive_path, content)
                
                # Add file statistics as CSV
                if self.options.include_file_stats:
                    stats_content = self._generate_file_stats_csv(files)
                    zipf.writestr("file_statistics.csv", stats_content)
                
                # Add HTML index for browsing
                html_index = self._generate_html_index(files, metadata)
                zipf.writestr("index.html", html_index)
            
            return True
            
        except Exception as e:
            print(f"Failed to export to ZIP: {e}")
            return False
    
    def _generate_readme(self, files: List[FileExportInfo], metadata: ExportMetadata) -> str:
        """Generate README content for the ZIP archive."""
        project_name = Path(metadata.project_path).name
        
        readme = f"""# {project_name} - CodeWeaver Archive

This archive contains a CodeWeaver digest of the project with preserved directory structure.

## Contents

- `source/` - Original source files with directory structure preserved
- `metadata.json` - Project metadata and export information  
- `file_statistics.csv` - Detailed statistics for all files
- `index.html` - Interactive HTML viewer for browsing the code
- `README.md` - This file

## Statistics

- **Total Files:** {metadata.total_files}
- **Total Tokens:** {metadata.total_tokens:,}
- **Generated:** {metadata.generated_at}
- **Export Format:** {metadata.export_format}

## File Breakdown

"""
        
        # Language breakdown
        languages = {}
        for file_info in files:
            lang = file_info.language or "unknown"
            languages[lang] = languages.get(lang, 0) + 1
        
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(files)) * 100
            readme += f"- **{lang.title()}:** {count} files ({percentage:.1f}%)\n"
        
        readme += f"""

## Usage

1. **Browse Code:** Open `index.html` in your web browser for an interactive view
2. **Access Files:** Navigate the `source/` directory to access individual files
3. **View Statistics:** Open `file_statistics.csv` for detailed file metrics
4. **Metadata:** Check `metadata.json` for export settings and project information

Generated by CodeWeaver
"""
        
        return readme
    
    def _generate_file_stats_csv(self, files: List[FileExportInfo]) -> str:
        """Generate CSV content with file statistics."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['path', 'language', 'tokens', 'size_bytes', 
                        'importance_score', 'file_type', 'content_hash'])
        
        # Data rows
        for file_info in files:
            writer.writerow([
                file_info.path,
                file_info.language or 'unknown',
                file_info.tokens,
                file_info.size_bytes,
                file_info.importance_score,
                file_info.file_type,
                file_info.content_hash or ''
            ])
        
        return output.getvalue()
    
    def _generate_html_index(self, files: List[FileExportInfo], metadata: ExportMetadata) -> str:
        """Generate HTML index for browsing the archive."""
        project_name = Path(metadata.project_path).name
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - CodeWeaver Archive</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .file-list {{ max-width: 100%; }}
        .file-item {{ padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }}
        .file-path {{ font-family: monospace; color: #0066cc; }}
        .file-meta {{ color: #666; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        input[type="text"] {{ width: 100%; padding: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{project_name}</h1>
        <p>CodeWeaver Archive - {metadata.total_files} files, {metadata.total_tokens:,} tokens</p>
        <p>Generated: {metadata.generated_at}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>Total Files</h3>
            <div style="font-size: 2em; color: #0066cc;">{metadata.total_files}</div>
        </div>
        <div class="stat-card">
            <h3>Total Tokens</h3>
            <div style="font-size: 2em; color: #0066cc;">{metadata.total_tokens:,}</div>
        </div>
        <div class="stat-card">
            <h3>Total Size</h3>
            <div style="font-size: 2em; color: #0066cc;">{sum(f.size_bytes for f in files):,} bytes</div>
        </div>
    </div>
    
    <input type="text" id="search" placeholder="Search files..." onkeyup="filterFiles()">
    
    <table id="fileTable">
        <thead>
            <tr>
                <th>File Path</th>
                <th>Language</th>
                <th>Tokens</th>
                <th>Size</th>
                <th>Importance</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for file_info in files:
            html += f"""
            <tr>
                <td><a href="source/{file_info.path}" class="file-path">{file_info.path}</a></td>
                <td>{file_info.language}</td>
                <td>{file_info.tokens:,}</td>
                <td>{file_info.size_bytes:,}</td>
                <td>{file_info.importance_score:.2f}</td>
            </tr>
"""
        
        html += """
        </tbody>
    </table>
    
    <script>
        function filterFiles() {
            const input = document.getElementById('search');
            const filter = input.value.toLowerCase();
            const table = document.getElementById('fileTable');
            const rows = table.getElementsByTagName('tr');
            
            for (let i = 1; i < rows.length; i++) {
                const cells = rows[i].getElementsByTagName('td');
                const filePath = cells[0].textContent.toLowerCase();
                
                if (filePath.indexOf(filter) > -1) {
                    rows[i].style.display = '';
                } else {
                    rows[i].style.display = 'none';
                }
            }
        }
    </script>
</body>
</html>"""
        
        return html


class PDFExporter(BaseExporter):
    """Export to PDF format."""
    
    @property
    def format_name(self) -> str:
        return "PDF"
    
    @property
    def file_extension(self) -> str:
        return ".pdf"
    
    def export(self, files: List[FileExportInfo], metadata: ExportMetadata, 
               output_path: Path) -> bool:
        """Export to PDF format."""
        try:
            # Try to import reportlab
            try:
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                from reportlab.lib.enums import TA_CENTER, TA_LEFT
            except ImportError:
                print("PDF export requires reportlab. Install with: pip install reportlab")
                return False
            
            # Create PDF document
            doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            
            # Container for the 'Flowable' objects
            story = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=20,
                alignment=TA_CENTER
            )
            
            # Title page
            project_name = Path(metadata.project_path).name
            story.append(Paragraph(f"{project_name}", title_style))
            story.append(Paragraph("CodeWeaver Digest", subtitle_style))
            story.append(Spacer(1, 20))
            
            # Metadata table
            if self.options.include_metadata:
                metadata_data = [
                    ['Project Path', metadata.project_path],
                    ['Generated', metadata.generated_at],
                    ['Total Files', str(metadata.total_files)],
                    ['Total Tokens', f"{metadata.total_tokens:,}"],
                    ['Generator', f"{metadata.generator} v{metadata.version}"]
                ]
                
                metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
                metadata_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(metadata_table)
                story.append(Spacer(1, 30))
            
            # File statistics
            if self.options.include_file_stats and files:
                story.append(Paragraph("File Statistics", styles['Heading2']))
                
                # Language breakdown
                languages = {}
                for file_info in files:
                    lang = file_info.language or "unknown"
                    languages[lang] = languages.get(lang, 0) + 1
                
                lang_data = [['Language', 'Files', 'Percentage']]
                for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(files)) * 100
                    lang_data.append([lang.title(), str(count), f"{percentage:.1f}%"])
                
                lang_table = Table(lang_data, colWidths=[2*inch, 1*inch, 1*inch])
                lang_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(lang_table)
                story.append(PageBreak())
            
            # Table of contents
            story.append(Paragraph("Table of Contents", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for i, file_info in enumerate(files, 1):
                toc_entry = f"{i}. {file_info.path} ({file_info.tokens:,} tokens)"
                story.append(Paragraph(toc_entry, styles['Normal']))
            
            story.append(PageBreak())
            
            # File contents
            if self.options.include_content:
                code_style = ParagraphStyle(
                    'Code',
                    parent=styles['Normal'],
                    fontName='Courier',
                    fontSize=8,
                    leading=10,
                    spaceAfter=12
                )
                
                for i, file_info in enumerate(files, 1):
                    # File header
                    story.append(Paragraph(f"File {i}: {file_info.path}", styles['Heading3']))
                    
                    # File metadata
                    file_meta = f"Language: {file_info.language} | Tokens: {file_info.tokens:,} | Size: {file_info.size_bytes:,} bytes"
                    if file_info.importance_score > 0:
                        file_meta += f" | Importance: {file_info.importance_score:.2f}"
                    
                    story.append(Paragraph(file_meta, styles['Normal']))
                    story.append(Spacer(1, 12))
                    
                    # File content
                    if self.options.compress_content:
                        content = self._prepare_content(file_info.content)
                        story.append(Paragraph("Content is compressed and base64 encoded:", styles['Normal']))
                        story.append(Preformatted(content[:1000] + "..." if len(content) > 1000 else content, code_style))
                    else:
                        # Limit content length for PDF
                        content = file_info.content
                        if len(content) > 5000:  # Limit to ~5000 chars per file
                            content = content[:5000] + "\n... (content truncated for PDF)"
                        
                        # Split into smaller chunks to avoid ReportLab issues
                        lines = content.split('\n')
                        chunk_size = 50  # Lines per chunk
                        
                        for j in range(0, len(lines), chunk_size):
                            chunk = '\n'.join(lines[j:j + chunk_size])
                            story.append(Preformatted(chunk, code_style))
                    
                    story.append(PageBreak())
            
            # Build PDF
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"Failed to export to PDF: {e}")
            return False