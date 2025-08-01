import pytest
import json
import tempfile
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from codeweaver.export.formats import (
    ExportManager, ExportOptions, ExportMetadata, FileExportInfo,
    MarkdownExporter, JSONExporter, XMLExporter, HTMLExporter,
    CSVExporter, ZipExporter, PDFExporter
)
from codeweaver.export.chunked_exporter import (
    ChunkedExporter, ChunkConfiguration, ChunkStrategy,
    ChunkMetadata, ChunkReference, create_chunked_export
)
from codeweaver.core.importance_scorer import FileImportanceInfo, FileType
from codeweaver.core.token_budget import BudgetAllocation, BudgetStrategy


class TestExportDataClasses:
    """Test export data classes."""
    
    def test_export_metadata_creation(self):
        """Test ExportMetadata dataclass creation."""
        metadata = ExportMetadata(
            project_path="/test/project",
            generated_at="2023-01-01T00:00:00Z",
            generator="CodeWeaver",
            version="1.0.0",
            total_files=10,
            total_tokens=5000,
            export_format="markdown",
            options={"test": True}
        )
        
        assert metadata.project_path == "/test/project"
        assert metadata.total_files == 10
        assert metadata.total_tokens == 5000
        assert metadata.export_format == "markdown"
        assert metadata.options["test"] is True
    
    def test_file_export_info_creation(self):
        """Test FileExportInfo dataclass creation."""
        file_info = FileExportInfo(
            path="test.py",
            content="print('hello')",
            language="python",
            tokens=50,
            size_bytes=15,
            importance_score=0.8,
            file_type="source",
            encoding="utf-8",
            content_hash="abc123"
        )
        
        assert file_info.path == "test.py"
        assert file_info.content == "print('hello')"
        assert file_info.language == "python"
        assert file_info.tokens == 50
        assert file_info.importance_score == 0.8
    
    def test_export_options_defaults(self):
        """Test ExportOptions default values."""
        options = ExportOptions()
        
        assert options.include_metadata is True
        assert options.include_file_stats is True
        assert options.include_content is True
        assert options.compress_content is False
        assert options.pretty_format is True
        assert options.custom_template is None
        assert options.template_variables is None
    
    def test_export_options_custom(self):
        """Test ExportOptions with custom values."""
        options = ExportOptions(
            include_metadata=False,
            compress_content=True,
            custom_template="custom.jinja2",
            template_variables={"title": "My Project"}
        )
        
        assert options.include_metadata is False
        assert options.compress_content is True
        assert options.custom_template == "custom.jinja2"
        assert options.template_variables["title"] == "My Project"


class TestMarkdownExporter:
    """Test Markdown export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.exporter = MarkdownExporter()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_exporter_properties(self):
        """Test exporter properties."""
        assert self.exporter.format_name == "Markdown"
        assert self.exporter.file_extension == ".md"
    
    def test_export_basic(self):
        """Test basic markdown export."""
        files = [
            FileExportInfo(
                path="main.py",
                content="def main():\n    print('Hello, World!')",
                language="python",
                tokens=20,
                size_bytes=35,
                importance_score=0.9
            ),
            FileExportInfo(
                path="utils.py",
                content="def helper():\n    return True",
                language="python",
                tokens=15,
                size_bytes=25,
                importance_score=0.7
            )
        ]
        
        metadata = ExportMetadata(
            project_path="/test/project",
            generated_at=datetime.now().isoformat(),
            total_files=2,
            total_tokens=35,
            export_format="markdown"
        )
        
        output_path = self.temp_dir / "output.md"
        success = self.exporter.export(files, metadata, output_path)
        
        assert success is True
        assert output_path.exists()
        
        content = output_path.read_text(encoding='utf-8')
        assert "# CodeWeaver Export" in content
        assert "main.py" in content
        assert "utils.py" in content
        assert "def main():" in content
        assert "def helper():" in content
        assert "```python" in content
    
    def test_export_with_metadata_disabled(self):
        """Test markdown export with metadata disabled."""
        options = ExportOptions(include_metadata=False)
        exporter = MarkdownExporter(options)
        
        files = [
            FileExportInfo(
                path="test.py",
                content="print('test')",
                language="python",
                tokens=10,
                size_bytes=12
            )
        ]
        
        metadata = ExportMetadata(
            project_path="/test",
            generated_at=datetime.now().isoformat(),
            total_files=1,
            total_tokens=10,
            export_format="markdown"
        )
        
        output_path = self.temp_dir / "no_metadata.md"
        success = exporter.export(files, metadata, output_path)
        
        assert success is True
        content = output_path.read_text(encoding='utf-8')
        
        # Should not contain metadata section
        assert "## Export Metadata" not in content
        assert "test.py" in content
    
    def test_export_empty_files(self):
        """Test markdown export with empty file list."""
        files = []
        metadata = ExportMetadata(
            project_path="/empty",
            generated_at=datetime.now().isoformat(),
            total_files=0,
            total_tokens=0,
            export_format="markdown"
        )
        
        output_path = self.temp_dir / "empty.md"
        success = self.exporter.export(files, metadata, output_path)
        
        assert success is True
        content = output_path.read_text(encoding='utf-8')
        assert "No files to export" in content


class TestJSONExporter:
    """Test JSON export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.exporter = JSONExporter()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_exporter_properties(self):
        """Test exporter properties."""
        assert self.exporter.format_name == "JSON"
        assert self.exporter.file_extension == ".json"
    
    def test_export_basic(self):
        """Test basic JSON export."""
        files = [
            FileExportInfo(
                path="config.py",
                content="CONFIG = {'debug': True}",
                language="python",
                tokens=12,
                size_bytes=25,
                importance_score=0.6
            )
        ]
        
        metadata = ExportMetadata(
            project_path="/test/project",
            generated_at="2023-01-01T00:00:00Z",
            total_files=1,
            total_tokens=12,
            export_format="json"
        )
        
        output_path = self.temp_dir / "output.json"
        success = self.exporter.export(files, metadata, output_path)
        
        assert success is True
        assert output_path.exists()
        
        # Parse and validate JSON
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "files" in data
        assert data["metadata"]["total_files"] == 1
        assert data["metadata"]["export_format"] == "json"
        assert len(data["files"]) == 1
        assert data["files"][0]["path"] == "config.py"
        assert data["files"][0]["language"] == "python"
    
    def test_export_pretty_format(self):
        """Test JSON export with pretty formatting."""
        options = ExportOptions(pretty_format=True)
        exporter = JSONExporter(options)
        
        files = [FileExportInfo(
            path="test.py", content="pass", language="python",
            tokens=5, size_bytes=4
        )]
        
        metadata = ExportMetadata(
            project_path="/test", generated_at="2023-01-01T00:00:00Z",
            total_files=1, total_tokens=5, export_format="json"
        )
        
        output_path = self.temp_dir / "pretty.json"
        success = exporter.export(files, metadata, output_path)
        
        assert success is True
        content = output_path.read_text(encoding='utf-8')
        
        # Pretty formatted JSON should have indentation
        assert "\n" in content
        assert "  " in content  # Indentation
    
    def test_export_compact_format(self):
        """Test JSON export with compact formatting."""
        options = ExportOptions(pretty_format=False)
        exporter = JSONExporter(options)
        
        files = [FileExportInfo(
            path="test.py", content="pass", language="python",
            tokens=5, size_bytes=4
        )]
        
        metadata = ExportMetadata(
            project_path="/test", generated_at="2023-01-01T00:00:00Z",
            total_files=1, total_tokens=5, export_format="json"
        )
        
        output_path = self.temp_dir / "compact.json"
        success = exporter.export(files, metadata, output_path)
        
        assert success is True
        content = output_path.read_text(encoding='utf-8')
        
        # Compact JSON should have minimal whitespace
        assert content.count('\n') <= 1  # Only the final newline


class TestXMLExporter:
    """Test XML export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.exporter = XMLExporter()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_exporter_properties(self):
        """Test exporter properties."""
        assert self.exporter.format_name == "XML"
        assert self.exporter.file_extension == ".xml"
    
    def test_export_basic(self):
        """Test basic XML export."""
        files = [
            FileExportInfo(
                path="app.py",
                content="from flask import Flask\napp = Flask(__name__)",
                language="python",
                tokens=25,
                size_bytes=45,
                importance_score=0.85
            )
        ]
        
        metadata = ExportMetadata(
            project_path="/flask/app",
            generated_at="2023-01-01T00:00:00Z",
            total_files=1,
            total_tokens=25,
            export_format="xml"
        )
        
        output_path = self.temp_dir / "output.xml"
        success = self.exporter.export(files, metadata, output_path)
        
        assert success is True
        assert output_path.exists()
        
        # Parse and validate XML
        import xml.etree.ElementTree as ET
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        assert root.tag == "codeweaver_export"
        
        metadata_elem = root.find("metadata")
        assert metadata_elem is not None
        assert metadata_elem.find("total_files").text == "1"
        
        files_elem = root.find("files")
        assert files_elem is not None
        
        file_elem = files_elem.find("file")
        assert file_elem is not None
        assert file_elem.find("path").text == "app.py"
        assert file_elem.find("language").text == "python"
    
    def test_export_special_characters(self):
        """Test XML export with special characters."""
        files = [
            FileExportInfo(
                path="special.py",
                content="# Special chars: <>&\"'\nprint('Hello & goodbye')",
                language="python",
                tokens=15,
                size_bytes=40
            )
        ]
        
        metadata = ExportMetadata(
            project_path="/test",
            generated_at="2023-01-01T00:00:00Z",
            total_files=1,
            total_tokens=15,
            export_format="xml"
        )
        
        output_path = self.temp_dir / "special.xml"
        success = self.exporter.export(files, metadata, output_path)
        
        assert success is True
        
        # Parse XML to ensure special characters are properly escaped
        import xml.etree.ElementTree as ET
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        file_elem = root.find(".//file")
        content_elem = file_elem.find("content")
        
        # Content should be properly decoded from XML
        assert "Hello & goodbye" in content_elem.text
        assert "<>" in content_elem.text


class TestCSVExporter:
    """Test CSV export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.exporter = CSVExporter()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_exporter_properties(self):
        """Test exporter properties."""
        assert self.exporter.format_name == "CSV"
        assert self.exporter.file_extension == ".csv"
    
    def test_export_basic(self):
        """Test basic CSV export."""
        files = [
            FileExportInfo(
                path="file1.py", content="print(1)", language="python",
                tokens=10, size_bytes=9, importance_score=0.8
            ),
            FileExportInfo(
                path="file2.js", content="console.log(2)", language="javascript",
                tokens=12, size_bytes=15, importance_score=0.6
            )
        ]
        
        metadata = ExportMetadata(
            project_path="/mixed/project",
            generated_at="2023-01-01T00:00:00Z",
            total_files=2,
            total_tokens=22,
            export_format="csv"
        )
        
        output_path = self.temp_dir / "output.csv"
        success = self.exporter.export(files, metadata, output_path)
        
        assert success is True
        assert output_path.exists()
        
        # Read and validate CSV
        import csv
        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        
        # Check first row
        assert rows[0]['path'] == 'file1.py'
        assert rows[0]['language'] == 'python'
        assert rows[0]['tokens'] == '10'
        assert rows[0]['importance_score'] == '0.8'
        
        # Check second row
        assert rows[1]['path'] == 'file2.js'
        assert rows[1]['language'] == 'javascript'
        assert rows[1]['tokens'] == '12'
    
    def test_export_content_handling(self):
        """Test CSV export with content that has special characters."""
        files = [
            FileExportInfo(
                path="quotes.py",
                content='print("Hello, \"world\"!")',
                language="python",
                tokens=8,
                size_bytes=20
            )
        ]
        
        metadata = ExportMetadata(
            project_path="/test",
            generated_at="2023-01-01T00:00:00Z",
            total_files=1,
            total_tokens=8,
            export_format="csv"
        )
        
        output_path = self.temp_dir / "quotes.csv"
        success = self.exporter.export(files, metadata, output_path)
        
        assert success is True
        
        # Read CSV and check content is properly escaped
        import csv
        with open(output_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1
        # Content should be properly handled in CSV
        assert 'Hello' in rows[0]['content']
        assert 'world' in rows[0]['content']


class TestZipExporter:
    """Test ZIP export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.exporter = ZipExporter()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_exporter_properties(self):
        """Test exporter properties."""
        assert self.exporter.format_name == "ZIP Archive"
        assert self.exporter.file_extension == ".zip"
    
    def test_export_basic(self):
        """Test basic ZIP export."""
        files = [
            FileExportInfo(
                path="src/main.py",
                content="def main():\n    pass",
                language="python",
                tokens=15,
                size_bytes=20
            ),
            FileExportInfo(
                path="README.md",
                content="# My Project\n\nThis is a test.",
                language="markdown",
                tokens=10,
                size_bytes=30
            )
        ]
        
        metadata = ExportMetadata(
            project_path="/my/project",
            generated_at="2023-01-01T00:00:00Z",
            total_files=2,
            total_tokens=25,
            export_format="zip"
        )
        
        output_path = self.temp_dir / "archive.zip"
        success = self.exporter.export(files, metadata, output_path)
        
        assert success is True
        assert output_path.exists()
        
        # Open and validate ZIP contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            file_list = zf.namelist()
            
            # Should contain the files and metadata
            assert "src/main.py" in file_list
            assert "README.md" in file_list
            assert "_codeweaver_metadata.json" in file_list
            
            # Check file contents
            main_content = zf.read("src/main.py").decode('utf-8')
            assert "def main():" in main_content
            
            readme_content = zf.read("README.md").decode('utf-8')
            assert "# My Project" in readme_content
            
            # Check metadata
            metadata_content = zf.read("_codeweaver_metadata.json").decode('utf-8')
            metadata_data = json.loads(metadata_content)
            assert metadata_data["total_files"] == 2
            assert metadata_data["export_format"] == "zip"
    
    def test_export_directory_structure(self):
        """Test ZIP export preserves directory structure."""
        files = [
            FileExportInfo(
                path="deep/nested/dir/file.py",
                content="# Deep file",
                language="python",
                tokens=5,
                size_bytes=11
            )
        ]
        
        metadata = ExportMetadata(
            project_path="/test",
            generated_at="2023-01-01T00:00:00Z",
            total_files=1,
            total_tokens=5,
            export_format="zip"
        )
        
        output_path = self.temp_dir / "nested.zip"
        success = self.exporter.export(files, metadata, output_path)
        
        assert success is True
        
        with zipfile.ZipFile(output_path, 'r') as zf:
            file_list = zf.namelist()
            assert "deep/nested/dir/file.py" in file_list
            
            content = zf.read("deep/nested/dir/file.py").decode('utf-8')
            assert "# Deep file" in content


class TestExportManager:
    """Test the export manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = ExportManager()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_available_formats(self):
        """Test getting available export formats."""
        formats = self.manager.get_available_formats()
        
        expected_formats = ['markdown', 'json', 'xml', 'html', 'csv', 'zip', 'pdf']
        for fmt in expected_formats:
            assert fmt in formats
        
        # Each format should have required properties
        for fmt_name, fmt_info in formats.items():
            assert 'name' in fmt_info
            assert 'extension' in fmt_info
            assert 'description' in fmt_info
    
    def test_export_files_markdown(self):
        """Test exporting files using the manager - Markdown format."""
        files = [Path("test.py")]
        project_path = Path("/test/project")
        output_path = self.temp_dir / "output.md"
        
        # Mock file reading
        with patch('pathlib.Path.read_text') as mock_read, \
             patch('pathlib.Path.stat') as mock_stat:
            
            mock_read.return_value = "print('hello')"
            mock_stat.return_value.st_size = 15
            
            success = self.manager.export_files(
                files=files,
                project_path=project_path,
                output_path=output_path,
                format_name="markdown",
                options=ExportOptions(),
                file_importance_info=[],
                budget_allocation=None
            )
            
            assert success is True
            assert output_path.exists()
    
    def test_export_files_json(self):
        """Test exporting files using the manager - JSON format."""
        files = [Path("config.py")]
        project_path = Path("/test/project")
        output_path = self.temp_dir / "output.json"
        
        with patch('pathlib.Path.read_text') as mock_read, \
             patch('pathlib.Path.stat') as mock_stat:
            
            mock_read.return_value = "CONFIG = {}"
            mock_stat.return_value.st_size = 12
            
            success = self.manager.export_files(
                files=files,
                project_path=project_path,
                output_path=output_path,
                format_name="json",
                options=ExportOptions()
            )
            
            assert success is True
            assert output_path.exists()
            
            # Validate JSON structure
            with open(output_path, 'r') as f:
                data = json.load(f)
            assert "metadata" in data
            assert "files" in data
    
    def test_export_files_unsupported_format(self):
        """Test exporting with unsupported format."""
        files = [Path("test.py")]
        project_path = Path("/test")
        output_path = self.temp_dir / "output.unknown"
        
        success = self.manager.export_files(
            files=files,
            project_path=project_path,
            output_path=output_path,
            format_name="unsupported_format"
        )
        
        assert success is False
    
    def test_export_with_importance_info(self):
        """Test export with file importance information."""
        files = [Path("important.py")]
        project_path = Path("/test/project")
        output_path = self.temp_dir / "with_importance.json"
        
        importance_info = [
            FileImportanceInfo(
                path=Path("important.py"),
                relative_path="important.py",
                file_type=FileType.CORE_LIBRARY,
                importance_score=0.95,
                tokens=100,
                language="python",
                suggestions=["Critical file"]
            )
        ]
        
        with patch('pathlib.Path.read_text') as mock_read, \
             patch('pathlib.Path.stat') as mock_stat:
            
            mock_read.return_value = "def critical_function(): pass"
            mock_stat.return_value.st_size = 30
            
            success = self.manager.export_files(
                files=files,
                project_path=project_path,
                output_path=output_path,
                format_name="json",
                file_importance_info=importance_info
            )
            
            assert success is True
            
            # Check that importance score was included
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            file_data = data["files"][0]
            assert file_data["importance_score"] == 0.95
    
    def test_export_with_budget_allocation(self):
        """Test export with budget allocation information."""
        files = [Path("budgeted.py")]
        project_path = Path("/test/project")
        output_path = self.temp_dir / "with_budget.json"
        
        budget_allocation = BudgetAllocation(
            selected_files=[],
            filtered_files=[],
            budget_used=5000,
            budget_remaining=45000,
            total_budget=50000,
            over_budget_files=0,
            under_budget_threshold=0,
            strategy=BudgetStrategy.BALANCED
        )
        
        with patch('pathlib.Path.read_text') as mock_read, \
             patch('pathlib.Path.stat') as mock_stat:
            
            mock_read.return_value = "# Budget test file"
            mock_stat.return_value.st_size = 20
            
            success = self.manager.export_files(
                files=files,
                project_path=project_path,
                output_path=output_path,
                format_name="json",
                budget_allocation=budget_allocation
            )
            
            assert success is True
            
            # Check that budget information was included in metadata
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert "budget_allocation" in data["metadata"]
            budget_data = data["metadata"]["budget_allocation"]
            assert budget_data["total_budget"] == 50000
            assert budget_data["budget_used"] == 5000
    
    def test_detect_language_from_extension(self):
        """Test language detection from file extension."""
        test_cases = {
            "test.py": "python",
            "app.js": "javascript",
            "component.tsx": "typescript",
            "Main.java": "java",
            "script.sh": "bash",
            "style.css": "css",
            "README.md": "markdown",
            "config.json": "json",
            "data.xml": "xml",
            "unknown.xyz": "text"
        }
        
        for filename, expected_language in test_cases.items():
            detected = self.manager._detect_language(Path(filename))
            assert detected == expected_language, f"Failed for {filename}: expected {expected_language}, got {detected}"


class TestChunkedExporter:
    """Test chunked export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test files
        for i in range(5):
            (self.input_dir / f"file_{i}.py").write_text(f"# File {i}\nprint('File {i}')")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_chunk_configuration_defaults(self):
        """Test ChunkConfiguration default values."""
        config = ChunkConfiguration()
        
        assert config.chunk_strategy == ChunkStrategy.BALANCED
        assert config.max_tokens_per_chunk == 50000
        assert config.max_files_per_chunk == 50
        assert config.max_size_per_chunk == 10485760  # 10MB
        assert config.generate_cross_references is True
        assert config.generate_index is True
        assert config.generate_manifest is True
    
    @patch('codeweaver.export.chunked_exporter.ChunkedExporter')
    def test_create_chunked_export(self, mock_exporter_class):
        """Test create_chunked_export function."""
        mock_exporter = MagicMock()
        mock_exporter.create_chunks.return_value = True
        mock_exporter_class.return_value = mock_exporter
        
        config = ChunkConfiguration(max_files_per_chunk=25)
        
        success = create_chunked_export(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            config=config,
            export_format="markdown",
            base_name="test_chunk"
        )
        
        assert success is True
        mock_exporter_class.assert_called_once()
        mock_exporter.create_chunks.assert_called_once()
    
    def test_chunk_strategies(self):
        """Test all chunk strategies are defined."""
        strategies = list(ChunkStrategy)
        
        expected_strategies = [
            ChunkStrategy.BY_SIZE,
            ChunkStrategy.BY_COUNT,
            ChunkStrategy.BY_DIRECTORY,
            ChunkStrategy.BY_IMPORTANCE,
            ChunkStrategy.BY_TYPE,
            ChunkStrategy.BALANCED
        ]
        
        for expected in expected_strategies:
            assert expected in strategies
    
    def test_chunk_metadata_creation(self):
        """Test ChunkMetadata dataclass."""
        metadata = ChunkMetadata(
            chunk_id="chunk_001",
            file_count=25,
            total_tokens=45000,
            total_size_bytes=1024000,
            created_at="2023-01-01T00:00:00Z",
            strategy_used=ChunkStrategy.BY_COUNT,
            cross_references=["chunk_002", "chunk_003"]
        )
        
        assert metadata.chunk_id == "chunk_001"
        assert metadata.file_count == 25
        assert metadata.strategy_used == ChunkStrategy.BY_COUNT
        assert len(metadata.cross_references) == 2
    
    def test_chunk_reference_creation(self):
        """Test ChunkReference dataclass."""
        reference = ChunkReference(
            source_chunk="chunk_001",
            target_chunk="chunk_002",
            reference_type="import",
            source_file="app.py",
            target_file="utils.py",
            line_number=5,
            context="from utils import helper"
        )
        
        assert reference.source_chunk == "chunk_001"
        assert reference.target_chunk == "chunk_002"
        assert reference.reference_type == "import"
        assert reference.line_number == 5
