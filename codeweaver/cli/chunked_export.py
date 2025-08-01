"""
CLI commands for chunked exports.
"""

import click
import sys
from pathlib import Path
from typing import Optional

from ..export.chunked_exporter import ChunkConfiguration, ChunkStrategy, create_chunked_export
from ..export.formats import ExportManager, ExportOptions, FileExportInfo, ExportMetadata
from ..core.processor import CodebaseProcessor
from ..core.models import ProcessingOptions

@click.group('chunked')
def chunked_cli():
    """Export large codebases in manageable chunks with cross-references."""
    pass

@chunked_cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--format', 'export_format', default='markdown', 
              type=click.Choice(['markdown', 'json', 'html']),
              help='Export format (only formats supporting chunking)')
@click.option('--strategy', default='balanced',
              type=click.Choice(['by_size', 'by_count', 'by_directory', 'by_importance', 'by_type', 'balanced']),
              help='Strategy for splitting files into chunks')
@click.option('--max-tokens', default=50000, type=int,
              help='Maximum tokens per chunk')
@click.option('--max-files', default=50, type=int,
              help='Maximum files per chunk')
@click.option('--max-size', default=10485760, type=int,  # 10MB
              help='Maximum size per chunk in bytes')
@click.option('--no-cross-refs', is_flag=True, default=False,
              help='Disable cross-reference generation')
@click.option('--no-index', is_flag=True, default=False,
              help='Do not create index file')
@click.option('--no-manifest', is_flag=True, default=False,
              help='Do not create manifest file')
@click.option('--ignore', multiple=True,
              help='Additional ignore patterns (can be used multiple times)')
@click.option('--base-name', default='codeweaver_chunk',
              help='Base name for chunk files')
def export(input_dir: str, output_dir: str, export_format: str, strategy: str,
          max_tokens: int, max_files: int, max_size: int,
          no_cross_refs: bool, no_index: bool, no_manifest: bool,
          ignore: tuple, base_name: str):
    """Export a codebase in chunks with cross-references."""
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        click.echo(f"🔧 Starting chunked export of {input_path}")
        click.echo(f"📁 Output directory: {output_path}")
        click.echo(f"📄 Format: {export_format}")
        click.echo(f"🎯 Strategy: {strategy}")
        click.echo(f"📊 Max tokens per chunk: {max_tokens:,}")
        click.echo(f"📁 Max files per chunk: {max_files}")
        click.echo(f"💾 Max size per chunk: {max_size:,} bytes")
        
        # Create chunk configuration
        chunk_strategy = ChunkStrategy(strategy)
        chunk_config = ChunkConfiguration(
            max_tokens_per_chunk=max_tokens,
            max_files_per_chunk=max_files,
            max_size_per_chunk=max_size,
            strategy=chunk_strategy,
            include_cross_references=not no_cross_refs,
            create_index=not no_index,
            create_manifest=not no_manifest,
            output_format=export_format
        )
        
        # Process codebase to get files
        processor = CodebaseProcessor()
        ignore_patterns = list(ignore) if ignore else []
        ignore_patterns.extend([
            "*.pyc", "__pycache__", ".git", "node_modules", 
            "*.min.js", "*.bundle.js", "dist", "build",
            ".vscode", ".idea", "*.log"
        ])
        
        options = ProcessingOptions(
            input_dir=str(input_path),
            ignore_patterns=ignore_patterns,
            size_limit_mb=50.0,  # Increased for chunked exports
            mode='preview'
        )
        
        click.echo("📂 Processing codebase...")
        result = processor.process(options)
        
        if not result.success or not result.files:
            click.echo("❌ Failed to process codebase or no files found", err=True)
            sys.exit(1)
        
        click.echo(f"✅ Found {len(result.files)} files")
        
        # Prepare file export information
        file_export_infos = []
        for file_path in result.files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                relative_path = str(file_path.relative_to(input_path)).replace("\\", "/")
                language = _detect_language(file_path)
                tokens = len(content.split()) * 1.3  # Rough estimate
                
                file_info = FileExportInfo(
                    path=relative_path,
                    content=content,
                    language=language,
                    tokens=int(tokens),
                    size_bytes=len(content.encode('utf-8')),
                    importance_score=0.5,  # Default importance
                    file_type='code',
                    content_hash=_calculate_hash(content)
                )
                
                file_export_infos.append(file_info)
                
            except Exception as e:
                click.echo(f"⚠️  Failed to process {file_path}: {e}")
                continue
        
        if not file_export_infos:
            click.echo("❌ No files could be processed", err=True)
            sys.exit(1)
        
        # Create metadata
        total_tokens = sum(f.tokens for f in file_export_infos)
        metadata = ExportMetadata(
            project_path=str(input_path),
            generated_at="",  # Will be set by exporter
            total_files=len(file_export_infos),
            total_tokens=total_tokens,
            export_format=f"{export_format}_chunked"
        )
        
        click.echo(f"📊 Total files: {len(file_export_infos)}")
        click.echo(f"📊 Total tokens: {total_tokens:,}")
        click.echo(f"📊 Estimated chunks: {max(1, total_tokens // max_tokens)}")
        
        # Create chunked export
        click.echo("🚀 Creating chunked export...")
        success = create_chunked_export(file_export_infos, metadata, output_path, chunk_config, base_name)
        
        if success:
            click.echo("✅ Chunked export completed successfully!")
            click.echo(f"📁 Check output directory: {output_path}")
            if not no_index:
                click.echo(f"📖 View INDEX.md for navigation")
            if not no_manifest:
                click.echo(f"🔧 Check manifest.json for technical details")
        else:
            click.echo("❌ Chunked export failed", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Export failed: {e}", err=True)
        sys.exit(1)

@chunked_cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def analyze(input_dir: str):
    """Analyze a codebase to recommend chunking strategy."""
    try:
        input_path = Path(input_dir)
        
        click.echo(f"🔍 Analyzing codebase: {input_path}")
        
        # Process codebase
        processor = CodebaseProcessor()
        options = ProcessingOptions(
            input_dir=str(input_path),
            ignore_patterns=[
                "*.pyc", "__pycache__", ".git", "node_modules", 
                "*.min.js", "*.bundle.js", "dist", "build"
            ],
            size_limit_mb=50.0,
            mode='preview'
        )
        
        result = processor.process(options)
        
        if not result.success or not result.files:
            click.echo("❌ Failed to analyze codebase", err=True)
            sys.exit(1)
        
        # Analyze structure
        total_files = len(result.files)
        total_size = sum(f.stat().st_size for f in result.files if f.exists())
        
        # Estimate tokens
        total_tokens = 0
        languages = {}
        directories = {}
        large_files = []
        
        for file_path in result.files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    tokens = len(content.split()) * 1.3
                    total_tokens += tokens
                    
                    # Track languages
                    lang = _detect_language(file_path)
                    languages[lang] = languages.get(lang, 0) + 1
                    
                    # Track directories
                    rel_path = file_path.relative_to(input_path)
                    if len(rel_path.parts) > 1:
                        top_dir = rel_path.parts[0]
                        directories[top_dir] = directories.get(top_dir, 0) + 1
                    
                    # Track large files
                    if tokens > 5000:  # Files with > 5k tokens
                        large_files.append((str(rel_path), int(tokens)))
                        
            except:
                continue
        
        # Analysis results
        click.echo(f"\n📊 Codebase Analysis Results:")
        click.echo(f"   Total Files: {total_files}")
        click.echo(f"   Total Size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
        click.echo(f"   Total Tokens: {int(total_tokens):,}")
        
        click.echo(f"\n🗂️  Top Directories:")
        for directory, count in sorted(directories.items(), key=lambda x: x[1], reverse=True)[:5]:
            click.echo(f"   {directory}: {count} files")
        
        click.echo(f"\n💾 Languages:")
        for language, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]:
            click.echo(f"   {language}: {count} files")
        
        if large_files:
            click.echo(f"\n📄 Large Files (>5k tokens):")
            for file_path, tokens in sorted(large_files, key=lambda x: x[1], reverse=True)[:5]:
                click.echo(f"   {file_path}: {tokens:,} tokens")
        
        # Recommendations
        click.echo(f"\n💡 Chunking Recommendations:")
        
        # Recommend chunk size
        if total_tokens < 30000:
            click.echo("   ✅ Small codebase - single export recommended")
        elif total_tokens < 100000:
            click.echo("   📦 Medium codebase - 2-3 chunks recommended")
            click.echo("   🎯 Recommended strategy: by_directory")
            click.echo("   📊 Suggested max tokens per chunk: 40,000")
        else:
            estimated_chunks = max(2, total_tokens // 50000)
            click.echo(f"   📦 Large codebase - {estimated_chunks} chunks recommended")
            click.echo("   🎯 Recommended strategy: balanced")
            click.echo("   📊 Suggested max tokens per chunk: 50,000")
        
        # Strategy recommendations based on structure
        if len(directories) > 5:
            click.echo("   🗂️  Many directories detected - consider 'by_directory' strategy")
        
        if len(languages) > 3:
            click.echo("   🔤 Multiple languages detected - consider 'by_type' strategy")
        
        if len(large_files) > 10:
            click.echo("   📄 Many large files detected - consider 'by_importance' strategy")
        
        # Sample command
        max_tokens = min(50000, max(20000, total_tokens // 4)) if total_tokens > 30000 else 50000
        click.echo(f"\n🚀 Sample Command:")
        click.echo(f"   codeweaver chunked export \"{input_path}\" \"./output\" \\")
        click.echo(f"     --strategy balanced --max-tokens {int(max_tokens)} \\")
        click.echo(f"     --format markdown")
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)

def _detect_language(file_path: Path) -> str:
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
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.txt': 'text',
        '.sql': 'sql',
        '.sh': 'bash'
    }
    
    ext = file_path.suffix.lower()
    return extension_map.get(ext, 'unknown')

def _calculate_hash(content: str) -> str:
    """Calculate hash of content."""
    import hashlib
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

if __name__ == '__main__':
    chunked_cli()