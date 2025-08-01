"""
CodeWeaver CLI - Command-line interface for intelligent code packaging
"""

import click
import json
import sys
import asyncio
from pathlib import Path
import logging

# Import command groups from other modules
from .chunked_export import chunked_cli
from .embedding_config import embedding_cli

# Import necessary components from the core application
from ..core.processor import CodebaseProcessor
from ..core.models import ProcessingOptions
from ..ai.optimization_engine import OptimizationEngine
from ..export.formats import ExportManager, ExportOptions
from ..core.json_utils import safe_json_dumps, serialize_optimization_result

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(package_name="codeweaver")
def main():
    """
    CodeWeaver CLI: Intelligent code packaging for AI systems.

    A powerful tool to analyze, select, and package codebase context for
    Large Language Models (LLMs).
    """
    pass

@main.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path (default: codebase.md in project directory).')
@click.option('--format', '-f', 'export_format', type=click.Choice(['markdown', 'json', 'xml', 'html', 'csv', 'zip', 'pdf']), default='markdown', help='Output format.')
@click.option('--template', '-t', help='Template to use for file selection.')
@click.option('--budget', '-b', type=int, help='Token budget limit for smart selection.')
@click.option('--purpose', '-p', help='Describe the purpose to guide AI-powered file selection.')
@click.option('--exclude', '-e', multiple=True, help='Ignore patterns (e.g., "*/tests/*").')
@click.option('--preview', is_flag=True, help='Preview selected files without generating a digest.')
@click.option('--strip-comments', is_flag=True, help='Remove comments from code files.')
@click.option('--optimize-whitespace', is_flag=True, help='Remove extra blank lines.')
@click.option('--intelligent-sampling', is_flag=True, help='Enable intelligent sampling for large files.')
def digest(path, output, export_format, template, budget, purpose, exclude, preview, strip_comments, optimize_whitespace, intelligent_sampling):
    """Generate an intelligent codebase digest."""
    project_path = Path(path)
    logger.info(f"Starting digest for: {project_path}")

    if not purpose and not budget:
        click.echo("Warning: Running digest without a --purpose or --budget. This may include all files.", err=True)

    # Use AI Optimization Engine for purpose-driven selection
    try:
        engine = OptimizationEngine(project_path)
        
        # Get all processable files first
        processor = CodebaseProcessor()
        base_options = ProcessingOptions(
            input_dir=str(project_path),
            ignore_patterns=list(exclude),
            size_limit_mb=10.0,
            mode='preview'
        )
        file_result = processor.process(base_options)
        if not file_result.success or not file_result.files:
            logger.error("No files found to process.")
            sys.exit(1)

        # Optimize selection
        optimization_result = asyncio.run(engine.optimize_file_selection(
            purpose=purpose or "general project overview",
            available_files=file_result.files,
            token_budget=budget or 200000  # Default to a large budget if not specified
        ))

        selected_files = optimization_result.selected_files
        logger.info(f"AI selected {len(selected_files)} files based on purpose '{purpose}'.")
        
        if preview:
            click.echo("\n--- AI-Selected Files for Digest ---")
            for score in optimization_result.file_scores:
                if score.file_path in [str(p) for p in selected_files]:
                    click.echo(f"- {Path(score.file_path).relative_to(project_path)} (Score: {score.combined_score:.2f})")
            click.echo("--- End of Preview ---")
            return

        # Prepare for export
        output_path = Path(output) if output else project_path / f"codebase.{export_format}"
        export_manager = ExportManager()
        
        export_options = ExportOptions() # Use default options for now
        
        success = export_manager.export_files(
            files=selected_files,
            project_path=project_path,
            output_path=output_path,
            format_name=export_format,
            options=export_options,
            file_importance_info=optimization_result.file_scores,
            budget_allocation=optimization_result.budget_allocation
        )

        if success:
            logger.info(f"Digest successfully generated at: {output_path}")
        else:
            logger.error("Failed to generate digest.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"An error occurred during digest generation: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        logger.debug(traceback.format_exc())
        sys.exit(1)

@main.command('mcp-server')
@click.argument('path', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('--port', '-p', type=int, help='Server port (uses STDIO by default).')
@click.option('--host', default='localhost', help='Server host.')
def mcp_server_cli(path, port, host):
    """Start the Model Context Protocol (MCP) server."""
    from ..mcp.server import CodeWeaverMCPServer

    project_path = Path(path)
    logger.info(f"Starting MCP server for project: {project_path}")
    
    server = CodeWeaverMCPServer(root_path=project_path)
    
    if port:
        logger.error("HTTP server not yet implemented. Use STDIO mode.")
        sys.exit(1)
    else:
        logger.info("Starting MCP server in STDIO mode. Waiting for requests...")
        asyncio.run(server.main())

# Add command groups from other files
main.add_command(chunked_cli, name="chunked")
main.add_command(embedding_cli, name="embedding")

if __name__ == '__main__':
    main()