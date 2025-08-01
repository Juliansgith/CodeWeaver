import re

class ContentFilter:
    """Provides methods for filtering and optimizing file content."""

    def strip_comments(self, content: str, language: str) -> str:
        """Strips comments from code based on the language."""
        if language == 'python':
            return re.sub(r'#.*?\n', '\n', content)
        if language in ['javascript', 'typescript', 'java', 'csharp', 'c', 'cpp', 'go', 'rust']:
            content = re.sub(r'//.*?\n', '\n', content)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            return content
        return content

    def optimize_whitespace(self, content: str) -> str:
        """Optimizes whitespace by removing redundant blank lines."""
        return re.sub(r'\n{3,}', '\n\n', content)

    def sample_large_file(self, content: str, language: str, max_lines: int = 1000) -> str:
        """Intelligently samples a large file."""
        lines = content.split('\n')
        if len(lines) <= max_lines:
            return content

        # Get the first 500 and last 200 lines
        sampled_lines = lines[:500]
        sampled_lines.append('\n... [content truncated] ...\n')
        sampled_lines.extend(lines[-200:])

        return '\n'.join(sampled_lines)

