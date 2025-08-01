import asyncio
import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import sqlite3
import pickle
from abc import ABC, abstractmethod

from ..core.tokenizer import TokenEstimator, LLMProvider
from ..config.embedding_config import get_embedding_config, EmbeddingProvider, EmbeddingConfig
from .cost_tracker import get_cost_tracker, AIProvider


class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum tokens supported by the model."""
        pass


@dataclass
class EmbeddingMetadata:
    """Metadata for a code embedding."""
    file_path: str
    content_hash: str
    embedding_model: str
    created_at: float
    tokens: int
    file_type: str
    language: str
    size_bytes: int


@dataclass
class CodeChunk:
    """Represents a chunk of code with its embedding."""
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'full_file', 'section'
    language: str
    tokens: int
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SemanticMatch:
    """Represents a semantic match result."""
    file_path: str
    similarity_score: float
    chunk: CodeChunk
    match_reason: str
    context_window: Optional[str] = None
    full_chunk_content: Optional[str] = None


class EmbeddingCache:
    """SQLite-based cache for embeddings."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = cache_dir / "embeddings.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    tokens INTEGER NOT NULL,
                    file_type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    embedding_data BLOB NOT NULL,
                    chunk_metadata TEXT,
                    UNIQUE(file_path, content_hash, embedding_model)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path ON embeddings(file_path)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash ON embeddings(content_hash)
            """)
    
    def get_embedding(self, file_path: str, content_hash: str, model: str) -> Optional[Tuple[np.ndarray, EmbeddingMetadata]]:
        """Get cached embedding if available."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT embedding_data, tokens, file_type, language, size_bytes, created_at, chunk_metadata
                FROM embeddings 
                WHERE file_path = ? AND content_hash = ? AND embedding_model = ?
            """, (file_path, content_hash, model))
            
            row = cursor.fetchone()
            if row:
                embedding_data, tokens, file_type, language, size_bytes, created_at, chunk_metadata = row
                embedding = pickle.loads(embedding_data)
                
                metadata = EmbeddingMetadata(
                    file_path=file_path,
                    content_hash=content_hash,
                    embedding_model=model,
                    created_at=created_at,
                    tokens=tokens,
                    file_type=file_type,
                    language=language,
                    size_bytes=size_bytes
                )
                
                return embedding, metadata
        
        return None
    
    def store_embedding(self, file_path: str, content_hash: str, model: str,
                       embedding: np.ndarray, metadata: EmbeddingMetadata,
                       chunk_metadata: Optional[Dict[str, Any]] = None):
        """Store embedding in cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO embeddings 
                (file_path, content_hash, embedding_model, created_at, tokens, 
                 file_type, language, size_bytes, embedding_data, chunk_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_path, content_hash, model, metadata.created_at,
                metadata.tokens, metadata.file_type, metadata.language,
                metadata.size_bytes, pickle.dumps(embedding),
                json.dumps(chunk_metadata) if chunk_metadata else None
            ))
    
    def cleanup_old_embeddings(self, max_age_days: int = 30):
        """Remove old embeddings from cache."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM embeddings WHERE created_at < ?", (cutoff_time,))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM embeddings")
            count, total_size = cursor.fetchone()
            
            cursor = conn.execute("""
                SELECT embedding_model, COUNT(*) 
                FROM embeddings 
                GROUP BY embedding_model
            """)
            by_model = dict(cursor.fetchall())
            
            return {
                'total_embeddings': count or 0,
                'total_size_bytes': total_size or 0,
                'embeddings_by_model': by_model,
                'cache_file_size': self.db_path.stat().st_size if self.db_path.exists() else 0
            }


class GeminiEmbeddingService(BaseEmbeddingService):
    """
    Service for generating embeddings using Google Gemini API.
    Uses configurable Gemini embedding models.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None, cache_dir: Optional[Path] = None):
        self.config = config or get_embedding_config().get_config(EmbeddingProvider.GEMINI)
        self.cache = EmbeddingCache(cache_dir or Path.home() / '.codeweaver' / 'embeddings')
        self.api_key = self.config.api_key
        
        # Rate limiting
        self.max_requests_per_minute = self.config.rate_limit_per_minute
        self.request_timestamps: List[float] = []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            'provider': 'gemini',
            'model_name': self.config.model_name,
            'embedding_dimensions': self.config.embedding_dimensions,
            'max_tokens': self.config.max_tokens,
            'batch_size': self.config.batch_size
        }
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens supported by the model."""
        return self.config.max_tokens
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        import os
        return os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    async def generate_embedding(self, text: str, title: str = "") -> np.ndarray:
        """Generate embedding for text using Gemini API."""
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable.")
        
        # Rate limiting
        await self._check_rate_limit()
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Use the embedding model
            result = genai.embed_content(
                model=self.config.model_name,
                content=text,
                title=title,
                task_type="retrieval_document"
            )
            
            embedding = np.array(result['embedding'], dtype=np.float32)
            self.request_timestamps.append(time.time())
            
            return embedding
            
        except ImportError:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        now = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for caching."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    async def embed_code_file(self, file_path: Path, max_chunk_size: int = 4000) -> List[CodeChunk]:
        """
        Embed a code file, potentially chunking it if it's too large.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file {file_path}: {e}")
        
        # Detect language
        language = self._detect_language(file_path)
        
        # Calculate content hash
        content_hash = self._calculate_content_hash(content)
        
        # Check cache first
        relative_path = str(file_path)
        cached = self.cache.get_embedding(relative_path, content_hash, self.config.model_name)
        
        if cached:
            embedding, metadata = cached
            return [CodeChunk(
                file_path=relative_path,
                content=content,
                start_line=1,
                end_line=len(content.split('\n')),
                chunk_type='full_file',
                language=language,
                tokens=metadata.tokens,
                embedding=embedding
            )]
        
        # Estimate tokens
        token_estimates = TokenEstimator.estimate_tokens(content, LLMProvider.GEMINI)
        tokens = token_estimates.get("gemini-pro", len(content.split()) * 1.3)
        
        # Decide whether to chunk
        if tokens <= max_chunk_size:
            # Small file - embed as single chunk
            chunks = await self._embed_single_chunk(
                file_path, content, language, content_hash, int(tokens)
            )
        else:
            # Large file - smart chunking
            chunks = await self._embed_with_chunking(
                file_path, content, language, content_hash, max_chunk_size
            )
        
        return chunks
    
    async def _embed_single_chunk(self, file_path: Path, content: str, 
                                language: str, content_hash: str, tokens: int) -> List[CodeChunk]:
        """Embed a single chunk of code."""
        # Generate title for better embedding
        title = f"{file_path.name} - {language} source code"
        
        # Generate embedding
        embedding = await self.generate_embedding(content, title)
        
        # Store in cache
        metadata = EmbeddingMetadata(
            file_path=str(file_path),
            content_hash=content_hash,
            embedding_model=self.config.model_name,
            created_at=time.time(),
            tokens=tokens,
            file_type=file_path.suffix[1:] if file_path.suffix else 'unknown',
            language=language,
            size_bytes=len(content.encode('utf-8'))
        )
        
        self.cache.store_embedding(
            str(file_path), content_hash, self.config.model_name, embedding, metadata
        )
        
        return [CodeChunk(
            file_path=str(file_path),
            content=content,
            start_line=1,
            end_line=len(content.split('\n')),
            chunk_type='full_file',
            language=language,
            tokens=tokens,
            embedding=embedding
        )]
    
    async def _embed_with_chunking(self, file_path: Path, content: str,
                                 language: str, content_hash: str, max_chunk_size: int) -> List[CodeChunk]:
        """Embed code with intelligent chunking."""
        chunks = []
        
        # Try to parse code structure for intelligent chunking
        parsed_chunks = self._parse_code_structure(content, language)
        
        if not parsed_chunks:
            # Fall back to simple line-based chunking
            parsed_chunks = self._chunk_by_lines(content, max_chunk_size)
        
        # Embed each chunk
        for i, (chunk_content, start_line, end_line, chunk_type) in enumerate(parsed_chunks):
            title = f"{file_path.name} - {chunk_type} (lines {start_line}-{end_line})"
            
            try:
                embedding = await self.generate_embedding(chunk_content, title)
                tokens = len(chunk_content.split()) * 1.3  # Rough estimate
                
                chunk = CodeChunk(
                    file_path=str(file_path),
                    content=chunk_content,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=chunk_type,
                    language=language,
                    tokens=int(tokens),
                    embedding=embedding,
                    metadata={'chunk_index': i, 'total_chunks': len(parsed_chunks)}
                )
                
                chunks.append(chunk)
                
                # Small delay to be respectful to the API
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Failed to embed chunk {i} of {file_path}: {e}")
                continue
        
        return chunks
    
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
            '.sh': 'bash',
            '.bash': 'bash',
            '.zsh': 'zsh',
            '.fish': 'fish',
            '.ps1': 'powershell',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.xml': 'xml',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'ini',
            '.md': 'markdown',
            '.rst': 'rst',
            '.txt': 'text',
            '.sql': 'sql',
            '.r': 'r',
            '.m': 'matlab',
            '.pl': 'perl',
            '.lua': 'lua',
            '.dart': 'dart',
            '.elm': 'elm',
            '.ex': 'elixir',
            '.exs': 'elixir',
            '.erl': 'erlang',
            '.hrl': 'erlang',
            '.clj': 'clojure',
            '.cljs': 'clojure',
            '.cljc': 'clojure',
            '.fs': 'fsharp',
            '.fsx': 'fsharp',
            '.ml': 'ocaml',
            '.mli': 'ocaml',
            '.hs': 'haskell',
            '.lhs': 'haskell',
            '.vim': 'vim',
            '.dockerfile': 'dockerfile'
        }
        
        ext = file_path.suffix.lower()
        return extension_map.get(ext, 'unknown')
    
    def _parse_code_structure(self, content: str, language: str) -> List[Tuple[str, int, int, str]]:
        """Parse code structure for intelligent chunking."""
        chunks = []
        
        # Simple regex-based parsing for common languages
        if language == 'python':
            chunks = self._parse_python_structure(content)
        elif language in ['javascript', 'typescript']:
            chunks = self._parse_js_structure(content)
        elif language == 'java':
            chunks = self._parse_java_structure(content)
        elif language == 'cpp':
            chunks = self._parse_cpp_structure(content)
        
        return chunks
    
    def _parse_python_structure(self, content: str) -> List[Tuple[str, int, int, str]]:
        """Parse Python code structure."""
        import re
        chunks = []
        lines = content.split('\n')
        
        # Find classes and functions
        current_chunk = []
        current_start = 1
        current_type = 'section'
        
        for i, line in enumerate(lines, 1):
            # Check for class definition
            if re.match(r'^class\s+\w+', line.strip()):
                if current_chunk:
                    chunks.append(('\n'.join(current_chunk), current_start, i-1, current_type))
                current_chunk = [line]
                current_start = i
                current_type = 'class'
            # Check for function definition
            elif re.match(r'^def\s+\w+', line.strip()) or re.match(r'^async\s+def\s+\w+', line.strip()):
                if current_chunk and current_type != 'class':
                    chunks.append(('\n'.join(current_chunk), current_start, i-1, current_type))
                    current_chunk = [line]
                    current_start = i
                    current_type = 'function'
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append(('\n'.join(current_chunk), current_start, len(lines), current_type))
        
        return chunks
    
    def _parse_js_structure(self, content: str) -> List[Tuple[str, int, int, str]]:
        """Parse JavaScript/TypeScript code structure."""
        import re
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_start = 1
        current_type = 'section'
        brace_count = 0
        
        for i, line in enumerate(lines, 1):
            # Check for function/class definitions
            if re.search(r'(function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\(|var\s+\w+\s*=\s*\()', line):
                if current_chunk and brace_count == 0:
                    chunks.append(('\n'.join(current_chunk), current_start, i-1, current_type))
                    current_chunk = [line]
                    current_start = i
                    current_type = 'function' if 'function' in line or '=>' in line else 'class'
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)
            
            # Track braces for proper chunking
            brace_count += line.count('{') - line.count('}')
        
        # Add final chunk
        if current_chunk:
            chunks.append(('\n'.join(current_chunk), current_start, len(lines), current_type))
        
        return chunks
    
    def _parse_java_structure(self, content: str) -> List[Tuple[str, int, int, str]]:
        """Parse Java code structure."""
        import re
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_start = 1
        current_type = 'section'
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for class definition
            if re.match(r'(public\s+|private\s+|protected\s+)?class\s+\w+', stripped):
                if current_chunk:
                    chunks.append(('\n'.join(current_chunk), current_start, i-1, current_type))
                current_chunk = [line]
                current_start = i
                current_type = 'class'
            # Check for method definition
            elif re.search(r'(public\s+|private\s+|protected\s+)?.*\s+\w+\s*\([^)]*\)\s*\{?', stripped) and not stripped.startswith('//'):
                if current_chunk and current_type != 'class':
                    chunks.append(('\n'.join(current_chunk), current_start, i-1, current_type))
                    current_chunk = [line]
                    current_start = i
                    current_type = 'method'
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append(('\n'.join(current_chunk), current_start, len(lines), current_type))
        
        return chunks
    
    def _parse_cpp_structure(self, content: str) -> List[Tuple[str, int, int, str]]:
        """Parse C++ code structure."""
        import re
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_start = 1
        current_type = 'section'
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for class/struct definition
            if re.match(r'(class|struct)\s+\w+', stripped):
                if current_chunk:
                    chunks.append(('\n'.join(current_chunk), current_start, i-1, current_type))
                current_chunk = [line]
                current_start = i
                current_type = 'class'
            # Check for function definition
            elif re.search(r'\w+\s*\([^)]*\)\s*\{?', stripped) and '(' in stripped and not stripped.startswith('//'):
                if current_chunk and current_type != 'class':
                    chunks.append(('\n'.join(current_chunk), current_start, i-1, current_type))
                    current_chunk = [line]
                    current_start = i
                    current_type = 'function'
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append(('\n'.join(current_chunk), current_start, len(lines), current_type))
        
        return chunks
    
    def _chunk_by_lines(self, content: str, max_tokens: int) -> List[Tuple[str, int, int, str]]:
        """Fall back to simple line-based chunking."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_start = 1
        current_tokens = 0
        
        for i, line in enumerate(lines, 1):
            line_tokens = len(line.split()) * 1.3  # Rough estimate
            
            if current_tokens + line_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunks.append(('\n'.join(current_chunk), current_start, i-1, 'section'))
                current_chunk = [line]
                current_start = i
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(('\n'.join(current_chunk), current_start, len(lines), 'section'))
        
        return chunks


class SemanticCodeSearch:
    """
    Semantic search engine for code using embeddings.
    """
    
    def __init__(self, embedding_service: GeminiEmbeddingService):
        self.embedding_service = embedding_service
        self.code_chunks: List[CodeChunk] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.file_embeddings: Dict[str, List[CodeChunk]] = defaultdict(list)
    
    async def index_files(self, file_paths: List[Path], progress_callback=None) -> Dict[str, Any]:
        """
        Index multiple files for semantic search.
        """
        indexed_files = 0
        total_chunks = 0
        failed_files = []
        
        for i, file_path in enumerate(file_paths):
            try:
                chunks = await self.embedding_service.embed_code_file(file_path)
                
                for chunk in chunks:
                    self.code_chunks.append(chunk)
                    self.file_embeddings[chunk.file_path].append(chunk)
                
                total_chunks += len(chunks)
                indexed_files += 1
                
                if progress_callback:
                    progress_callback(i + 1, len(file_paths), file_path)
                
            except Exception as e:
                failed_files.append((str(file_path), str(e)))
                print(f"Failed to index {file_path}: {e}")
        
        # Build embeddings matrix for efficient search
        self._build_embeddings_matrix()
        
        return {
            'indexed_files': indexed_files,
            'total_chunks': total_chunks,
            'failed_files': failed_files,
            'embedding_dimension': self.embedding_service.embedding_dimension
        }
    
    def _build_embeddings_matrix(self):
        """Build matrix of all embeddings for efficient similarity search."""
        if not self.code_chunks:
            return
        
        embeddings = []
        for chunk in self.code_chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
        
        if embeddings:
            self.embeddings_matrix = np.vstack(embeddings)
    
    async def semantic_search(self, query: str, top_k: int = 10, 
                            min_similarity: float = 0.3) -> List[SemanticMatch]:
        """
        Perform semantic search for the given query.
        """
        if not self.code_chunks or self.embeddings_matrix is None:
            return []
        
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(
            query, title="Code search query"
        )
        
        # Calculate similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings_matrix)
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        matches = []
        for idx in top_indices:
            similarity = similarities[idx]
            
            if similarity < min_similarity:
                break
            
            chunk = self.code_chunks[idx]
            
            # Generate match reasoning
            match_reason = self._generate_match_reason(query, chunk, similarity)
            
            # Get context window if needed
            context_window = self._get_context_window(chunk)
            
            match = SemanticMatch(
                file_path=chunk.file_path,
                similarity_score=float(similarity),
                chunk=chunk,
                match_reason=match_reason,
                context_window=context_window,
                full_chunk_content=chunk.content
            )
            
            matches.append(match)
        
        return matches
    
    def _cosine_similarity(self, query_embedding: np.ndarray, embeddings_matrix: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and all embeddings."""
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities
    
    def _generate_match_reason(self, query: str, chunk: CodeChunk, similarity: float) -> str:
        """Generate human-readable reason for the match."""
        confidence_level = "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
        
        reason = f"{confidence_level.title()} semantic similarity ({similarity:.2f}) "
        
        if chunk.chunk_type == 'function':
            reason += f"in function from {chunk.file_path}"
        elif chunk.chunk_type == 'class':
            reason += f"in class definition from {chunk.file_path}"
        else:
            reason += f"in {chunk.chunk_type} from {chunk.file_path}"
        
        return reason
    
    def _get_context_window(self, chunk: CodeChunk, window_lines: int = 5) -> Optional[str]:
        """Get context window around the chunk."""
        # This would require access to the full file content
        # For now, return the chunk content itself
        return chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
    
    def get_files_by_similarity(self, query: str, file_paths: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get files ranked by their relevance to the query.
        """
        file_scores = defaultdict(list)
        
        # Group chunks by file and calculate average similarity
        for chunk in self.code_chunks:
            if chunk.file_path in file_paths and chunk.embedding is not None:
                # This would require the query embedding - simplified for now
                file_scores[chunk.file_path].append(0.5)  # Placeholder
        
        # Calculate average scores per file
        file_rankings = []
        for file_path, scores in file_scores.items():
            avg_score = sum(scores) / len(scores)
            file_rankings.append((file_path, avg_score))
        
        # Sort by score and return top k
        file_rankings.sort(key=lambda x: x[1], reverse=True)
        return file_rankings[:top_k]
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index."""
        if not self.code_chunks:
            return {'total_chunks': 0, 'total_files': 0}
        
        file_count = len(self.file_embeddings)
        chunk_types = defaultdict(int)
        languages = defaultdict(int)
        
        for chunk in self.code_chunks:
            chunk_types[chunk.chunk_type] += 1
            languages[chunk.language] += 1
        
        return {
            'total_chunks': len(self.code_chunks),
            'total_files': file_count,
            'chunk_types': dict(chunk_types),
            'languages': dict(languages),
            'embedding_dimension': self.embedding_service.config.embedding_dimensions,
            'cache_stats': self.embedding_service.cache.get_cache_stats()
        }


class OpenAIEmbeddingService(BaseEmbeddingService):
    """
    Service for generating embeddings using OpenAI API.
    Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None, cache_dir: Optional[Path] = None):
        self.config = config or get_embedding_config().get_config(EmbeddingProvider.OPENAI)
        self.cache = EmbeddingCache(cache_dir or Path.home() / '.codeweaver' / 'embeddings')
        self.api_key = self.config.api_key
        
        # Add missing attributes expected by other parts of the system
        self.embedding_dimension = self.config.embedding_dimensions
        
        # Cost tracking
        self.cost_tracker = get_cost_tracker()
        
        # Rate limiting
        self.max_requests_per_minute = self.config.rate_limit_per_minute
        self.request_timestamps: List[float] = []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            'provider': 'openai',
            'model_name': self.config.model_name,
            'embedding_dimensions': self.config.embedding_dimensions,
            'max_tokens': self.config.max_tokens,
            'batch_size': self.config.batch_size
        }
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens supported by the model."""
        return self.config.max_tokens
    
    async def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings for multiple texts using OpenAI API."""
        if not self.api_key:
            logging.warning("OpenAI API key not configured")
            return None
        
        if not texts:
            return []
        
        try:
            # Import OpenAI (will need to be installed)
            try:
                import openai
            except ImportError:
                logging.error("OpenAI package not installed. Install with: pip install openai")
                return None
            
            # Wait for rate limit if needed
            await self._wait_for_rate_limit()
            
            # Configure OpenAI client
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Process in batches
            batch_size = min(self.config.batch_size, len(texts))
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Truncate texts that are too long - very aggressive approach
                truncated_batch = []
                for text in batch:
                    # Use only 50% of max tokens for safety (4096 tokens max)
                    max_safe_tokens = int(self.config.max_tokens * 0.5)
                    
                    # More accurate token estimation: 1 token ≈ 4 characters for code
                    estimated_tokens = len(text) / 4
                    
                    if estimated_tokens > max_safe_tokens:
                        # Calculate how many characters to keep
                        chars_to_keep = int(max_safe_tokens * 4)
                        truncated_text = text[:chars_to_keep]
                        truncated_batch.append(truncated_text)
                        logging.info(f"Truncated text from {len(text)} to {len(truncated_text)} characters for embedding")
                    else:
                        truncated_batch.append(text)
                
                # Get embeddings from OpenAI
                response = await client.embeddings.create(
                    model=self.config.model_name,
                    input=truncated_batch,
                    encoding_format="float"
                )
                
                # Extract embeddings
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Record request timestamp
                self.request_timestamps.append(time.time())
                
                # Track costs for this batch
                total_tokens = sum(len(text.split()) * 1.3 for text in truncated_batch)  # Rough estimate
                cost = self.cost_tracker.track_embedding_cost(
                    provider=AIProvider.OPENAI,
                    model_name=self.config.model_name,
                    token_count=int(total_tokens),
                    metadata={"batch_size": len(truncated_batch)}
                )
                logging.debug(f"OpenAI embedding cost: ${cost:.4f} for {len(truncated_batch)} texts")
            
            return all_embeddings
            
        except Exception as e:
            logging.error(f"Failed to get OpenAI embeddings: {e}")
            return None
    
    async def generate_embedding_for_file(self, file_path: Path, language: str = "") -> List[CodeChunk]:
        """Generate embedding for a code file using OpenAI."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Calculate content hash for caching
            content_hash = self._calculate_content_hash(content)
            
            # Check cache first
            relative_path = str(file_path)
            cached = self.cache.get_embedding(relative_path, content_hash, self.config.model_name)
            
            if cached:
                embedding, metadata = cached
                return [CodeChunk(
                    file_path=str(file_path),
                    content=content,
                    start_line=1,
                    end_line=len(content.split('\n')),
                    chunk_type='full_file',
                    language=language or self._detect_language(file_path),
                    tokens=metadata.tokens,
                    embedding=embedding
                )]
            
            # Generate new embedding
            embeddings = await self.get_embeddings([content])
            if not embeddings:
                return []
            
            embedding = np.array(embeddings[0])
            
            # Estimate tokens (rough approximation)
            tokens = len(content.split()) * 1.3
            
            # Store in cache
            metadata = EmbeddingMetadata(
                file_path=str(file_path),
                content_hash=content_hash,
                embedding_model=self.config.model_name,
                created_at=time.time(),
                tokens=int(tokens),
                file_type=file_path.suffix[1:] if file_path.suffix else 'unknown',
                language=language or self._detect_language(file_path),
                size_bytes=len(content.encode('utf-8'))
            )
            
            self.cache.store_embedding(
                str(file_path), content_hash, self.config.model_name, embedding, metadata
            )
            
            return [CodeChunk(
                file_path=str(file_path),
                content=content,
                start_line=1,
                end_line=len(content.split('\n')),
                chunk_type='full_file',
                language=language or self._detect_language(file_path),
                tokens=int(tokens),
                embedding=embedding
            )]
            
        except Exception as e:
            logging.error(f"Failed to generate OpenAI embedding for {file_path}: {e}")
            return []
    
    async def _wait_for_rate_limit(self):
        """Wait if we're hitting rate limits."""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 60
        ]
        
        # If we're at the rate limit, wait
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            oldest_request = min(self.request_timestamps)
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def embed_code_file(self, file_path: Path, max_chunk_size: int = 4000) -> List[CodeChunk]:
        """
        Embed a code file (compatibility method for SemanticCodeSearch).
        This delegates to generate_embedding_for_file for OpenAI.
        """
        return await self.generate_embedding_for_file(file_path)

    async def generate_embedding(self, text: str, title: str = "") -> np.ndarray:
        """Generate single embedding (compatibility method)."""
        embeddings = await self.get_embeddings([text])
        if not embeddings:
            raise RuntimeError("Failed to generate embedding")
        return np.array(embeddings[0])

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
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala'
        }
        
        ext = file_path.suffix.lower()
        return extension_map.get(ext, 'unknown')


def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    # Convert to numpy arrays
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def create_embedding_service(provider: EmbeddingProvider = None, 
                           config: Optional[EmbeddingConfig] = None,
                           cache_dir: Optional[Path] = None) -> BaseEmbeddingService:
    """Factory function to create embedding service instances."""
    config_manager = get_embedding_config()
    
    if provider is None:
        # Auto-select provider based on available API keys
        available = config_manager.get_available_providers()
        if available.get(EmbeddingProvider.GEMINI):
            provider = EmbeddingProvider.GEMINI
        elif available.get(EmbeddingProvider.OPENAI):
            provider = EmbeddingProvider.OPENAI
        else:
            # Default to Gemini
            provider = EmbeddingProvider.GEMINI
    
    if config is None:
        config = config_manager.get_config(provider)
    
    if provider == EmbeddingProvider.GEMINI:
        return GeminiEmbeddingService(config, cache_dir)
    elif provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddingService(config, cache_dir)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")