import pytest
import asyncio
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from codeweaver.ai.embeddings import (
    OpenAIEmbeddingService, GeminiEmbeddingService, SemanticCodeSearch,
    EmbeddingCache, CodeChunk, SemanticMatch, EmbeddingMetadata,
    calculate_cosine_similarity, create_embedding_service
)
from codeweaver.config.embedding_config import EmbeddingProvider, EmbeddingConfig


class TestEmbeddingCache:
    """Test the embedding cache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = EmbeddingCache(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_initialization(self):
        """Test cache initialization and database creation."""
        assert self.cache.cache_dir.exists()
        assert self.cache.db_path.exists()
        
        # Test that database tables were created
        import sqlite3
        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
            )
            assert cursor.fetchone() is not None
    
    def test_store_and_retrieve_embedding(self):
        """Test storing and retrieving embeddings from cache."""
        # Create test data
        file_path = "test/file.py"
        content_hash = "abc123"
        model = "text-embedding-3-small"
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        metadata = EmbeddingMetadata(
            file_path=file_path,
            content_hash=content_hash,
            embedding_model=model,
            created_at=1234567890.0,
            tokens=100,
            file_type="py",
            language="python",
            size_bytes=500
        )
        
        # Store embedding
        self.cache.store_embedding(file_path, content_hash, model, embedding, metadata)
        
        # Retrieve embedding
        result = self.cache.get_embedding(file_path, content_hash, model)
        
        assert result is not None
        retrieved_embedding, retrieved_metadata = result
        
        # Check embedding
        np.testing.assert_array_equal(retrieved_embedding, embedding)
        
        # Check metadata
        assert retrieved_metadata.file_path == file_path
        assert retrieved_metadata.content_hash == content_hash
        assert retrieved_metadata.embedding_model == model
        assert retrieved_metadata.tokens == 100
    
    def test_cache_miss(self):
        """Test cache miss when embedding doesn't exist."""
        result = self.cache.get_embedding("nonexistent.py", "hash123", "model")
        assert result is None
    
    def test_cleanup_old_embeddings(self):
        """Test cleanup of old cache entries."""
        import time
        
        # Store an old embedding
        old_metadata = EmbeddingMetadata(
            file_path="old_file.py",
            content_hash="old_hash",
            embedding_model="old_model",
            created_at=time.time() - (40 * 24 * 60 * 60),  # 40 days ago
            tokens=50,
            file_type="py",
            language="python",
            size_bytes=200
        )
        
        embedding = np.array([0.1, 0.2])
        self.cache.store_embedding(
            "old_file.py", "old_hash", "old_model", embedding, old_metadata
        )
        
        # Store a recent embedding
        recent_metadata = EmbeddingMetadata(
            file_path="recent_file.py",
            content_hash="recent_hash",
            embedding_model="recent_model",
            created_at=time.time(),
            tokens=75,
            file_type="py",
            language="python",
            size_bytes=300
        )
        
        self.cache.store_embedding(
            "recent_file.py", "recent_hash", "recent_model", embedding, recent_metadata
        )
        
        # Cleanup old embeddings (older than 30 days)
        self.cache.cleanup_old_embeddings(30)
        
        # Old embedding should be gone
        old_result = self.cache.get_embedding("old_file.py", "old_hash", "old_model")
        assert old_result is None
        
        # Recent embedding should still exist
        recent_result = self.cache.get_embedding("recent_file.py", "recent_hash", "recent_model")
        assert recent_result is not None
    
    def test_get_cache_stats(self):
        """Test cache statistics functionality."""
        # Initially empty
        stats = self.cache.get_cache_stats()
        assert stats['total_embeddings'] == 0
        assert stats['total_size_bytes'] == 0
        
        # Add some embeddings
        for i in range(3):
            metadata = EmbeddingMetadata(
                file_path=f"file_{i}.py",
                content_hash=f"hash_{i}",
                embedding_model="test_model",
                created_at=1234567890.0,
                tokens=100 + i,
                file_type="py",
                language="python",
                size_bytes=200 + i * 50
            )
            
            embedding = np.array([float(i), float(i + 1)])
            self.cache.store_embedding(
                f"file_{i}.py", f"hash_{i}", "test_model", embedding, metadata
            )
        
        stats = self.cache.get_cache_stats()
        assert stats['total_embeddings'] == 3
        assert stats['total_size_bytes'] == 200 + 250 + 300  # Sum of size_bytes
        assert stats['embeddings_by_model']['test_model'] == 3


@pytest.mark.skipif(not pytest.config.getoption("--run-integration", default=False), 
                   reason="Integration tests require --run-integration flag")
class TestOpenAIEmbeddingServiceIntegration:
    """Integration tests for OpenAI embedding service with real API calls."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_cache_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_cache_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_get_embeddings_real_api(self, embedding_config):
        """Test getting embeddings from real OpenAI API."""
        service = OpenAIEmbeddingService(embedding_config, self.temp_cache_dir)
        
        texts = [
            "This is a test function that calculates the sum of two numbers.",
            "def add(a, b): return a + b",
            "Python is a programming language."
        ]
        
        embeddings = await service.get_embeddings(texts)
        
        assert embeddings is not None
        assert len(embeddings) == 3
        
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == embedding_config.embedding_dimensions
            assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_generate_embedding_for_file_real_api(self, embedding_config, temp_project_dir):
        """Test generating embeddings for a real file."""
        service = OpenAIEmbeddingService(embedding_config, self.temp_cache_dir)
        
        # Use one of the test files
        test_file = temp_project_dir / "utils.py"
        
        chunks = await service.generate_embedding_for_file(test_file, "python")
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert isinstance(chunk, CodeChunk)
        assert chunk.file_path == str(test_file)
        assert chunk.language == "python"
        assert chunk.tokens > 0
        assert chunk.embedding is not None
        assert isinstance(chunk.embedding, np.ndarray)
        assert chunk.embedding.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_caching_functionality_real_api(self, embedding_config, temp_project_dir):
        """Test that caching works with real API calls."""
        service = OpenAIEmbeddingService(embedding_config, self.temp_cache_dir)
        
        test_file = temp_project_dir / "main.py"
        
        # First call - should hit the API
        start_time = asyncio.get_event_loop().time()
        chunks1 = await service.generate_embedding_for_file(test_file)
        first_call_time = asyncio.get_event_loop().time() - start_time
        
        # Second call - should use cache
        start_time = asyncio.get_event_loop().time()
        chunks2 = await service.generate_embedding_for_file(test_file)
        second_call_time = asyncio.get_event_loop().time() - start_time
        
        # Cache should make second call much faster
        assert second_call_time < first_call_time * 0.5
        
        # Results should be identical
        assert len(chunks1) == len(chunks2)
        np.testing.assert_array_equal(chunks1[0].embedding, chunks2[0].embedding)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, embedding_config):
        """Test rate limiting functionality."""
        # Use a very low rate limit for testing
        limited_config = EmbeddingConfig(
            api_key=embedding_config.api_key,
            model_name=embedding_config.model_name,
            embedding_dimensions=embedding_config.embedding_dimensions,
            max_tokens=embedding_config.max_tokens,
            batch_size=1,
            rate_limit_per_minute=2  # Very low limit
        )
        
        service = OpenAIEmbeddingService(limited_config, self.temp_cache_dir)
        
        # Make multiple requests
        texts = [f"Test text {i}" for i in range(3)]
        
        start_time = asyncio.get_event_loop().time()
        embeddings = await service.get_embeddings(texts)
        total_time = asyncio.get_event_loop().time() - start_time
        
        # Should have taken some time due to rate limiting
        assert total_time > 1.0  # At least 1 second delay
        assert embeddings is not None
        assert len(embeddings) == 3


class TestSemanticCodeSearch:
    """Test semantic code search functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_embedding_service = MagicMock()
        self.mock_embedding_service.config.embedding_dimensions = 1536
        self.search = SemanticCodeSearch(self.mock_embedding_service)
    
    @pytest.mark.asyncio
    async def test_index_files_success(self, temp_project_dir):
        """Test successful file indexing."""
        files = [
            temp_project_dir / "main.py",
            temp_project_dir / "utils.py",
            temp_project_dir / "models.py"
        ]
        
        # Mock the embedding service to return fake chunks
        async def mock_embed_file(file_path):
            embedding = np.random.rand(1536).astype(np.float32)
            return [CodeChunk(
                file_path=str(file_path),
                content=f"mock content for {file_path.name}",
                start_line=1,
                end_line=10,
                chunk_type="full_file",
                language="python",
                tokens=100,
                embedding=embedding
            )]
        
        self.mock_embedding_service.embed_code_file = mock_embed_file
        
        progress_updates = []
        def progress_callback(current, total, file_path):
            progress_updates.append((current, total, file_path))
        
        stats = await self.search.index_files(files, progress_callback)
        
        assert stats['indexed_files'] == 3
        assert stats['total_chunks'] == 3
        assert len(stats['failed_files']) == 0
        assert len(progress_updates) == 3
        
        # Check that chunks were added
        assert len(self.search.code_chunks) == 3
        assert self.search.embeddings_matrix is not None
        assert self.search.embeddings_matrix.shape == (3, 1536)
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, temp_project_dir):
        """Test semantic search functionality."""
        # Set up some mock chunks with embeddings
        embeddings = [
            np.array([1.0, 0.0, 0.0] + [0.0] * 1533, dtype=np.float32),
            np.array([0.0, 1.0, 0.0] + [0.0] * 1533, dtype=np.float32),
            np.array([0.0, 0.0, 1.0] + [0.0] * 1533, dtype=np.float32)
        ]
        
        chunks = []
        for i, embedding in enumerate(embeddings):
            chunk = CodeChunk(
                file_path=f"file_{i}.py",
                content=f"def function_{i}(): pass",
                start_line=1,
                end_line=2,
                chunk_type="function",
                language="python",
                tokens=20,
                embedding=embedding
            )
            chunks.append(chunk)
        
        self.search.code_chunks = chunks
        self.search._build_embeddings_matrix()
        
        # Mock the embedding service to return a query embedding
        query_embedding = np.array([0.9, 0.1, 0.0] + [0.0] * 1533, dtype=np.float32)
        self.mock_embedding_service.generate_embedding = AsyncMock(return_value=query_embedding)
        
        # Perform search
        matches = await self.search.semantic_search("function 0", top_k=2, min_similarity=0.1)
        
        assert len(matches) <= 2
        assert all(isinstance(match, SemanticMatch) for match in matches)
        
        # First match should have highest similarity (to query similar to embedding 0)
        if matches:
            assert matches[0].file_path == "file_0.py"
            assert matches[0].similarity_score > 0.5
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        # Test identical vectors
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        similarity = self.search._cosine_similarity(vec1, np.array([vec2]))[0]
        assert abs(similarity - 1.0) < 1e-6
        
        # Test orthogonal vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = self.search._cosine_similarity(vec1, np.array([vec2]))[0]
        assert abs(similarity) < 1e-6
        
        # Test opposite vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        similarity = self.search._cosine_similarity(vec1, np.array([vec2]))[0]
        assert abs(similarity - (-1.0)) < 1e-6
    
    def test_get_search_stats(self):
        """Test search statistics functionality."""
        # Empty index
        stats = self.search.get_search_stats()
        assert stats['total_chunks'] == 0
        assert stats['total_files'] == 0
        
        # Add some mock chunks
        chunks = [
            CodeChunk("file1.py", "content1", 1, 10, "function", "python", 50),
            CodeChunk("file1.py", "content2", 11, 20, "class", "python", 75),
            CodeChunk("file2.js", "content3", 1, 15, "function", "javascript", 60)
        ]
        
        for chunk in chunks:
            self.search.code_chunks.append(chunk)
            self.search.file_embeddings[chunk.file_path].append(chunk)
        
        stats = self.search.get_search_stats()
        assert stats['total_chunks'] == 3
        assert stats['total_files'] == 2
        assert stats['chunk_types']['function'] == 2
        assert stats['chunk_types']['class'] == 1
        assert stats['languages']['python'] == 2
        assert stats['languages']['javascript'] == 1


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_cosine_similarity(self):
        """Test standalone cosine similarity function."""
        # Test with lists
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        similarity = calculate_cosine_similarity(embedding1, embedding2)
        assert abs(similarity) < 1e-6
        
        # Test with numpy arrays
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([1.0, 2.0, 3.0])
        similarity = calculate_cosine_similarity(embedding1, embedding2)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test with zero vectors
        embedding1 = [0.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]
        similarity = calculate_cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0
    
    @patch('codeweaver.ai.embeddings.get_embedding_config')
    def test_create_embedding_service_factory(self, mock_config_manager):
        """Test embedding service factory function."""
        # Mock configuration manager
        mock_config = EmbeddingConfig(
            api_key="test_key",
            model_name="text-embedding-3-small",
            embedding_dimensions=1536,
            max_tokens=8192,
            batch_size=100,
            rate_limit_per_minute=1000
        )
        
        mock_manager = MagicMock()
        mock_manager.get_config.return_value = mock_config
        mock_manager.get_available_providers.return_value = {
            EmbeddingProvider.OPENAI: True,
            EmbeddingProvider.GEMINI: False
        }
        mock_config_manager.return_value = mock_manager
        
        # Test OpenAI service creation
        service = create_embedding_service(EmbeddingProvider.OPENAI)
        assert isinstance(service, OpenAIEmbeddingService)
        assert service.config.model_name == "text-embedding-3-small"
        
        # Test auto-selection (should pick OpenAI since it's available)
        service = create_embedding_service()
        assert isinstance(service, OpenAIEmbeddingService)
    
    def test_data_classes(self):
        """Test data class instantiation and attributes."""
        # Test CodeChunk
        chunk = CodeChunk(
            file_path="test.py",
            content="def test(): pass",
            start_line=1,
            end_line=2,
            chunk_type="function",
            language="python",
            tokens=10,
            embedding=np.array([0.1, 0.2, 0.3]),
            metadata={"test": True}
        )
        
        assert chunk.file_path == "test.py"
        assert chunk.chunk_type == "function"
        assert chunk.metadata["test"] is True
        
        # Test SemanticMatch
        match = SemanticMatch(
            file_path="test.py",
            similarity_score=0.85,
            chunk=chunk,
            match_reason="High similarity to query",
            context_window="def test(): pass",
            full_chunk_content="def test(): pass"
        )
        
        assert match.similarity_score == 0.85
        assert match.match_reason == "High similarity to query"
        assert match.chunk == chunk


# Add integration test configuration
def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require API keys"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        # Run all tests including integration tests
        return
    
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
