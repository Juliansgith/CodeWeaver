"""
Cost tracking service for AI operations including embeddings and LLM usage.
Provides real-time cost tracking and historical usage analytics.
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging

# Import LLM config for dynamic pricing
try:
    from ..config.llm_config import get_llm_config
    HAS_LLM_CONFIG = True
except ImportError:
    HAS_LLM_CONFIG = False

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"

class OperationType(Enum):
    EMBEDDING = "embedding"
    COMPLETION = "completion"
    IMAGE_GENERATION = "image_generation"
    TTS = "tts"
    STT = "stt"

@dataclass
class CostEntry:
    """Single cost entry for tracking AI usage."""
    timestamp: float
    provider: AIProvider
    operation_type: OperationType
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    session_id: Optional[str] = None
    project_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CostSummary:
    """Summary of costs for a time period."""
    total_cost: float
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    by_provider: Dict[str, float]
    by_operation: Dict[str, float]
    by_model: Dict[str, float]
    start_time: float
    end_time: float

class CostTracker:
    """
    Tracks costs for AI operations with persistent storage and real-time updates.
    """
    
    # Fallback pricing per 1K tokens (USD) - Used when LLM config is not available
    FALLBACK_PRICING = {
        AIProvider.OPENAI: {
            # Embeddings
            "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
            "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
            "text-embedding-ada-002": {"input": 0.00010, "output": 0.0},
            
            # Completions
            "gpt-4.1-nano": {"input": 0.00015, "output": 0.0006},  # Updated with gpt-4.1-nano
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        },
        
        AIProvider.GEMINI: {
            # Embeddings
            "text-embedding-004": {"input": 0.00001, "output": 0.0},
            "embedding-001": {"input": 0.00001, "output": 0.0},
            
            # Completions
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
        },
        
        AIProvider.ANTHROPIC: {
            # Completions
            "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        }
    }
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize cost tracker with SQLite database.
        
        Args:
            db_path: Path to SQLite database file. Defaults to ~/.codeweaver/costs.db
        """
        if db_path is None:
            db_path = Path.home() / '.codeweaver' / 'costs.db'
        
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Current session tracking
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self.session_costs: List[CostEntry] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        logger.info(f"CostTracker initialized with database at {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with cost tracking tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    provider TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    session_id TEXT,
                    project_path TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON cost_entries(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_provider ON cost_entries(provider)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session ON cost_entries(session_id)
            """)
    
    def start_session(self, session_id: str, project_path: Optional[str] = None):
        """Start a new cost tracking session."""
        with self._lock:
            self.current_session_id = session_id
            self.session_start_time = time.time()
            self.session_costs = []
            logger.info(f"Started cost tracking session: {session_id}")
    
    def end_session(self) -> Optional[CostSummary]:
        """End current session and return summary."""
        with self._lock:
            if not self.current_session_id:
                return None
            
            summary = self.get_session_summary(self.current_session_id)
            self.current_session_id = None
            self.session_start_time = None
            self.session_costs = []
            
            logger.info(f"Ended cost tracking session. Total cost: ${summary.total_cost:.4f}")
            return summary
    
    def track_embedding_cost(self, provider: AIProvider, model_name: str, 
                           token_count: int, project_path: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Track cost for embedding operation.
        
        Args:
            provider: AI provider (OpenAI, Gemini, etc.)
            model_name: Name of the embedding model
            token_count: Number of tokens processed
            project_path: Optional project path
            metadata: Optional metadata dictionary
            
        Returns:
            Cost in USD for this operation
        """
        cost = self._calculate_cost(provider, model_name, OperationType.EMBEDDING, 
                                  token_count, 0)
        
        entry = CostEntry(
            timestamp=time.time(),
            provider=provider,
            operation_type=OperationType.EMBEDDING,
            model_name=model_name,
            input_tokens=token_count,
            output_tokens=0,
            cost_usd=cost,
            session_id=self.current_session_id,
            project_path=project_path,
            metadata=metadata
        )
        
        self._store_entry(entry)
        
        with self._lock:
            self.session_costs.append(entry)
        
        logger.debug(f"Tracked embedding cost: ${cost:.4f} for {token_count} tokens")
        return cost
    
    def track_completion_cost(self, provider: AIProvider, model_name: str,
                            input_tokens: int, output_tokens: int,
                            project_path: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        Track cost for completion/generation operation.
        
        Args:
            provider: AI provider
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            project_path: Optional project path
            metadata: Optional metadata dictionary
            
        Returns:
            Cost in USD for this operation
        """
        cost = self._calculate_cost(provider, model_name, OperationType.COMPLETION,
                                  input_tokens, output_tokens)
        
        entry = CostEntry(
            timestamp=time.time(),
            provider=provider,
            operation_type=OperationType.COMPLETION,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            session_id=self.current_session_id,
            project_path=project_path,
            metadata=metadata
        )
        
        self._store_entry(entry)
        
        with self._lock:
            self.session_costs.append(entry)
        
        logger.debug(f"Tracked completion cost: ${cost:.4f} for {input_tokens}+{output_tokens} tokens")
        return cost
    
    def _calculate_cost(self, provider: AIProvider, model_name: str,
                       operation_type: OperationType, input_tokens: int, 
                       output_tokens: int) -> float:
        """Calculate cost for an operation using dynamic pricing from LLM config."""
        
        # Try to use dynamic pricing from LLM config first
        if HAS_LLM_CONFIG:
            try:
                llm_config = get_llm_config()
                cost = llm_config.get_cost_estimate(model_name, input_tokens, output_tokens)
                if cost > 0:
                    return cost
            except Exception as e:
                logger.warning(f"Failed to get cost from LLM config: {e}")
        
        # Fall back to static pricing
        provider_pricing = self.FALLBACK_PRICING.get(provider, {})
        model_pricing = provider_pricing.get(model_name, {"input": 0.001, "output": 0.001})
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def _store_entry(self, entry: CostEntry):
        """Store cost entry in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO cost_entries 
                (timestamp, provider, operation_type, model_name, input_tokens, 
                 output_tokens, cost_usd, session_id, project_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp, entry.provider.value, entry.operation_type.value,
                entry.model_name, entry.input_tokens, entry.output_tokens,
                entry.cost_usd, entry.session_id, entry.project_path,
                json.dumps(entry.metadata) if entry.metadata else None
            ))
    
    def get_current_session_cost(self) -> float:
        """Get total cost for current session."""
        with self._lock:
            return sum(entry.cost_usd for entry in self.session_costs)
    
    def get_current_session_stats(self) -> Dict[str, Any]:
        """Get detailed stats for current session."""
        with self._lock:
            if not self.current_session_id:
                return {"active": False}
            
            total_cost = sum(entry.cost_usd for entry in self.session_costs)
            total_tokens = sum(entry.input_tokens + entry.output_tokens 
                             for entry in self.session_costs)
            
            by_operation = {}
            by_provider = {}
            by_model = {}
            
            for entry in self.session_costs:
                # By operation
                op_key = entry.operation_type.value
                by_operation[op_key] = by_operation.get(op_key, 0) + entry.cost_usd
                
                # By provider
                prov_key = entry.provider.value
                by_provider[prov_key] = by_provider.get(prov_key, 0) + entry.cost_usd
                
                # By model
                by_model[entry.model_name] = by_model.get(entry.model_name, 0) + entry.cost_usd
            
            return {
                "active": True,
                "session_id": self.current_session_id,
                "duration": time.time() - (self.session_start_time or time.time()),
                "total_cost": total_cost,
                "total_operations": len(self.session_costs),
                "total_tokens": total_tokens,
                "by_operation": by_operation,
                "by_provider": by_provider,
                "by_model": by_model,
                "recent_operations": [
                    {
                        "timestamp": entry.timestamp,
                        "provider": entry.provider.value,
                        "operation": entry.operation_type.value,
                        "model": entry.model_name,
                        "tokens": entry.input_tokens + entry.output_tokens,
                        "cost": entry.cost_usd
                    }
                    for entry in self.session_costs[-10:]  # Last 10 operations
                ]
            }
    
    def get_session_summary(self, session_id: str) -> CostSummary:
        """Get cost summary for a specific session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, provider, operation_type, model_name, 
                       input_tokens, output_tokens, cost_usd
                FROM cost_entries 
                WHERE session_id = ?
                ORDER BY timestamp
            """, (session_id,))
            
            entries = cursor.fetchall()
        
        if not entries:
            return CostSummary(0, 0, 0, 0, {}, {}, {}, 0, 0)
        
        total_cost = sum(entry[6] for entry in entries)
        total_input_tokens = sum(entry[4] for entry in entries)
        total_output_tokens = sum(entry[5] for entry in entries)
        
        by_provider = {}
        by_operation = {}
        by_model = {}
        
        for entry in entries:
            provider = entry[1]
            operation = entry[2]
            model = entry[3]
            cost = entry[6]
            
            by_provider[provider] = by_provider.get(provider, 0) + cost
            by_operation[operation] = by_operation.get(operation, 0) + cost
            by_model[model] = by_model.get(model, 0) + cost
        
        return CostSummary(
            total_cost=total_cost,
            total_requests=len(entries),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            by_provider=by_provider,
            by_operation=by_operation,
            by_model=by_model,
            start_time=entries[0][0],
            end_time=entries[-1][0]
        )
    
    def get_period_summary(self, hours: int = 24) -> CostSummary:
        """Get cost summary for the last N hours."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, provider, operation_type, model_name,
                       input_tokens, output_tokens, cost_usd
                FROM cost_entries 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """, (start_time, end_time))
            
            entries = cursor.fetchall()
        
        if not entries:
            return CostSummary(0, 0, 0, 0, {}, {}, {}, start_time, end_time)
        
        total_cost = sum(entry[6] for entry in entries)
        total_input_tokens = sum(entry[4] for entry in entries)
        total_output_tokens = sum(entry[5] for entry in entries)
        
        by_provider = {}
        by_operation = {}
        by_model = {}
        
        for entry in entries:
            provider = entry[1]
            operation = entry[2] 
            model = entry[3]
            cost = entry[6]
            
            by_provider[provider] = by_provider.get(provider, 0) + cost
            by_operation[operation] = by_operation.get(operation, 0) + cost
            by_model[model] = by_model.get(model, 0) + cost
        
        return CostSummary(
            total_cost=total_cost,
            total_requests=len(entries),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            by_provider=by_provider,
            by_operation=by_operation,
            by_model=by_model,
            start_time=start_time,
            end_time=end_time
        )
    
    def cleanup_old_entries(self, days: int = 90):
        """Remove cost entries older than specified days."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cost_entries WHERE timestamp < ?", 
                                (cutoff_time,))
            deleted_count = cursor.rowcount
        
        logger.info(f"Cleaned up {deleted_count} old cost entries")
        return deleted_count
    
    def export_data(self, output_path: Path, start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> int:
        """Export cost data to JSON file."""
        conditions = []
        params = []
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT timestamp, provider, operation_type, model_name,
                       input_tokens, output_tokens, cost_usd, session_id,
                       project_path, metadata
                FROM cost_entries {where_clause}
                ORDER BY timestamp
            """, params)
            
            entries = []
            for row in cursor.fetchall():
                entry = {
                    "timestamp": row[0],
                    "datetime": datetime.fromtimestamp(row[0]).isoformat(),
                    "provider": row[1],
                    "operation_type": row[2],
                    "model_name": row[3],
                    "input_tokens": row[4],
                    "output_tokens": row[5],
                    "cost_usd": row[6],
                    "session_id": row[7],
                    "project_path": row[8],
                    "metadata": json.loads(row[9]) if row[9] else None
                }
                entries.append(entry)
        
        with open(output_path, 'w') as f:
            json.dump({
                "export_timestamp": time.time(),
                "export_datetime": datetime.now().isoformat(),
                "total_entries": len(entries),
                "entries": entries
            }, f, indent=2)
        
        logger.info(f"Exported {len(entries)} cost entries to {output_path}")
        return len(entries)


# Global cost tracker instance
_global_cost_tracker: Optional[CostTracker] = None

def get_cost_tracker() -> CostTracker:
    """Get or create global cost tracker instance."""
    global _global_cost_tracker
    if _global_cost_tracker is None:
        _global_cost_tracker = CostTracker()
    return _global_cost_tracker