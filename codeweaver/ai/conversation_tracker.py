"""
Conversation-aware context selection and learning from chat history.
"""

import time
import json
import logging
import sqlite3
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from collections import deque, defaultdict, Counter
from datetime import datetime, timedelta
import re

from .embeddings import calculate_cosine_similarity, BaseEmbeddingService
from .factory import ai_factory

@dataclass
class ConversationMessage:
    """Represents a message in the conversation."""
    timestamp: float
    role: str              # 'user' or 'assistant'
    content: str
    message_type: str      # 'query', 'response', 'context_request', etc.
    context_files: List[str] = None  # Files that were in context
    topics: List[str] = None         # Extracted topics/keywords
    embedding: Optional[List[float]] = None

@dataclass
class ConversationSession:
    """Represents a conversation session."""
    session_id: str
    start_time: float
    end_time: Optional[float]
    messages: List[ConversationMessage]
    project_path: str
    primary_topics: List[str] = None
    file_access_patterns: Dict[str, int] = None  # file_path -> access_count

@dataclass 
class ContextAdaptation:
    """Result of conversation-aware context adaptation."""
    recommended_files: List[str]
    confidence_scores: Dict[str, float]
    reasoning: List[str]
    adaptation_strategy: str
    conversation_relevance: float

class ConversationAnalyzer:
    """Analyzes conversation patterns to extract insights."""
    
    def __init__(self, embedding_service: Optional[BaseEmbeddingService] = None):
        self._embedding_service = embedding_service
        self.topic_patterns = self._build_topic_patterns()
        self.intent_patterns = self._build_intent_patterns()
    
    @property
    def embedding_service(self) -> Optional[BaseEmbeddingService]:
        """Lazy load the embedding service from the factory."""
        if self._embedding_service is None:
            self._embedding_service = ai_factory.get_embedding_service()
        return self._embedding_service

    def _build_topic_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for topic extraction."""
        return {
            'debugging': [
                r'\b(debug|fix|bug|error|issue|problem|broken|crash|exception|fail)\b',
                r'\b(not working|wrong|incorrect|unexpected)\b',
                r'\b(trace|troubleshoot|diagnose|investigate)\b'
            ],
            'implementation': [
                r'\b(implement|add|create|build|develop|write|code)\b',
                r'\b(feature|functionality|method|function|class)\b',
                r'\b(how to|need to|want to|trying to)\b'
            ],
            'understanding': [
                r'\b(understand|explain|how does|what is|show me)\b',
                r'\b(learn|explore|analyze|review|examine)\b',
                r'\b(workflow|process|flow|architecture)\b'
            ],
            'optimization': [
                r'\b(optimize|improve|performance|speed|faster|slow)\b',
                r'\b(refactor|restructure|cleanup|simplify)\b',
                r'\b(memory|cpu|efficiency|bottleneck)\b'
            ],
            'testing': [
                r'\b(test|testing|unit test|integration test|spec)\b',
                r'\b(coverage|mock|stub|fixture|assertion)\b',
                r'\b(validate|verify|check|ensure)\b'
            ],
            'security': [
                r'\b(security|secure|vulnerability|auth|authorization)\b',
                r'\b(permission|access|role|encrypt|decrypt)\b',
                r'\b(sanitize|validate|xss|injection|csrf)\b'
            ]
        }
    
    def _build_intent_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for intent recognition."""
        return {
            'question': [
                r'^(what|how|why|when|where|which|who)',
                r'\?$',
                r'\b(can you|could you|would you|help me)\b'
            ],
            'request': [
                r'\b(please|can you|could you|would you|help me)\b',
                r'\b(show me|give me|provide|generate)\b',
                r'^(find|search|look for|get me)\b'
            ],
            'explanation': [
                r'\b(explain|describe|tell me about|what does)\b',
                r'\b(how does it work|how is it used|what is it for)\b'
            ],
            'modification': [
                r'\b(change|modify|update|alter|edit)\b',
                r'\b(add|remove|delete|insert|replace)\b'
            ]
        }
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        topics = []
        text_lower = text.lower()
        
        for topic, patterns in self.topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    topics.append(topic)
                    break
        
        return list(set(topics))
    
    def detect_intent(self, text: str) -> str:
        """Detect the intent of a message."""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        return 'statement'
    
    def extract_file_mentions(self, text: str) -> List[str]:
        """Extract file mentions from text."""
        file_patterns = [
            r'\b(\w+\.\w+)\b',  # filename.ext
            r'`([^`]+\.\w+)`',  # `filename.ext`
            r'"([^"]+\.\w+)"',  # "filename.ext"
            r"'([^']+\.\w+)'",  # 'filename.ext'
            r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+)\b'
        ]
        
        mentioned_files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, text)
            mentioned_files.extend(matches)
        
        false_positives = {
            'www.', 'http.', 'https.', 'ftp.', 'localhost.',
            'example.com', 'test.com', 'domain.com'
        }
        
        return [f for f in mentioned_files if not any(fp in f.lower() for fp in false_positives)]
    
    def analyze_conversation_flow(self, messages: List[ConversationMessage]) -> Dict[str, Any]:
        """Analyze the flow and patterns in a conversation."""
        if not messages:
            return {}
        
        analysis = {
            'duration_minutes': (messages[-1].timestamp - messages[0].timestamp) / 60,
            'message_count': len(messages),
            'topic_evolution': [],
            'question_answer_pairs': 0,
            'context_switches': 0,
            'dominant_topics': [],
            'file_focus_patterns': []
        }
        
        window_size = 3
        for i in range(0, len(messages), window_size):
            window_messages = messages[i:i + window_size]
            window_topics = []
            for msg in window_messages:
                if msg.topics:
                    window_topics.extend(msg.topics)
            
            if window_topics:
                top_topics = Counter(window_topics).most_common(3)
                analysis['topic_evolution'].append({
                    'timestamp': window_messages[0].timestamp,
                    'topics': [topic for topic, _ in top_topics]
                })
        
        for i in range(len(messages) - 1):
            current, next_msg = messages[i], messages[i + 1]
            if (current.role == 'user' and 
                self.detect_intent(current.content) == 'question' and
                next_msg.role == 'assistant'):
                analysis['question_answer_pairs'] += 1
        
        previous_topics = set()
        for msg in messages:
            if msg.topics:
                current_topics = set(msg.topics)
                if previous_topics and not current_topics.intersection(previous_topics):
                    analysis['context_switches'] += 1
                previous_topics = current_topics
        
        all_topics = []
        for msg in messages:
            if msg.topics:
                all_topics.extend(msg.topics)
        
        if all_topics:
            analysis['dominant_topics'] = [topic for topic, _ in Counter(all_topics).most_common(5)]
        
        return analysis

class ConversationTracker:
    """
    Tracks conversation history and provides conversation-aware context selection.
    """
    
    def __init__(self, db_path: Path, embedding_service: Optional[BaseEmbeddingService] = None):
        self.db_path = db_path
        self._embedding_service = embedding_service
        self.analyzer = ConversationAnalyzer(self._embedding_service)
        
        self.current_session: Optional[ConversationSession] = None
        self.message_buffer = deque(maxlen=50)
        self.context_cache = {}
        
        self.learning_window_hours = 24
        self.min_confidence_threshold = 0.3
        self.context_relevance_decay = 0.1
        
        self._init_database()
    
    @property
    def embedding_service(self) -> Optional[BaseEmbeddingService]:
        """Lazy load the embedding service from the factory."""
        if self._embedding_service is None:
            self._embedding_service = ai_factory.get_embedding_service()
        return self._embedding_service
    
    def _init_database(self):
        """Initialize SQLite database for conversation history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    project_path TEXT NOT NULL,
                    primary_topics TEXT,
                    message_count INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    context_files TEXT,
                    topics TEXT,
                    embedding_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS file_access_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    last_access_time REAL NOT NULL,
                    context_relevance REAL DEFAULT 1.0,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS context_adaptations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    user_query TEXT NOT NULL,
                    recommended_files TEXT NOT NULL,
                    adaptation_strategy TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    user_feedback TEXT,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                )
            ''')
    
    def start_session(self, project_path: str, session_id: Optional[str] = None) -> str:
        """Start a new conversation session."""
        if not session_id:
            session_id = f"session_{int(time.time())}"
        
        self.current_session = ConversationSession(
            session_id=session_id,
            start_time=time.time(),
            end_time=None,
            messages=[],
            project_path=project_path,
            file_access_patterns=defaultdict(int)
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO conversation_sessions 
                (session_id, start_time, project_path) 
                VALUES (?, ?, ?)
            ''', (session_id, self.current_session.start_time, project_path))
        
        logging.info(f"Started conversation session: {session_id}")
        return session_id
    
    def add_message(self, role: str, content: str, message_type: str = 'general',
                   context_files: Optional[List[str]] = None) -> ConversationMessage:
        """Add a message to the current conversation."""
        if not self.current_session:
            raise ValueError("No active conversation session")
        
        # Analyze message
        topics = self.analyzer.extract_topics(content)
        file_mentions = self.analyzer.extract_file_mentions(content)
        
        # Get embedding if service is available
        embedding = None
        if self.embedding_service:
            try:
                embeddings = asyncio.run(self.embedding_service.get_embeddings([content]))
                if embeddings:
                    embedding = embeddings[0]
            except Exception as e:
                logging.warning(f"Failed to get embedding for message: {e}")
        
        message = ConversationMessage(
            timestamp=time.time(),
            role=role,
            content=content,
            message_type=message_type,
            context_files=context_files or [],
            topics=topics,
            embedding=embedding
        )
        
        # Add to session and buffer
        self.current_session.messages.append(message)
        self.message_buffer.append(message)
        
        # Update file access patterns
        if context_files:
            for file_path in context_files:
                self.current_session.file_access_patterns[file_path] += 1
        
        # Store in database
        self._store_message(message)
        
        # Update file mentions as potential file access
        for mentioned_file in file_mentions:
            self._record_file_mention(mentioned_file)
        
        return message
    
    def _store_message(self, message: ConversationMessage):
        """Store a message in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO conversation_messages 
                (session_id, timestamp, role, content, message_type, context_files, topics, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_session.session_id,
                message.timestamp,
                message.role,
                message.content,
                message.message_type,
                json.dumps(message.context_files) if message.context_files else None,
                json.dumps(message.topics) if message.topics else None,
                json.dumps(message.embedding) if message.embedding else None
            ))
    
    def _record_file_mention(self, file_path: str):
        """Record that a file was mentioned in conversation."""
        if not self.current_session:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Update or insert file access pattern
            conn.execute('''
                INSERT OR REPLACE INTO file_access_patterns 
                (session_id, file_path, access_count, last_access_time, context_relevance)
                VALUES (?, ?, 
                    COALESCE((SELECT access_count FROM file_access_patterns 
                             WHERE session_id = ? AND file_path = ?), 0) + 1,
                    ?, 1.0)
            ''', (self.current_session.session_id, file_path, 
                  self.current_session.session_id, file_path, time.time()))
    
    async def get_context_adaptation(self, user_query: str, 
                                   available_files: List[str],
                                   current_context: Optional[List[str]] = None,
                                   max_files: int = 10) -> ContextAdaptation:
        """
        Get conversation-aware context adaptation recommendations.
        """
        if not self.current_session:
            # Return basic adaptation without conversation context
            return ContextAdaptation(
                recommended_files=available_files[:max_files],
                confidence_scores={f: 0.5 for f in available_files[:max_files]},
                reasoning=["No active conversation session"],
                adaptation_strategy="default",
                conversation_relevance=0.0
            )
        
        # Analyze current conversation context
        conversation_analysis = self.analyzer.analyze_conversation_flow(
            list(self.message_buffer)
        )
        
        # Extract topics from current query
        query_topics = self.analyzer.extract_topics(user_query)
        query_intent = self.analyzer.detect_intent(user_query)
        mentioned_files = self.analyzer.extract_file_mentions(user_query)
        
        # Get semantic relevance if embedding service is available
        semantic_scores = {}
        if self.embedding_service:
            semantic_scores = await self._get_semantic_relevance_scores(
                user_query, available_files
            )
        
        # Calculate file relevance scores
        file_scores = {}
        reasoning = []
        
        for file_path in available_files:
            score = 0.0
            file_reasoning = []
            
            # 1. Semantic relevance (40% weight)
            semantic_score = semantic_scores.get(file_path, 0.0)
            score += semantic_score * 0.4
            if semantic_score > 0.3:
                file_reasoning.append(f"Semantically relevant ({semantic_score:.2f})")
            
            # 2. Conversation history (30% weight)
            history_score = self._get_conversation_history_score(file_path, query_topics)
            score += history_score * 0.3
            if history_score > 0.2:
                file_reasoning.append(f"Relevant to conversation history ({history_score:.2f})")
            
            # 3. File mentions (20% weight)
            mention_score = 1.0 if any(mentioned in file_path for mentioned in mentioned_files) else 0.0
            score += mention_score * 0.2
            if mention_score > 0:
                file_reasoning.append("Explicitly mentioned in query")
            
            # 4. Topic alignment (10% weight)
            topic_score = self._get_topic_alignment_score(file_path, query_topics, conversation_analysis)
            score += topic_score * 0.1
            if topic_score > 0.3:
                file_reasoning.append(f"Aligns with conversation topics ({topic_score:.2f})")
            
            file_scores[file_path] = score
            if file_reasoning:
                reasoning.append(f"{file_path}: {', '.join(file_reasoning)}")
        
        # Sort and select top files
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        recommended_files = [f for f, _ in sorted_files[:max_files]]
        confidence_scores = dict(sorted_files[:max_files])
        
        # Determine adaptation strategy
        strategy = self._determine_adaptation_strategy(
            query_intent, query_topics, conversation_analysis
        )
        
        # Calculate overall conversation relevance
        conversation_relevance = self._calculate_conversation_relevance(
            query_topics, conversation_analysis
        )
        
        # Store adaptation for learning
        adaptation = ContextAdaptation(
            recommended_files=recommended_files,
            confidence_scores=confidence_scores,
            reasoning=reasoning[:10],  # Limit reasoning entries
            adaptation_strategy=strategy,
            conversation_relevance=conversation_relevance
        )
        
        self._store_context_adaptation(user_query, adaptation)
        
        return adaptation
    
    async def _get_semantic_relevance_scores(self, query: str, 
                                           available_files: List[str]) -> Dict[str, float]:
        """Get semantic relevance scores for files based on query."""
        if not self.embedding_service:
            return {}
        
        try:
            # Get query embedding
            query_embeddings = await self.embedding_service.get_embeddings([query])
            if not query_embeddings:
                return {}
            
            query_embedding = query_embeddings[0]
            scores = {}
            
            # For efficiency, we'll use cached embeddings from recent messages
            # that mentioned these files, or compute new ones for a sample
            sample_files = available_files[:20]  # Limit for performance
            
            for file_path in sample_files:
                # Try to find recent message embeddings that mentioned this file
                file_embedding = self._get_cached_file_embedding(file_path)
                
                if file_embedding:
                    similarity = calculate_cosine_similarity(query_embedding, file_embedding)
                    scores[file_path] = similarity
                else:
                    # For files without cached embeddings, use filename similarity
                    filename_similarity = self._calculate_filename_similarity(query, file_path)
                    scores[file_path] = filename_similarity * 0.3  # Lower weight for filename-only
            
            return scores
            
        except Exception as e:
            logging.warning(f"Failed to get semantic relevance scores: {e}")
            return {}
    
    def _get_cached_file_embedding(self, file_path: str) -> Optional[List[float]]:
        """Get cached embedding for a file from recent conversation messages."""
        for message in reversed(self.message_buffer):
            if (message.embedding and message.context_files and 
                file_path in message.context_files):
                return message.embedding
        return None
    
    def _calculate_filename_similarity(self, query: str, file_path: str) -> float:
        """Calculate similarity between query and filename."""
        query_words = set(re.findall(r'\w+', query.lower()))
        filename_words = set(re.findall(r'\w+', Path(file_path).stem.lower()))
        
        if not query_words or not filename_words:
            return 0.0
        
        intersection = query_words.intersection(filename_words)
        union = query_words.union(filename_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _get_conversation_history_score(self, file_path: str, query_topics: List[str]) -> float:
        """Get relevance score based on conversation history."""
        if not self.current_session:
            return 0.0
        
        score = 0.0
        
        # File access frequency
        access_count = self.current_session.file_access_patterns.get(file_path, 0)
        if access_count > 0:
            score += min(access_count * 0.1, 0.5)  # Cap at 0.5
        
        # Topic alignment with conversation history
        session_topics = []
        for message in self.current_session.messages[-10:]:  # Recent messages
            if message.topics:
                session_topics.extend(message.topics)
        
        if session_topics and query_topics:
            topic_overlap = len(set(query_topics).intersection(set(session_topics)))
            topic_score = topic_overlap / max(len(query_topics), len(session_topics))
            score += topic_score * 0.3
        
        return min(score, 1.0)
    
    def _get_topic_alignment_score(self, file_path: str, query_topics: List[str],
                                 conversation_analysis: Dict[str, Any]) -> float:
        """Get score based on topic alignment."""
        if not query_topics:
            return 0.0
        
        score = 0.0
        
        # Check if file path contains topic-related keywords
        file_path_lower = file_path.lower()
        for topic in query_topics:
            if topic in file_path_lower:
                score += 0.3
        
        # Check alignment with dominant conversation topics
        dominant_topics = conversation_analysis.get('dominant_topics', [])
        if dominant_topics:
            topic_overlap = len(set(query_topics).intersection(set(dominant_topics)))
            if topic_overlap > 0:
                score += topic_overlap / len(query_topics) * 0.4
        
        return min(score, 1.0)
    
    def _determine_adaptation_strategy(self, intent: str, topics: List[str],
                                     conversation_analysis: Dict[str, Any]) -> str:
        """Determine the best adaptation strategy."""
        if intent == 'question' and 'debugging' in topics:
            return 'debug_focused'
        elif intent == 'request' and 'implementation' in topics:
            return 'implementation_focused'
        elif 'understanding' in topics:
            return 'exploration_focused'
        elif conversation_analysis.get('context_switches', 0) > 2:
            return 'broad_context'
        elif conversation_analysis.get('dominant_topics'):
            return 'topic_focused'
        else:
            return 'balanced'
    
    def _calculate_conversation_relevance(self, query_topics: List[str],
                                        conversation_analysis: Dict[str, Any]) -> float:
        """Calculate how relevant the query is to the ongoing conversation."""
        if not query_topics:
            return 0.5
        
        dominant_topics = conversation_analysis.get('dominant_topics', [])
        if not dominant_topics:
            return 0.3
        
        topic_overlap = len(set(query_topics).intersection(set(dominant_topics)))
        relevance = topic_overlap / len(query_topics) if query_topics else 0.0
        
        # Boost relevance if conversation is recent and active
        recent_messages = len([m for m in self.message_buffer if m.timestamp > time.time() - 300])  # 5 min
        if recent_messages > 3:
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _store_context_adaptation(self, query: str, adaptation: ContextAdaptation):
        """Store context adaptation for learning."""
        if not self.current_session:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO context_adaptations 
                (session_id, timestamp, user_query, recommended_files, adaptation_strategy, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.current_session.session_id,
                time.time(),
                query,
                json.dumps(adaptation.recommended_files),
                adaptation.adaptation_strategy,
                adaptation.conversation_relevance
            ))
    
    def provide_feedback(self, adaptation_id: int, feedback: Dict[str, Any]):
        """Provide feedback on a context adaptation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE context_adaptations 
                SET user_feedback = ? 
                WHERE id = ?
            ''', (json.dumps(feedback), adaptation_id))
        
        # Learn from feedback (simplified implementation)
        self._learn_from_feedback(feedback)
    
    def _learn_from_feedback(self, feedback: Dict[str, Any]):
        """Learn from user feedback to improve future recommendations."""
        # This is a placeholder for more sophisticated learning
        # In a full implementation, this would update model parameters
        # or retrain recommendation algorithms
        
        helpful_files = feedback.get('helpful_files', [])
        unhelpful_files = feedback.get('unhelpful_files', [])
        
        # Simple learning: boost/penalize file patterns
        for file_path in helpful_files:
            # Increase affinity for this file pattern
            pass
        
        for file_path in unhelpful_files:
            # Decrease affinity for this file pattern
            pass
    
    def end_session(self):
        """End the current conversation session."""
        if not self.current_session:
            return
        
        self.current_session.end_time = time.time()
        
        # Update session summary
        topics = []
        for message in self.current_session.messages:
            if message.topics:
                topics.extend(message.topics)
        
        primary_topics = [topic for topic, _ in Counter(topics).most_common(5)]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE conversation_sessions 
                SET end_time = ?, primary_topics = ?, message_count = ?
                WHERE session_id = ?
            ''', (
                self.current_session.end_time,
                json.dumps(primary_topics),
                len(self.current_session.messages),
                self.current_session.session_id
            ))
        
        logging.info(f"Ended conversation session: {self.current_session.session_id}")
        self.current_session = None
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about conversation sessions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT COUNT(*) as total_sessions,
                       AVG(message_count) as avg_messages_per_session,
                       AVG(end_time - start_time) as avg_duration_seconds
                FROM conversation_sessions 
                WHERE end_time IS NOT NULL
            ''')
            
            session_stats = cursor.fetchone()
            
            cursor = conn.execute('''
                SELECT COUNT(*) as total_adaptations,
                       AVG(confidence_score) as avg_confidence
                FROM context_adaptations
            ''')
            
            adaptation_stats = cursor.fetchone()
        
        return {
            'total_sessions': session_stats[0] if session_stats else 0,
            'avg_messages_per_session': session_stats[1] if session_stats else 0,
            'avg_session_duration_minutes': (session_stats[2] / 60) if session_stats and session_stats[2] else 0,
            'total_adaptations': adaptation_stats[0] if adaptation_stats else 0,
            'avg_adaptation_confidence': adaptation_stats[1] if adaptation_stats else 0,
            'current_session_active': self.current_session is not None
        }