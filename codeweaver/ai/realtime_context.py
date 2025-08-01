"""
Real-time context updates as conversation evolves.
Provides streaming updates to context selection based on ongoing conversation.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Set
from dataclasses import dataclass, asdict
from collections import deque
import weakref

from .conversation_tracker import ConversationTracker, ConversationMessage, ContextAdaptation
from .embeddings import GeminiEmbeddingService

@dataclass
class ContextUpdate:
    """Represents a real-time context update."""
    timestamp: float
    session_id: str
    update_type: str           # 'addition', 'removal', 'rerank', 'confidence_change'
    affected_files: List[str]
    new_context: List[str]     # Current recommended context
    confidence_scores: Dict[str, float]
    reasoning: List[str]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class StreamingClient:
    """Represents a client subscribed to real-time updates."""
    client_id: str
    session_id: str
    callback: Callable[[ContextUpdate], None]
    filters: Dict[str, Any]    # Client-specific filtering preferences
    last_update: float

class RealTimeContextManager:
    """
    Manages real-time context updates as conversations evolve.
    Provides streaming updates to subscribed clients.
    """
    
    def __init__(self, conversation_tracker: ConversationTracker,
                 embedding_service: Optional[GeminiEmbeddingService] = None):
        self.conversation_tracker = conversation_tracker
        self.embedding_service = embedding_service
        
        # Client management
        self.clients: Dict[str, StreamingClient] = {}
        self.session_clients: Dict[str, Set[str]] = {}  # session_id -> client_ids
        
        # Update management
        self.update_queue = asyncio.Queue()
        self.context_cache: Dict[str, ContextAdaptation] = {}  # session_id -> latest context
        self.message_buffer: Dict[str, deque] = {}  # session_id -> recent messages
        
        # Configuration
        self.update_threshold = 0.1        # Minimum change to trigger update
        self.max_update_frequency = 2.0    # Max updates per second per client
        self.context_window_size = 10      # Number of recent messages to consider
        self.similarity_threshold = 0.3    # Threshold for semantic similarity
        
        # Background task management
        self.update_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Weak references to avoid memory leaks
        self._session_refs = weakref.WeakValueDictionary()
    
    async def start(self):
        """Start the real-time context manager."""
        if self.running:
            return
        
        self.running = True
        self.update_task = asyncio.create_task(self._process_updates())
        logging.info("Real-time context manager started")
    
    async def stop(self):
        """Stop the real-time context manager."""
        self.running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        logging.info("Real-time context manager stopped")
    
    async def subscribe_client(self, client_id: str, session_id: str,
                               callback: Callable[[ContextUpdate], None],
                               filters: Optional[Dict[str, Any]] = None) -> bool:
        """Subscribe a client to real-time context updates."""
        try:
            client = StreamingClient(
                client_id=client_id,
                session_id=session_id,
                callback=callback,
                filters=filters or {},
                last_update=time.time()
            )
            
            self.clients[client_id] = client
            
            # Add to session tracking
            if session_id not in self.session_clients:
                self.session_clients[session_id] = set()
            self.session_clients[session_id].add(client_id)
            
            logging.info(f"Client {client_id} subscribed to session {session_id}")
            
            # Send initial context if available
            if session_id in self.context_cache:
                initial_update = ContextUpdate(
                    timestamp=time.time(),
                    session_id=session_id,
                    update_type='initial',
                    affected_files=[],
                    new_context=self.context_cache[session_id].recommended_files,
                    confidence_scores=self.context_cache[session_id].confidence_scores,
                    reasoning=["Initial context load"],
                    metadata={"adaptation_strategy": self.context_cache[session_id].adaptation_strategy}
                )
                await self._send_update_to_client(client, initial_update)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to subscribe client {client_id}: {e}")
            return False
    
    def unsubscribe_client(self, client_id: str):
        """Unsubscribe a client from updates."""
        if client_id in self.clients:
            client = self.clients[client_id]
            session_id = client.session_id
            
            # Remove from clients
            del self.clients[client_id]
            
            # Remove from session tracking
            if session_id in self.session_clients:
                self.session_clients[session_id].discard(client_id)
                if not self.session_clients[session_id]:
                    del self.session_clients[session_id]
            
            logging.info(f"Client {client_id} unsubscribed")
    
    async def on_message_added(self, message: ConversationMessage, session_id: str):
        """Handle new message addition to trigger context updates."""
        if not self.running:
            return
        
        # Add to message buffer
        if session_id not in self.message_buffer:
            self.message_buffer[session_id] = deque(maxlen=self.context_window_size)
        
        self.message_buffer[session_id].append(message)
        
        # Queue update for processing
        await self.update_queue.put({
            'type': 'message_added',
            'session_id': session_id,
            'message': message,
            'timestamp': time.time()
        })
    
    async def on_context_changed(self, session_id: str, new_context: ContextAdaptation,
                                previous_context: Optional[ContextAdaptation] = None):
        """Handle explicit context changes."""
        if not self.running:
            return
        
        # Update cache
        self.context_cache[session_id] = new_context
        
        # Queue update for processing
        await self.update_queue.put({
            'type': 'context_changed',
            'session_id': session_id,
            'new_context': new_context,
            'previous_context': previous_context,
            'timestamp': time.time()
        })
    
    async def _process_updates(self):
        """Background task to process context updates."""
        while self.running:
            try:
                # Wait for updates with timeout
                update_data = await asyncio.wait_for(
                    self.update_queue.get(), 
                    timeout=1.0
                )
                
                if update_data['type'] == 'message_added':
                    await self._handle_message_update(update_data)
                elif update_data['type'] == 'context_changed':
                    await self._handle_context_change(update_data)
                
            except asyncio.TimeoutError:
                # Periodic maintenance
                await self._cleanup_inactive_clients()
            except Exception as e:
                logging.error(f"Error processing context update: {e}")
    
    async def _handle_message_update(self, update_data: Dict[str, Any]):
        """Handle updates triggered by new messages."""
        session_id = update_data['session_id']
        message = update_data['message']
        
        # Check if this session has subscribed clients
        if session_id not in self.session_clients or not self.session_clients[session_id]:
            return
        
        # Analyze if the new message should trigger context updates
        should_update = await self._should_trigger_update(session_id, message)
        
        if should_update:
            # Generate new context adaptation
            try:
                # Get current available files (this would need to be passed or cached)
                # For now, we'll simulate with cached context
                if session_id in self.context_cache:
                    current_context = self.context_cache[session_id]
                    
                    # Re-evaluate context with new message
                    new_adaptation = await self.conversation_tracker.get_context_adaptation(
                        user_query=message.content,
                        available_files=current_context.recommended_files,
                        max_files=len(current_context.recommended_files)
                    )
                    
                    # Compare with previous context
                    changes = self._analyze_context_changes(current_context, new_adaptation)
                    
                    if changes['significant']:
                        # Create and send update
                        context_update = ContextUpdate(
                            timestamp=time.time(),
                            session_id=session_id,
                            update_type=changes['type'],
                            affected_files=changes['affected_files'],
                            new_context=new_adaptation.recommended_files,
                            confidence_scores=new_adaptation.confidence_scores,
                            reasoning=changes['reasoning'],
                            metadata={
                                'trigger_message': message.content[:100],
                                'adaptation_strategy': new_adaptation.adaptation_strategy
                            }
                        )
                        
                        await self._broadcast_update(session_id, context_update)
                        
                        # Update cache
                        self.context_cache[session_id] = new_adaptation
                
            except Exception as e:
                logging.error(f"Failed to handle message update for session {session_id}: {e}")
    
    async def _handle_context_change(self, update_data: Dict[str, Any]):
        """Handle explicit context changes."""
        session_id = update_data['session_id']
        new_context = update_data['new_context']
        previous_context = update_data.get('previous_context')
        
        # Check if this session has subscribed clients
        if session_id not in self.session_clients or not self.session_clients[session_id]:
            return
        
        # Analyze changes
        changes = self._analyze_context_changes(previous_context, new_context)
        
        # Create and send update
        context_update = ContextUpdate(
            timestamp=time.time(),
            session_id=session_id,
            update_type=changes['type'],
            affected_files=changes['affected_files'],
            new_context=new_context.recommended_files,
            confidence_scores=new_context.confidence_scores,
            reasoning=changes['reasoning'],
            metadata={
                'adaptation_strategy': new_context.adaptation_strategy,
                'conversation_relevance': new_context.conversation_relevance
            }
        )
        
        await self._broadcast_update(session_id, context_update)
    
    async def _should_trigger_update(self, session_id: str, message: ConversationMessage) -> bool:
        """Determine if a new message should trigger context updates."""
        # Always update for user messages with questions
        if message.role == 'user' and ('?' in message.content or 
                                      any(word in message.content.lower() 
                                          for word in ['how', 'what', 'why', 'where', 'when'])):
            return True
        
        # Update for messages with high topic diversity
        if message.topics and len(message.topics) > 2:
            return True
        
        # Update for messages mentioning specific files
        if self._extract_file_mentions(message.content):
            return True
        
        # Check semantic similarity with recent messages
        if session_id in self.message_buffer and len(self.message_buffer[session_id]) > 1:
            recent_messages = list(self.message_buffer[session_id])[-3:]  # Last 3 messages
            
            if await self._has_semantic_shift(message, recent_messages):
                return True
        
        # Rate limiting - don't update too frequently
        if session_id in self.context_cache:
            last_update_time = getattr(self.context_cache[session_id], 'timestamp', 0)
            if time.time() - last_update_time < 1.0 / self.max_update_frequency:
                return False
        
        return False
    
    async def _has_semantic_shift(self, new_message: ConversationMessage, 
                                 recent_messages: List[ConversationMessage]) -> bool:
        """Check if the new message represents a semantic shift in conversation."""
        if not self.embedding_service or not new_message.embedding:
            return False
        
        try:
            # Get embeddings for recent messages
            recent_embeddings = [msg.embedding for msg in recent_messages 
                               if msg.embedding is not None]
            
            if not recent_embeddings:
                return False
            
            # Calculate average similarity with recent messages
            from .embeddings import calculate_cosine_similarity
            similarities = [
                calculate_cosine_similarity(new_message.embedding, recent_emb)
                for recent_emb in recent_embeddings
            ]
            
            avg_similarity = sum(similarities) / len(similarities)
            
            # If similarity is below threshold, it's a semantic shift
            return avg_similarity < self.similarity_threshold
            
        except Exception as e:
            logging.warning(f"Failed to calculate semantic shift: {e}")
            return False
    
    def _extract_file_mentions(self, content: str) -> List[str]:
        """Extract file mentions from message content."""
        import re
        
        # Basic file mention patterns
        patterns = [
            r'\b(\w+\.\w+)\b',  # filename.ext
            r'`([^`]+\.\w+)`',  # `filename.ext`
            r'"([^"]+\.\w+)"',  # "filename.ext"
        ]
        
        mentions = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            mentions.extend(matches)
        
        return mentions
    
    def _analyze_context_changes(self, previous: Optional[ContextAdaptation], 
                                current: ContextAdaptation) -> Dict[str, Any]:
        """Analyze changes between context adaptations."""
        if not previous:
            return {
                'significant': True,
                'type': 'initial',
                'affected_files': current.recommended_files,
                'reasoning': ['Initial context establishment']
            }
        
        prev_files = set(previous.recommended_files)
        curr_files = set(current.recommended_files)
        
        added_files = curr_files - prev_files
        removed_files = prev_files - curr_files
        
        # Check for significant confidence changes
        confidence_changes = []
        for file_path in prev_files.intersection(curr_files):
            prev_conf = previous.confidence_scores.get(file_path, 0.5)
            curr_conf = current.confidence_scores.get(file_path, 0.5)
            
            if abs(prev_conf - curr_conf) > self.update_threshold:
                confidence_changes.append(file_path)
        
        # Determine update type and significance
        if added_files or removed_files:
            significant = True
            if added_files and not removed_files:
                update_type = 'addition'
            elif removed_files and not added_files:
                update_type = 'removal'
            else:
                update_type = 'rerank'
        elif confidence_changes:
            significant = len(confidence_changes) > len(curr_files) * 0.3  # 30% threshold
            update_type = 'confidence_change'
        else:
            significant = False
            update_type = 'minor'
        
        # Generate reasoning
        reasoning = []
        if added_files:
            reasoning.append(f"Added {len(added_files)} files to context")
        if removed_files:
            reasoning.append(f"Removed {len(removed_files)} files from context")
        if confidence_changes:
            reasoning.append(f"Updated confidence for {len(confidence_changes)} files")
        
        return {
            'significant': significant,
            'type': update_type,
            'affected_files': list(added_files.union(removed_files).union(confidence_changes)),
            'reasoning': reasoning or ['Minor context adjustments']
        }
    
    async def _broadcast_update(self, session_id: str, update: ContextUpdate):
        """Broadcast an update to all clients subscribed to the session."""
        if session_id not in self.session_clients:
            return
        
        client_ids = list(self.session_clients[session_id])  # Copy to avoid modification during iteration
        
        for client_id in client_ids:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # Check rate limiting
                time_since_last = time.time() - client.last_update
                if time_since_last < 1.0 / self.max_update_frequency:
                    continue
                
                # Apply client-specific filters
                if self._should_send_to_client(client, update):
                    await self._send_update_to_client(client, update)
    
    def _should_send_to_client(self, client: StreamingClient, update: ContextUpdate) -> bool:
        """Check if an update should be sent to a specific client based on filters."""
        filters = client.filters
        
        # Update type filter
        if 'update_types' in filters:
            if update.update_type not in filters['update_types']:
                return False
        
        # Minimum confidence filter
        if 'min_confidence' in filters:
            max_confidence = max(update.confidence_scores.values()) if update.confidence_scores else 0
            if max_confidence < filters['min_confidence']:
                return False
        
        # File pattern filter
        if 'file_patterns' in filters:
            patterns = filters['file_patterns']
            if not any(pattern in file_path for pattern in patterns 
                      for file_path in update.new_context):
                return False
        
        return True
    
    async def _send_update_to_client(self, client: StreamingClient, update: ContextUpdate):
        """Send an update to a specific client."""
        try:
            client.callback(update)
            client.last_update = time.time()
        except Exception as e:
            logging.error(f"Failed to send update to client {client.client_id}: {e}")
            # Consider removing client if callback consistently fails
    
    async def _cleanup_inactive_clients(self):
        """Remove clients that haven't received updates recently."""
        current_time = time.time()
        inactive_threshold = 300  # 5 minutes
        
        inactive_clients = []
        for client_id, client in self.clients.items():
            if current_time - client.last_update > inactive_threshold:
                inactive_clients.append(client_id)
        
        for client_id in inactive_clients:
            self.unsubscribe_client(client_id)
            logging.info(f"Removed inactive client: {client_id}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of sessions with active clients."""
        return list(self.session_clients.keys())
    
    def get_client_count(self, session_id: Optional[str] = None) -> int:
        """Get number of active clients, optionally filtered by session."""
        if session_id:
            return len(self.session_clients.get(session_id, set()))
        return len(self.clients)
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get statistics about the real-time context manager."""
        return {
            'active_clients': len(self.clients),
            'active_sessions': len(self.session_clients),
            'cached_contexts': len(self.context_cache),
            'running': self.running,
            'update_queue_size': self.update_queue.qsize(),
            'message_buffers': len(self.message_buffer)
        }