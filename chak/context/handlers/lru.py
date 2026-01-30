# chak/context/handlers/lru.py
"""LRU (Least Recently Used) context handler."""

from typing import List

from .base import BaseContextHandler
from ...message import Message, SystemMessage


class LRUContextHandler(BaseContextHandler):
    """
    LRUContextHandler - Keep recently used (accessed) messages.
    
    Behavior:
    - Wraps SummarizationContextHandler
    - After summarization, analyzes which topics are still being discussed
    - Only keeps "hot" topics in the summary, drops "cold" topics
    
    Note: This is a simplified placeholder implementation.
          Full LRU logic requires topic detection and tracking,
          which needs more sophisticated implementation.
    
    For now, it just delegates to regular message count-based filtering.
    """
    
    def __init__(self, keep_recent: int = 10):
        """
        Initialize LRU handler.
        
        Args:
            keep_recent: Number of recent messages to keep
        """
        super().__init__()
        self.keep_recent = keep_recent
    
    def handle(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """
        Return recently used messages.
        
        Simplified implementation: just keep last N messages.
        A full implementation would analyze topic continuity.
        
        Args:
            messages: Complete conversation history
            conversation_id: Unique ID for this conversation
            
        Returns:
            Recently used messages
        """
        if not messages:
            return []
        
        # Separate system messages and conversation messages
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        conversation_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        
        # Keep only recent messages (simplified LRU)
        if len(conversation_messages) <= self.keep_recent:
            return messages
        
        recent_messages = conversation_messages[-self.keep_recent:]
        return system_messages + recent_messages
