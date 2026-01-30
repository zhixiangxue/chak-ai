# chak/context/handlers/noop.py
"""Noop (No Operation) context handler that passes through all messages."""

from typing import List

from .base import BaseContextHandler
from ...message import Message


class NoopContextHandler(BaseContextHandler):
    """
    NoopContextHandler (No Operation)
    
    Purpose:
    - Provide a pass-through context handler that performs no filtering
      or transformation of the conversation messages.
    
    Behavior:
    - Returns the original history_messages unchanged as context_messages.
    
    Notes:
    - Intended for debugging, baseline comparison, or scenarios where
      the caller wants full history sent to the LLM.
    - All message types (System/Human/AI/Tool) are passed through as-is.
    """
    
    def __init__(self):
        """Initialize the noop handler."""
        super().__init__()
    
    def handle(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """
        Return all messages without any processing.
        
        Args:
            messages: Complete conversation history
            conversation_id: Unique ID for this conversation
            
        Returns:
            All messages unchanged
        """
        return messages
