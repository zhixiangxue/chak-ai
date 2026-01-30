# chak/context/handlers/fifo.py
"""FIFO (First In First Out) context handler."""

from typing import List

from .base import BaseContextHandler
from ...message import Message, SystemMessage, HumanMessage


class FIFOContextHandler(BaseContextHandler):
    """
    FIFOContextHandler - Keep only recent conversation turns.
    
    Behavior:
    - Returns only the last N turns of conversation (turn = from HumanMessage to next HumanMessage)
    - SystemMessage are always included
    - No markers, no token counting, just pure message filtering
    
    Parameters:
    - keep_recent_turns: Number of recent turns to keep
    """
    
    def __init__(self, keep_recent_turns: int):
        """
        Initialize FIFO handler.
        
        Args:
            keep_recent_turns: Number of recent conversation turns to keep
        """
        super().__init__()
        if keep_recent_turns < 1:
            raise ValueError("keep_recent_turns must be >= 1")
        
        self.keep_recent_turns = keep_recent_turns
    
    def handle(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """
        Return system messages + last N turns.
        
        Args:
            messages: Complete conversation history
            conversation_id: Unique ID for this conversation
            
        Returns:
            Filtered messages (system + recent turns)
        """
        if not messages:
            return []
        
        # Separate system messages and conversation messages
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        conversation_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        
        if not conversation_messages:
            return system_messages
        
        # Find preserve boundary (Nth HumanMessage from end)
        preserve_start_idx = self._find_preserve_start(conversation_messages)
        
        if preserve_start_idx is None or preserve_start_idx == 0:
            # Keep all conversation messages
            return system_messages + conversation_messages
        
        # Keep only messages from preserve_start_idx onwards
        preserved_messages = conversation_messages[preserve_start_idx:]
        return system_messages + preserved_messages
    
    def _find_preserve_start(self, conversation_messages: List[Message]) -> int:
        """
        Find the start index of messages to preserve based on keep_recent_turns.
        
        Logic: Find the (keep_recent_turns + 1)th HumanMessage from the end.
        
        Args:
            conversation_messages: Conversation messages (excluding system)
            
        Returns:
            Start index in conversation_messages, or 0 if no truncation needed
        """
        if not conversation_messages:
            return 0
        
        # Find HumanMessage positions from end to start
        human_indices = []
        for i in range(len(conversation_messages) - 1, -1, -1):
            if isinstance(conversation_messages[i], HumanMessage):
                human_indices.append(i)
                # Found the (keep_recent_turns + 1)th HumanMessage
                if len(human_indices) == self.keep_recent_turns + 1:
                    return human_indices[-1]  # Return the earliest one
        
        # Not enough turns to truncate
        return 0
