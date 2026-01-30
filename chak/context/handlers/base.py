# chak/context/handlers/base.py
"""Base class for context management handlers."""

from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy

from ...message import Message


class BaseContextHandler(ABC):
    """
    Base class for context management handlers.
    
    Design principles:
    - Input: complete messages + conversation_id
    - Output: context_messages for this round of LLM call
    - Handler can freely add/delete/modify messages in the output
    - chak only validates message types, no correctness guarantee
    
    All context handlers must inherit from this class and implement
    the handle method.
    """
    
    def __init__(self):
        """Initialize handler."""
        self.input_messages = []
        self.output_messages = []
    
    def __call__(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """
        Make handler callable, delegate to handle method.
        
        Args:
            messages: Complete conversation history (read-only)
            conversation_id: Unique ID for this conversation
            
        Returns:
            context_messages: Messages to send to LLM in this round
        """
        self.input_messages = deepcopy(messages)
        processed_messages = self.handle(messages, conversation_id=conversation_id)
        self.output_messages = deepcopy(processed_messages)
        return processed_messages
    
    @abstractmethod
    def handle(
        self,
        messages: List[Message],
        *,
        conversation_id: str,
    ) -> List[Message]:
        """
        Process messages and return context for this round.
        
        Different handlers may apply different logic:
        - Noop: Return all messages unchanged
        - FIFO: Keep recent messages, drop old ones
        - LRU: Keep frequently accessed messages
        - Summarize: Compress old messages into summaries
        - Offload: Move large content to external storage
        
        Args:
            messages: Complete conversation history (read-only snapshot)
            conversation_id: Unique ID for this conversation (for persistence/offload)
            
        Returns:
            context_messages: Messages to send to LLM in this round
        """
        pass
