# chak/context/handlers/base.py
"""Base class for context management handlers."""

from abc import ABC, abstractmethod
from typing import List, Set
from copy import deepcopy

from ...message import Message, AIMessage, ToolMessage


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
        processed_messages = self._ensure_tool_call_integrity(processed_messages)
        self.output_messages = deepcopy(processed_messages)
        return processed_messages

    def _ensure_tool_call_integrity(self, messages: List[Message]) -> List[Message]:
        """Remove orphaned tool messages after context truncation.

        OpenAI requires each tool message to be preceded by an assistant
        message that contains a matching tool_call_id. When context handlers
        truncate the assistant message but keep the tool responses, the API
        returns HTTP 400: "messages with role 'tool' must be a response to a
        preceding message with 'tool_calls'".

        This method scans the result in order and silently drops any tool
        message whose tool_call_id was never registered by a prior assistant
        message in the same context window.

        Args:
            messages: Context messages produced by handle()

        Returns:
            Messages with orphaned tool messages removed
        """
        result: List[Message] = []
        pending_tool_call_ids: Set[str] = set()

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # Register all tool_call_ids this assistant expects responses for
                for tc in msg.tool_calls:
                    pending_tool_call_ids.add(tc.id)
                result.append(msg)
            elif isinstance(msg, ToolMessage):
                if msg.tool_call_id and msg.tool_call_id in pending_tool_call_ids:
                    # Valid: matched assistant tool_call exists earlier in context
                    pending_tool_call_ids.discard(msg.tool_call_id)
                    result.append(msg)
                # else: orphaned tool message — drop it silently
            else:
                result.append(msg)

        return result
    
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
