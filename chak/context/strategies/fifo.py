# chak/context/strategies/fifo.py
"""FIFO (First In First Out) context management strategy."""

from typing import List, Optional, Callable

from .base import BaseContextStrategy, StrategyRequest, StrategyResponse
from ...exceptions import ContextError
from ...message import Message, MarkerMessage, SystemMessage, HumanMessage


class FIFOStrategy(BaseContextStrategy):
    """
    FIFOStrategy - FIFO (First In First Out) context strategy.
    
    Purpose:
    - Truncate older conversation messages based on turn count or token limits.
    - Insert a truncate marker at the truncation point to preserve audit trail.
    
    Semantics:
    - StrategyResponse.messages: returns original messages with a truncate marker inserted
      at the boundary where old messages are cut off.
    - Conversation will extract: System messages + last marker (inclusive) -> end.
    
    Turn Definition:
    - A "turn" is defined by HumanMessage boundaries: from a HumanMessage (inclusive)
      up to (but not including) the next HumanMessage.
    - Example: HumanMessage -> AIMessage -> ... -> next HumanMessage
    
    Truncation Logic:
    - Find the preserve boundary (Nth HumanMessage from the end, where N = keep_recent_turns + 1).
    - Insert a TruncateMarker before the preserve boundary.
    - If max_input_tokens is specified, ensure total tokens don't exceed the limit.
    
    Token Control:
    - If tokens exceed max_input_tokens after applying keep_recent_turns, drop older turns.
    - Always preserve at least the most recent message to avoid empty context.
    
    Parameters:
    - keep_recent_turns (Optional[int]): number of recent turns to keep.
    - max_input_tokens (Optional[int]): upper bound for input tokens.
    - max_output_tokens (Optional[int]): declarative only; not enforced here.
    - token_counter (Optional[Callable[[str], int]]): custom token counter.
    
    Notes:
    - System messages are always included when Conversation extracts messages.
    - The truncate marker has content="从此处截断" and metadata with truncation details.
    - Original messages before the marker are preserved for audit purposes.
    """
    
    def __init__(
        self,
        keep_recent_turns: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        token_counter: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize FIFOStrategy.
        
        Args:
            keep_recent_turns: Number of recent conversation turns to keep
                         (HumanMessage-based turn counting)
            max_input_tokens: Maximum input token count (context limit)
            max_output_tokens: Maximum output token count (declarative, for documentation)
            token_counter: Custom token counting function
            
        Note:
            - If only keep_recent_turns is set: limit by turns only
            - If only max_input_tokens is set: limit by token count only
            - If both are set: satisfy both constraints
            - System messages are always preserved
            
        Raises:
            ValueError: If neither keep_recent_turns nor max_input_tokens is specified
        """
        super().__init__(token_counter=token_counter)
        
        self.keep_recent_turns = keep_recent_turns
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        
        # Validate parameters
        if keep_recent_turns is None and max_input_tokens is None:
            raise ValueError(
                "At least one of keep_recent_turns or max_input_tokens must be specified"
            )
    
    def process(self, request: StrategyRequest) -> StrategyResponse:
        """
        Process messages according to FIFO strategy.
        
        Logic:
        1. Extract system messages and conversation messages
        2. If no truncation needed, return original messages
        3. Find preserve boundary based on keep_recent_turns
        4. Apply token limit if specified
        5. Insert truncate marker at the boundary
        6. Return messages with marker inserted
        
        Args:
            request: Strategy request containing messages
            
        Returns:
            Strategy response with truncate marker inserted if truncation occurred
        """
        messages = request.messages
        
        if not messages:
            return StrategyResponse(messages=[])
        
        # 1. Separate by type
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        conversation_messages = [
            m for m in messages 
            if not isinstance(m, (SystemMessage, MarkerMessage))
        ]
        
        if not conversation_messages:
            return StrategyResponse(messages=messages)
        
        # 2. Check if truncation needed
        need_truncate = False
        preserve_start_idx = None
        
        # 2a. Check turn-based truncation
        if self.keep_recent_turns is not None:
            preserve_start_idx = self._find_preserve_start(list(conversation_messages))
            if preserve_start_idx is not None and preserve_start_idx > 0:
                need_truncate = True
        
        # 2b. Check token-based truncation
        if self.max_input_tokens is not None:
            # Calculate current tokens
            messages_to_check = list(system_messages) + list(conversation_messages)
            total_tokens = self.count_messages_tokens(list(messages_to_check))
            
            if total_tokens > self.max_input_tokens:
                need_truncate = True
                # Find preserve boundary that satisfies token limit
                preserve_start_idx = self._find_preserve_start_by_tokens(
                    list(system_messages),
                    list(conversation_messages)
                )
        
        # 3. If no truncation needed, return original
        if not need_truncate or preserve_start_idx is None or preserve_start_idx == 0:
            return StrategyResponse(messages=messages)
        
        # 4. Build truncation reason string
        reason_parts = []
        if self.keep_recent_turns is not None:
            reason_parts.append(f"keep_recent_turns={self.keep_recent_turns}")
        if self.max_input_tokens is not None:
            reason_parts.append(f"max_input_tokens={self.max_input_tokens}")
        reason = "FIFO truncation: " + ", ".join(reason_parts)
        
        # 5. Create truncate marker
        truncated_count = preserve_start_idx
        marker = MarkerMessage(
            content="",
            metadata={
                "type": "truncate",
                "truncated_count": truncated_count,
                "reason": reason
            }
        )
        
        # 6. Find insertion position in original messages
        preserve_message = conversation_messages[preserve_start_idx]
        insert_idx = self._find_message_index_in_original(messages, preserve_message)
        
        # 7. Build new messages with marker inserted
        new_messages = (
            messages[:insert_idx] + 
            [marker] + 
            messages[insert_idx:]
        )
        
        return StrategyResponse(messages=new_messages)
    
    def _find_preserve_start(self, conversation_messages: List[Message]) -> Optional[int]:
        """
        Find the start index of messages to preserve based on keep_recent_turns.
        
        Logic: Find the (keep_recent_turns + 1)th HumanMessage from the end.
        
        Args:
            conversation_messages: Conversation messages (excluding system and markers)
            
        Returns:
            Start index in conversation_messages, or None if no truncation needed
        """
        if not conversation_messages or self.keep_recent_turns is None:
            return None
        
        # Find HumanMessage positions from end to start
        human_indices = []
        for i in range(len(conversation_messages) - 1, -1, -1):
            if isinstance(conversation_messages[i], HumanMessage):
                human_indices.append(i)
                # Found the (keep_recent_turns + 1)th HumanMessage
                if len(human_indices) == self.keep_recent_turns + 1:
                    return human_indices[-1]  # Return the earliest one
        
        # Not enough turns to truncate
        return None
    
    def _find_preserve_start_by_tokens(
        self,
        system_messages: List[Message],
        conversation_messages: List[Message]
    ) -> Optional[int]:
        """
        Find the start index of messages to preserve based on token limit.
        
        Logic: Keep adding messages from the end until token limit is reached.
        
        Args:
            system_messages: System messages (always included)
            conversation_messages: Conversation messages
            
        Returns:
            Start index in conversation_messages, or None if all messages fit
        """
        if not conversation_messages or self.max_input_tokens is None:
            return None
        
        system_tokens = self.count_messages_tokens(list(system_messages))
        remaining_budget = self.max_input_tokens - system_tokens
        
        if remaining_budget <= 0:
            # System messages already exceed limit, keep only last message
            return len(conversation_messages) - 1
        
        # Add messages from end to start
        current_tokens = 0
        for i in range(len(conversation_messages) - 1, -1, -1):
            msg_tokens = 4 + self.count_tokens(conversation_messages[i].content or "")
            if current_tokens + msg_tokens <= remaining_budget:
                current_tokens += msg_tokens
            else:
                # Found the boundary, preserve from i+1 onwards
                return i + 1
        
        # All messages fit within budget
        return None
    
    def _find_message_index_in_original(
        self, 
        original_messages: List[Message], 
        target_message: Message
    ) -> int:
        """
        Find the index of target message in original message list.
        
        Args:
            original_messages: Original message list
            target_message: Target message to find
            
        Returns:
            Index of target message in original list
            
        Raises:
            ContextError: If target message not found
        """
        for i, msg in enumerate(original_messages):
            if msg is target_message:
                return i
        raise ContextError("Target message not found in original messages")

