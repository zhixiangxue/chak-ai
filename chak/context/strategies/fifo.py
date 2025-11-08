# chak/context/strategies/fifo.py
"""FIFO (First In First Out) context management strategy."""

from typing import List, Optional, Callable
from .base import BaseContextStrategy, StrategyRequest, StrategyResponse
from ...message import Message, MarkerMessage, SystemMessage, HumanMessage
from ...exceptions import ContextError


class FIFOStrategy(BaseContextStrategy):
    """
    FIFOStrategy - FIFO (First In First Out) context strategy.
    
    Purpose:
    - Determine the next messages to send to the LLM without mutating the conversation history.
    - Preserve all original conversation messages; do NOT insert any marker messages.
    
    Semantics:
    - StrategyResponse.messages: always returns the original messages unchanged.
    - StrategyResponse.messages_to_send: computed view to send this round.
      It consists of:
        1) All SystemMessage(s)
        2) The most recent turns (bounded by keep_recent_turns)
        3) Further trimmed to satisfy max_input_tokens if specified
    
    Turn Definition:
    - A "turn" is defined by HumanMessage boundaries: from a HumanMessage (inclusive)
      up to (but not including) the next HumanMessage.
    - This allows a turn to contain the following sequence:
      HumanMessage -> (Assistant / Tool messages) ... -> next HumanMessage
    
    Token Control:
    - If tokens exceed max_input_tokens, the oldest turn is dropped first.
    - If only the latest turn remains but still exceeds the limit, it is trimmed
      internally by keeping the most recent messages until within the limit (always
      preserving at least the most recent message to avoid empty context).
    
    Parameters:
    - keep_recent_turns (Optional[int]): number of recent turns to keep.
    - max_input_tokens (Optional[int]): upper bound for input tokens.
    - max_output_tokens (Optional[int]): declarative only; not enforced here.
    - token_counter (Optional[Callable[[str], int]]): custom token counter.
    
    Notes:
    - System messages are always included in messages_to_send.
    - MarkerMessage(s) are ignored by FIFO; no insertion or transformation is performed.
    - This strategy does not change the conversation history (no mutation).
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
                         (HumanMessage-to-HumanMessage segments)
            max_input_tokens: Maximum input token count (context limit)
            max_output_tokens: Maximum output token count (declarative, for documentation)
            token_counter: Custom token counting function
            
        Note:
            - If only keep_recent_turns is set: limit by turns only
            - If only max_input_tokens is set: limit by token count only
            - If both are set: satisfy both constraints
            - System messages are always preserved
            
        Raises:
            ValueError: If neither max_messages nor max_input_tokens is specified
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
        
        Args:
            request: Strategy request containing messages
            
        Returns:
            Strategy response with processed messages
        """
        messages = request.messages
        
        if not messages:
            return StrategyResponse(messages=[], messages_to_send=[])
        
        # 1. Separate by type
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        marker_messages = [m for m in messages if isinstance(m, MarkerMessage)]
        conversation_messages = [
            m for m in messages 
            if not isinstance(m, (SystemMessage, MarkerMessage))
        ]
        
        # 2. Determine messages to analyze (FIFO does not consider markers)
        to_analyze = conversation_messages
        base_messages = []
        
        # 4. Check if truncation needed
        need_truncate = False
        if self.keep_recent_turns is not None:
            need_truncate = True
        elif self.max_input_tokens:
            pass
        
        # 5. Build response (no marker insertion)
        # Split conversation messages into turns by HumanMessage boundaries
        turns = self._split_into_turns(list(to_analyze))
        
        # Apply max_turns if specified
        if self.keep_recent_turns is not None and self.keep_recent_turns > 0:
            turns = turns[-self.keep_recent_turns:]
        
        # Flatten selected turns
        selected: List[Message] = []
        for t in turns:
            selected.extend(t)
        
        # Apply token limit if specified
        if self.max_input_tokens is not None:
            while True:
                combined: List[Message] = list(system_messages) + list(selected)
                total_tokens = self.count_messages_tokens(combined)
                if total_tokens <= self.max_input_tokens:
                    break
                # If tokens exceed, drop oldest turn first
                if turns:
                    turns = turns[1:]
                    selected = []
                    for t in turns:
                        selected.extend(t)
                else:
                    # Fallback: limit within selected messages (keep most recent)
                    # Reserve tokens for system messages
                    system_tokens = self.count_messages_tokens(list(system_messages))
                    remaining = max(0, self.max_input_tokens - system_tokens)
                    selected = self._limit_by_tokens(selected, remaining)
                    break
        
        # Always return original messages unchanged
        full_messages = messages
        to_send_list: List[Message] = list(system_messages) + list(selected)
        
        return StrategyResponse(
            messages=full_messages,
            messages_to_send=to_send_list
        )
    
    def _split_into_turns(self, messages: List[Message]) -> List[List[Message]]:
        """按 HumanMessage 边界将消息拆分为轮次（turns）。"""
        if not messages:
            return []
        turns: List[List[Message]] = []
        current: List[Message] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # 开启新一轮：若当前非空，先结束上一轮
                if current:
                    turns.append(current)
                    current = []
                current.append(msg)
            else:
                # 累积到当前轮
                current.append(msg)
        # 最后一轮收尾
        if current:
            turns.append(current)
        return turns
    
    def _filter_messages(self, messages: List[Message]) -> List[Message]:
        """
        Filter messages by count and token limits.
        
        Args:
            messages: Messages to filter
            
        Returns:
            Filtered messages
        """
        if not messages:
            return []
        
        # Step 1: Limit by message count (keep most recent)
        # Turn-based count limiting is handled in process();
        # no count-based limiting in _filter_messages.
        
        # Step 2: Limit by token count (keep most recent)
        if self.max_input_tokens is not None:
            messages = self._limit_by_tokens(messages, self.max_input_tokens)
        
        return messages
    
    def _limit_by_tokens(
        self,
        messages: List[Message],
        max_tokens: int
    ) -> List[Message]:
        """
        Limit messages by token count, keeping most recent messages.
        
        Ensures at least the most recent message is kept, even if it
        exceeds the token limit (to avoid empty context).
        
        Args:
            messages: Messages to limit
            max_tokens: Maximum token count
            
        Returns:
            Messages within token limit
        """
        if not messages:
            return []
        
        # Always keep at least the most recent message
        result: List[Message] = [messages[-1]]
        current_tokens = self.count_messages_tokens(result)
        
        # Add messages from newest to oldest while within limit
        for msg in reversed(messages[:-1]):
            msg_tokens = 4 + self.count_tokens(msg.content or "")
            if current_tokens + msg_tokens <= max_tokens:
                result.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return result
