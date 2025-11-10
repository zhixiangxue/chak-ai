# chak/context/strategies/base.py
"""Base class for context management strategies."""

from abc import ABC, abstractmethod
from typing import List, Callable, Optional

from pydantic import BaseModel

from ...message import Message


class StrategyRequest(BaseModel):
    """Strategy 的请求"""
    messages: List[Message]
    
    class Config:
        arbitrary_types_allowed = True


class StrategyResponse(BaseModel):
    """Strategy 的响应"""
    messages: List[Message]  # 完整消息列表（含策略插入的标记）
    
    class Config:
        arbitrary_types_allowed = True


class BaseContextStrategy(ABC):
    """
    Abstract base class for context management strategies.
    
    All context strategies must inherit from this class and implement
    the process method.
    """
    
    def __init__(
        self,
        token_counter: Optional[Callable[[str], int]] = None,
        **config
    ):
        """
        Initialize the context strategy.
        
        Args:
            token_counter: Custom token counting function.
                          Input: text string
                          Output: token count
                          If not provided, uses default counter
            **config: Strategy-specific configuration parameters
        """
        self.token_counter = token_counter or self._default_token_counter
        self.config = config
    
    @abstractmethod
    def process(self, request: StrategyRequest) -> StrategyResponse:
        """
        Process the message list according to the strategy.
        
        Different strategies may apply different processing logic:
        - FIFO: Keep recent messages, drop old ones
        - Noop: Return all messages unchanged
        - Summarize: Compress old messages into summaries
        - Semantic: Reorder based on relevance (future)
        
        Args:
            request: Strategy request containing messages
            
        Returns:
            Strategy response with processed messages
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.token_counter(text)
    
    def count_messages_tokens(self, messages: List[Message]) -> int:
        """
        Count total tokens in a message list.
        
        Args:
            messages: List of messages
            
        Returns:
            Total token count (including format overhead)
        """
        total = 0
        for msg in messages:
            total += 4  # Format overhead per message
            total += self.count_tokens(msg.content or "")
        total += 2  # Conversation end marker
        return total
    
    @staticmethod
    def _default_token_counter(text: str) -> int:
        """
        Default token counter using tiktoken cl100k_base.
        
        This provides a reasonable approximation for most models.
        If tiktoken is not installed, falls back to character-based estimation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            # 允许特殊 token，避免编码错误
            return len(encoding.encode(text, disallowed_special=()))
        except ImportError:
            # Fallback: 1 token ≈ 4 characters
            return len(text) // 4
