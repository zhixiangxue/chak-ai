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
            text_content = self.extract_text_content(msg)
            total += self.count_tokens(text_content)
        total += 2  # Conversation end marker
        return total
    
    def extract_text_content(self, message: Message) -> str:
        """
        Extract pure text from message content (for token counting).
        
        Handles both simple text and multimodal content:
        - str: return as-is
        - list: extract only text parts, ignore images/audio
        
        Args:
            message: Message to extract text from
            
        Returns:
            Pure text string (empty if no text content)
        """
        if message.content is None:
            return ""
        
        # Simple text content
        if isinstance(message.content, str):
            return message.content
        
        # Multimodal content: extract only text parts
        if isinstance(message.content, list):
            texts = []
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return " ".join(texts)
        
        return ""
    
    def format_content_for_summary(self, message: Message) -> str:
        """
        Format message content for summary prompt (preserve multimodal context).
        
        Converts multimodal content to human-readable text with placeholders:
        - text: keep as-is
        - image: replace with [Image]
        - audio: replace with [Audio]
        
        This preserves context for summarization while avoiding passing
        raw multimodal data to text-only summarization models.
        
        Args:
            message: Message to format
            
        Returns:
            Formatted text string with placeholders for non-text content
        """
        if message.content is None:
            return ""
        
        # Simple text content
        if isinstance(message.content, str):
            return message.content
        
        # Multimodal content: convert to human-readable format
        if isinstance(message.content, list):
            parts = []
            for item in message.content:
                if not isinstance(item, dict):
                    continue
                    
                item_type = item.get("type", "")
                
                if item_type == "text":
                    parts.append(item.get("text", ""))
                elif item_type == "image_url":
                    parts.append("[Image]")
                elif item_type == "input_audio":
                    parts.append("[Audio]")
                # Add more types as needed
            
            return " ".join(parts)
        
        return ""
    
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
