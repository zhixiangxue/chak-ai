from .base import BaseProviderConfig, BaseMessageConverter, Provider
from pydantic import field_validator
from typing import Optional, List, Dict, Any, Iterator
import openai
from ...message import Message, MessageChunk, AIMessage


class BaiduConfig(BaseProviderConfig):
    """Baidu Qianfan-specific configuration."""
    base_url: Optional[str] = "https://qianfan.baidubce.com/v2"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Baidu Qianfan."""
        return v or "https://qianfan.baidubce.com/v2"


class BaiduMessageConverter(BaseMessageConverter):
    """Converter for Baidu Qianfan message formats."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert to Baidu Qianfan message format (OpenAI compatible)."""
        return [
            {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            for msg in messages
        ]
    
    def from_provider_response(self, response: Any) -> Message:
        """Convert Baidu Qianfan response to standard Message."""
        choice = response.choices[0]
        message = choice.message
        
        return Message(
            role="assistant",
            content=message.content or "",
            metadata={
                "provider": "baidu",
                "model": response.model,
                "usage": getattr(response, 'usage', {}),
                "finish_reason": choice.finish_reason,
            }
        )
    
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert Baidu Qianfan streaming chunk to standard MessageChunk."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata={
                "provider": "baidu", 
                "model": chunk.model,
                "finish_reason": choice.finish_reason if choice else None
            }
        )


class BaiduProvider(Provider):
    """Baidu Qianfan provider implementation."""
    
    def _initialize_client(self):
        """Initialize Baidu Qianfan client."""
        self._client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to Baidu Qianfan."""
        return self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs
        )
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to Baidu Qianfan."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk
