from typing import Optional

from pydantic import field_validator

from .base import BaseProviderConfig


class BailianConfig(BaseProviderConfig):
    """Bailian-specific configuration."""
    base_url: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    organization: Optional[str] = None
    project: Optional[str] = None
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for Bailian (DashScope compatible mode)."""
        return v or "https://dashscope.aliyuncs.com/compatible-mode/v1"


from .base import BaseMessageConverter
from typing import List, Dict, Any
from ...message import Message, MessageChunk, AIMessage


class BailianMessageConverter(BaseMessageConverter):
    """Converter for Bailian message formats."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert to Bailian message format."""
        return [
            {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            for msg in messages
        ]
    
    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert Bailian response to standard AIMessage."""
        choice = response.choices[0]
        message = choice.message
        
        return AIMessage(
            content=message.content or "",
            metadata={
                "provider": "bailian",
                "model": response.model,
                "usage": getattr(response, 'usage', {}),
                "finish_reason": choice.finish_reason,
            }
        )
    
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert Bailian streaming chunk to standard MessageChunk."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata={
                "provider": "bailian", 
                "model": chunk.model,
                "finish_reason": choice.finish_reason if choice else None
            }
        )


import openai
from typing import List, Any, Iterator
from .base import Provider


class BailianProvider(Provider):
    """Bailian provider implementation."""
    
    def _initialize_client(self):
        """Initialize Bailian client."""
        # Build client kwargs
        client_kwargs = {
            "api_key": self.config.api_key,
            "base_url": self.config.base_url,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "http_client": self._create_http_client(),
        }
        
        # Add optional organization if exists
        if isinstance(self.config, BailianConfig) and self.config.organization:
            client_kwargs["organization"] = self.config.organization
            
        self._client = openai.OpenAI(**client_kwargs)
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to Bailian."""
        return self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs
        )
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to Bailian."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk