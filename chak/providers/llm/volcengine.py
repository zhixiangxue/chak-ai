from typing import Optional, List, Dict, Any, Iterator

import openai
from pydantic import field_validator

from .base import BaseProviderConfig, BaseMessageConverter, Provider
from ...message import Message, MessageChunk, AIMessage


class VolcEngineConfig(BaseProviderConfig):
    """VolcEngine-specific configuration."""
    base_url: Optional[str] = "https://ark.cn-beijing.volces.com/api/v3"
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for VolcEngine."""
        return v or "https://ark.cn-beijing.volces.com/api/v3"


class VolcEngineMessageConverter(BaseMessageConverter):
    """Converter for VolcEngine message formats."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert to VolcEngine message format (OpenAI compatible)."""
        return [
            {
                "role": msg.role or "user",
                "content": msg.content or ""
            }
            for msg in messages
        ]
    
    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert VolcEngine response to standard AIMessage."""
        choice = response.choices[0]
        message = choice.message
        
        return AIMessage(
            content=message.content or "",
            metadata={
                "provider": "volcengine",
                "model": response.model,
                "usage": getattr(response, 'usage', {}),
                "finish_reason": choice.finish_reason,
            }
        )
    
    def from_provider_chunk(self, chunk: Any) -> MessageChunk:
        """Convert VolcEngine streaming chunk to standard MessageChunk."""
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        
        content = delta.content if delta and delta.content else ""
        is_final = bool(choice and choice.finish_reason is not None)
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata={
                "provider": "volcengine", 
                "model": chunk.model,
                "finish_reason": choice.finish_reason if choice else None
            }
        )


class VolcEngineProvider(Provider):
    """VolcEngine provider implementation."""
    
    def _initialize_client(self):
        """Initialize VolcEngine client."""
        self._client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            http_client=self._create_http_client(),
        )
    
    def _send_complete(self, messages: List, **kwargs) -> Any:
        """Send non-streaming request to VolcEngine."""
        return self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **kwargs
        )
    
    def _send_stream(self, messages: List, **kwargs) -> Iterator[Any]:
        """Send streaming request to VolcEngine."""
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield chunk
