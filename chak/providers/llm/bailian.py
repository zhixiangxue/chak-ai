"""
Bailian Provider (Alibaba Cloud)

Uses DashScope SDK for native integration with Alibaba Cloud's Bailian service.
Official documentation: https://help.aliyun.com/zh/model-studio/

Supported models:
- Qwen series: qwen-plus, qwen-turbo, qwen-max, etc.
- With reasoning: qwen-plus (enable_thinking), QwQ models, etc.
"""
from typing import Optional, Dict, Any, List, Iterator, Union
import os

from pydantic import field_validator
from dashscope import Generation
from dashscope.api_entities.dashscope_response import GenerationResponse

from .base import Provider, BaseProviderConfig, BaseMessageConverter
from ...message import Message, AIMessage, MessageChunk, ReasoningChunk


class BailianConfig(BaseProviderConfig):
    """Configuration for Bailian provider."""
    # DashScope doesn't need base_url, uses built-in endpoints
    base_url: Optional[str] = None
    
    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """DashScope SDK doesn't use base_url."""
        return None


class BailianMessageConverter(BaseMessageConverter):
    """Converter for Bailian (DashScope) message formats."""
    
    def to_provider_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert chak messages to DashScope format.
        
        DashScope uses OpenAI-compatible message format:
        [{"role": "user", "content": "text"}, ...]
        """
        provider_messages = []
        for msg in messages:
            provider_msg = {
                "role": msg.role,
                "content": msg.content
            }
            provider_messages.append(provider_msg)
        return provider_messages
    
    def from_provider_response(self, response: GenerationResponse) -> AIMessage:
        """Convert DashScope response to AIMessage.
        
        DashScope response structure:
        - response.output.choices[0].message.content: answer content
        - response.output.choices[0].message.reasoning_content: reasoning content
        """
        # Extract content
        content = ""
        reasoning_content = None
        
        if hasattr(response, 'output') and response.output:
            output = response.output
            
            # Try choices format (message format)
            if hasattr(output, 'choices') and output.choices:
                message = output.choices[0].message
                # Answer content
                content = getattr(message, 'content', '') or ""
                # Reasoning content
                reasoning_content = getattr(message, 'reasoning_content', None)
            
            # Fallback to text format
            elif hasattr(output, 'text'):
                content = output.text or ""
        
        # Build metadata
        metadata = self._build_metadata(response)
        
        return AIMessage(
            content=content,
            reasoning_content=reasoning_content,
            metadata=metadata,
        )
    
    def from_provider_chunk(self, chunk: Any) -> Union[MessageChunk, ReasoningChunk]:
        """Convert DashScope streaming chunk to MessageChunk or ReasoningChunk.
        
        DashScope streaming chunk structure (with incremental_output=True):
        - chunk.output.choices[0].message.content: answer content delta
        - chunk.output.choices[0].message.reasoning_content: reasoning content delta
        """
        # Convert chunk to dict to safely access fields (avoid __getattr__ bug)
        chunk_dict = dict(chunk) if hasattr(chunk, '__iter__') else {}
        
        if 'output' not in chunk_dict or not chunk_dict['output']:
            return MessageChunk(content="", is_final=False)
        
        output = chunk_dict['output']
        # If output is also a custom object, convert it too
        if hasattr(output, '__iter__') and not isinstance(output, str):
            output_dict = dict(output)
        else:
            output_dict = {}
        
        # Get choices (streaming uses choices format like non-streaming)
        choices = output_dict.get('choices')
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            return MessageChunk(content="", is_final=False)
        
        choice = choices[0]
        if hasattr(choice, '__iter__') and not isinstance(choice, str):
            choice_dict = dict(choice)
        else:
            choice_dict = {}
        
        # Get message
        message = choice_dict.get('message')
        if hasattr(message, '__iter__') and not isinstance(message, str):
            message_dict = dict(message)
        else:
            message_dict = {}
        
        # Check for reasoning content first
        reasoning_content = message_dict.get('reasoning_content')
        if reasoning_content:
            return ReasoningChunk(
                content=reasoning_content,
                is_final=False,
                metadata=self._build_chunk_metadata(chunk)
            )
        
        # Check for answer content
        content = message_dict.get('content', '')
        
        # Check if final
        finish_reason = choice_dict.get('finish_reason')
        is_final = (finish_reason is not None and finish_reason != "null")
        
        return MessageChunk(
            content=content,
            is_final=is_final,
            metadata=self._build_chunk_metadata(chunk)
        )
    
    def _build_metadata(self, response: GenerationResponse) -> Dict[str, Any]:
        """Build metadata from DashScope response."""
        # Convert to dict to safely access fields (avoid __getattr__ bug)
        response_dict = dict(response)
        
        metadata = {
            "provider": "bailian",
            "model": response_dict.get('model', None),
        }
        
        # Add usage info if available
        if 'usage' in response_dict and response_dict['usage']:
            usage = response_dict['usage']
            metadata["usage"] = {
                "prompt_tokens": usage.get('input_tokens', 0) if hasattr(usage, 'get') else getattr(usage, 'input_tokens', 0),
                "completion_tokens": usage.get('output_tokens', 0) if hasattr(usage, 'get') else getattr(usage, 'output_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0) if hasattr(usage, 'get') else getattr(usage, 'total_tokens', 0),
            }
        
        # Add request_id if available
        if 'request_id' in response_dict:
            metadata["request_id"] = response_dict['request_id']
        
        return metadata
    
    def _build_chunk_metadata(self, chunk: Any) -> Dict[str, Any]:
        """Build metadata from DashScope streaming chunk."""
        metadata = {
            "provider": "bailian",
        }
        
        # Add request_id if available
        if hasattr(chunk, 'request_id'):
            metadata["request_id"] = chunk.request_id
        
        return metadata


class BailianProvider(Provider):
    """Bailian provider implementation using DashScope SDK."""
    
    def __init__(self, config: BailianConfig, converter: BailianMessageConverter = None):
        self.config: BailianConfig = config
        self.converter = converter or BailianMessageConverter()
        super().__init__(config, self.converter)
    
    def _initialize_client(self):
        """DashScope doesn't require client initialization.
        
        DashScope uses static methods (Generation.call) instead of client instances.
        API key is set via environment variable in __init__.
        """
        pass
    
    def _send_complete(self, messages: List[Dict[str, Any]], **kwargs) -> GenerationResponse:
        """Send non-streaming request using DashScope SDK.
        
        Args:
            messages: Already converted provider-format messages
            **kwargs: Additional parameters including:
                - reasoning: dict with reasoning config (optional)
                - temperature, top_p, max_tokens, etc.
        
        Returns:
            DashScope GenerationResponse
        """
        # Apply reasoning parameters
        self._apply_reasoning_params(kwargs)
        
        # Build DashScope parameters
        params = {
            "api_key": self.config.api_key,  # Pass API key directly
            "model": self.config.model,
            "messages": messages,  # Already converted
            "result_format": "message",  # Use message format for easier parsing
        }
        
        # Add optional parameters
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        if "enable_thinking" in kwargs:
            params["enable_thinking"] = kwargs["enable_thinking"]
        if "thinking_budget" in kwargs:
            params["thinking_budget"] = kwargs["thinking_budget"]
        
        # Call DashScope API
        response = Generation.call(**params)
        
        # DEBUG: print raw response (convert to dict to avoid DashScope __getattr__ bug with rich)
        from rich import print as rprint
        rprint(dict(response))
        
        return response
    
    def _send_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        """Send streaming request using DashScope SDK.
        
        Args:
            messages: Already converted provider-format messages
            **kwargs: Additional parameters
        
        Returns:
            Iterator of DashScope streaming chunks
        """
        # Apply reasoning parameters
        self._apply_reasoning_params(kwargs)
        
        # Build DashScope parameters
        params = {
            "api_key": self.config.api_key,  # Pass API key directly
            "model": self.config.model,
            "messages": messages,  # Already converted
            "result_format": "message",
            "stream": True,
            "incremental_output": True,  # Required for streaming
        }
        
        # Add optional parameters
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        if "enable_thinking" in kwargs:
            params["enable_thinking"] = kwargs["enable_thinking"]
        if "thinking_budget" in kwargs:
            params["thinking_budget"] = kwargs["thinking_budget"]
        
        # Call DashScope streaming API
        responses = Generation.call(**params)
        
        # Return the iterator directly
        return responses
    
    async def _asend_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        """Async streaming is not yet implemented for DashScope.
        
        Falls back to sync streaming for now.
        """
        # TODO: Implement async streaming when DashScope supports it
        return self._send_stream(messages, **kwargs)
    
    def _apply_reasoning_params(self, kwargs: dict) -> None:
        """Transform chak's unified reasoning params to DashScope native format.
        
        DashScope native parameters:
        - enable_thinking: bool (to enable reasoning mode)
        - thinking_budget: int (optional, token budget for thinking)
        """
        reasoning = kwargs.pop('reasoning', None)
        if not reasoning:
            return
        
        # Enable thinking mode
        kwargs['enable_thinking'] = True
        
        # Map chak's parameters to DashScope
        # DashScope doesn't have 'effort' parameter, only thinking_budget
        if 'budget' in reasoning:
            kwargs['thinking_budget'] = reasoning['budget']
        
        # Note: DashScope doesn't support OpenAI's 'effort' levels (low/medium/high)
        # We could potentially map them to thinking_budget values if needed
