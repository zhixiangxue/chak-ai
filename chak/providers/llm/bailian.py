"""
Bailian Provider (Alibaba Cloud)

Uses DashScope SDK for native integration with Alibaba Cloud's Bailian service.
Official documentation: https://help.aliyun.com/zh/model-studio/

Supported models:
- Qwen series: qwen-plus, qwen-turbo, qwen-max, etc.
- With reasoning: qwen-plus (enable_thinking), QwQ models, etc.
"""
from typing import Optional, Dict, Any, List, Iterator

from pydantic import field_validator
from dashscope import Generation
from dashscope.api_entities.dashscope_response import GenerationResponse

from .base import Provider, BaseProviderConfig, BaseMessageConverter
from ...message import Message, AIMessage, MessageChunk, ReasoningChunk, UnifiedStreamChunk, ToolCallDelta
from ...metadata import Metadata, Usage
from ...schemas import Reasoning


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
        [{
            "role": "user",
            "content": "text",
            "tool_calls": [...],  # for assistant messages
            "tool_call_id": "...",  # for tool messages
        }, ...]
        """
        provider_messages = []
        for msg in messages:
            provider_msg = {
                "role": msg.role,
                "content": msg.content
            }
            
            # Add tool_calls if present (for assistant messages)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                provider_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            
            # Add tool_call_id if present (for tool messages)
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                provider_msg["tool_call_id"] = msg.tool_call_id
            
            provider_messages.append(provider_msg)
        return provider_messages
    
    def from_provider_response(self, response: GenerationResponse) -> AIMessage:
        """Convert DashScope response to AIMessage.
        
        DashScope response structure:
        - response.output.choices[0].message.content: answer content
        - response.output.choices[0].message.reasoning_content: reasoning content
        - response.output.choices[0].message.tool_calls: tool calls (if any)
        """
        # Extract content
        content = ""
        reasoning_content = None
        tool_calls = None
        
        if hasattr(response, 'output') and response.output:
            output = response.output
            
            # Try choices format (message format)
            if hasattr(output, 'choices') and output.choices:
                message = output.choices[0].message
                
                # Answer content
                content = getattr(message, 'content', '') or ""
                # Reasoning content (safely check existence)
                try:
                    reasoning_content = message.reasoning_content
                except (KeyError, AttributeError):
                    reasoning_content = None
                
                # Tool calls (if present) - use dict-style access for DashScope
                # DashScope message is a dict-like object, not a regular object
                if 'tool_calls' in message and message['tool_calls']:
                    from ...message import ChatCompletionMessageToolCall, Function
                    tool_calls = [
                        ChatCompletionMessageToolCall(
                            id=tc['id'],
                            type=tc['type'],
                            function=Function(
                                name=tc['function']['name'],
                                arguments=tc['function']['arguments']
                            )
                        )
                        for tc in message['tool_calls']
                    ]
            
            # Fallback to text format
            elif hasattr(output, 'text'):
                content = output.text or ""
        
        # Build metadata
        metadata = self._build_metadata(response)
        
        return AIMessage(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            metadata=metadata,
        )
    
    def from_provider_chunk(self, chunk: Any) -> UnifiedStreamChunk:
        """Convert DashScope streaming chunk to UnifiedStreamChunk.
        
        DashScope streaming chunk structure (with incremental_output=True):
        - chunk.output.choices[0].message.content: answer content delta
        - chunk.output.choices[0].message.reasoning_content: reasoning content delta
        - chunk.output.choices[0].message.tool_calls: incremental tool_calls
        - chunk.output.choices[0].finish_reason: finish reason
        
        Returns:
            UnifiedStreamChunk with all information extracted
        """
        # Convert chunk to dict to safely access fields (avoid __getattr__ bug)
        chunk_dict = dict(chunk) if hasattr(chunk, '__iter__') else {}
        
        if 'output' not in chunk_dict or not chunk_dict['output']:
            return UnifiedStreamChunk(content="", is_final=False)
        
        output = chunk_dict['output']
        # If output is also a custom object, convert it too
        if hasattr(output, '__iter__') and not isinstance(output, str):
            output_dict = dict(output)
        else:
            output_dict = {}
        
        # Get choices (streaming uses choices format like non-streaming)
        choices = output_dict.get('choices')
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            return UnifiedStreamChunk(content="", is_final=False)
        
        choice = choices[0]
        if hasattr(choice, '__iter__') and not isinstance(choice, str):
            choice_dict = dict(choice)
        else:
            choice_dict = {}
        
        # Get message
        message = choice_dict.get('message')
        if message is None:
            return UnifiedStreamChunk(content="", is_final=False)
        
        # Extract reasoning_content and content directly via attribute access
        # NOTE: dict(message) does NOT include reasoning_content in keys!
        # Must use getattr to access it.
        reasoning_content = None
        content = ''
        
        try:
            reasoning_content = getattr(message, 'reasoning_content', None)
        except Exception:
            reasoning_content = None
        
        try:
            content = getattr(message, 'content', '')
            if content is None:
                content = ''
        except Exception:
            content = ''
        
        # Extract tool_calls (DashScope uses dict-style access)
        tool_calls_delta = []
        if 'tool_calls' in message and message['tool_calls']:
            for tc in message['tool_calls']:
                # Skip empty tool_calls (DashScope sends empty ones on finish)
                if not tc.get('id') and not tc.get('function', {}).get('name') and not tc.get('function', {}).get('arguments'):
                    continue
                
                tool_calls_delta.append(ToolCallDelta(
                    index=tc.get('index', 0),
                    id=tc.get('id'),
                    type=tc.get('type'),
                    function_name=tc.get('function', {}).get('name'),
                    function_arguments=tc.get('function', {}).get('arguments')
                ))
        
        # Check if final
        finish_reason = choice_dict.get('finish_reason')
        is_final = (finish_reason is not None and finish_reason != "null")
        
        return UnifiedStreamChunk(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls_delta=tool_calls_delta,
            finish_reason=finish_reason,
            is_final=is_final,
            metadata=self._build_chunk_metadata(chunk)
        )
    
    def _build_metadata(self, response: GenerationResponse) -> Metadata:
        """Build metadata from DashScope response."""
        # Convert to dict to safely access fields (avoid __getattr__ bug)
        response_dict = dict(response)
        
        usage: Optional[Usage] = None
        
        # Add usage info if available
        if 'usage' in response_dict and response_dict['usage']:
            raw_usage = response_dict['usage']
            prompt_tokens = int(
                raw_usage.get('input_tokens', 0) if hasattr(raw_usage, 'get') 
                else getattr(raw_usage, 'input_tokens', 0)
            )
            completion_tokens = int(
                raw_usage.get('output_tokens', 0) if hasattr(raw_usage, 'get') 
                else getattr(raw_usage, 'output_tokens', 0)
            )
            total_tokens = int(
                raw_usage.get('total_tokens', 0) if hasattr(raw_usage, 'get') 
                else getattr(raw_usage, 'total_tokens', 0)
            )
            
            usage = Usage(
                prompt_tokens=max(prompt_tokens, 0),
                completion_tokens=max(completion_tokens, 0),
                total_tokens=max(total_tokens, 0),
            )
        
        return Metadata(
            provider="bailian",
            model=response_dict.get('model', None),
            usage=usage,
            request_id=response_dict.get('request_id', None),
        )
    
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
                - tools: list of tool definitions (optional)
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
        
        # Add tools if present
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = kwargs["tools"]
        
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
        
        return response
    
    def _send_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        """Send streaming request using DashScope SDK.
        
        Args:
            messages: Already converted provider-format messages
            **kwargs: Additional parameters including tools
        
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
        
        # Add tools if present
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = kwargs["tools"]
        
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
        
        # Normalize Reasoning object to dict
        if isinstance(reasoning, Reasoning):
            reasoning_dict = reasoning.model_dump(exclude_none=True)
        elif isinstance(reasoning, dict):
            reasoning_dict = reasoning
        else:
            # Unknown type, skip
            return
        
        # Enable thinking mode
        kwargs['enable_thinking'] = True
        
        # Map chak's parameters to DashScope
        # Prefer explicit budget if provided
        budget = reasoning_dict.get('budget')
        if isinstance(budget, int) and budget > 0:
            kwargs['thinking_budget'] = budget
        else:
            # Optional: map effort to a reasonable thinking_budget range
            effort = reasoning_dict.get('effort')
            if effort == 'low':
                kwargs['thinking_budget'] = 256
            elif effort == 'medium':
                kwargs['thinking_budget'] = 512
            elif effort == 'high':
                kwargs['thinking_budget'] = 1024
        
        # Note: DashScope doesn't support OpenAI's 'effort' levels directly, we
        # approximate them via thinking_budget when budget is not explicitly set.
