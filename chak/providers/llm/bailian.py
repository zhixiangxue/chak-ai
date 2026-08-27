"""
Bailian Provider (Alibaba Cloud)

Uses DashScope SDK for native integration with Alibaba Cloud's Bailian service.
Official documentation: https://help.aliyun.com/zh/model-studio/

Supported models:
- Text-only (Generation API): qwen-plus, qwen-turbo, qwen-max, etc.
- Multimodal (MultiModalConversation API): qwen-vl-max, qwen3-vl-plus, qwen3.6-plus, etc.
- With reasoning: qwen-plus (enable_thinking), QwQ models, etc.
"""
from typing import Optional, Dict, Any, List, Iterator

from pydantic import field_validator
from dashscope import Generation, MultiModalConversation
from dashscope.api_entities.dashscope_response import GenerationResponse

from .base import Provider, BaseProviderConfig, BaseMessageConverter
from ...exceptions import ProviderError, ErrorType
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
    
    @staticmethod
    def _convert_content_to_multimodal(content):
        """Convert OpenAI-compatible multimodal content to DashScope multimodal format.
        
        OpenAI-compatible format:
            [{"type": "image_url", "image_url": {"url": "..."}}, {"type": "text", "text": "..."}]
        
        DashScope MultiModalConversation format:
            [{"image": "..."}, {"text": "..."}]
        
        If content is a plain string, returns it unchanged.
        """
        if not isinstance(content, list):
            return content
        
        result = []
        for part in content:
            part_type = part.get("type", "")
            if part_type == "image_url":
                result.append({"image": part["image_url"]["url"]})
            elif part_type == "text":
                result.append({"text": part["text"]})
            elif part_type == "input_audio":
                audio = part.get("input_audio", {})
                result.append({"audio": audio.get("data", "")})
            elif part_type == "video":
                result.append({"video": part.get("video", {}).get("url", "")})
            else:
                # Unknown type, pass through as-is
                result.append(part)
        return result
    
    @staticmethod
    def _normalize_stream_content(value: Any) -> str:
        """Normalize provider stream content to text for UnifiedStreamChunk."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            text_parts = [
                part["text"]
                for part in value
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            return "".join(text_parts)
        return ""

    @staticmethod
    def _normalize_stream_reasoning_content(value: Any) -> Optional[str]:
        """Normalize provider stream reasoning content to optional text."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            text_parts = [
                part["text"]
                for part in value
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            return "".join(text_parts) or None
        return None
    
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
    
    def from_provider_response(self, response: GenerationResponse, is_multimodal: bool = False) -> AIMessage:
        """Convert DashScope response to AIMessage.
        
        DashScope Generation response structure:
        - response.output.choices[0].message.content: answer content (string)
        - response.output.choices[0].message.reasoning_content: reasoning content
        - response.output.choices[0].message.tool_calls: tool calls (if any)
        
        DashScope MultiModalConversation response structure:
        - response.output.choices[0].message.content: answer content (list of dicts)
          e.g. [{"text": "answer text"}]
        - response.output.choices[0].message.reasoning_content: reasoning content
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
                raw_content = getattr(message, 'content', '') or ""
                if is_multimodal and isinstance(raw_content, list):
                    # MultiModalConversation returns content as list of dicts
                    # e.g. [{"text": "..."}, {"text": "..."}]
                    text_parts = []
                    for part in raw_content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                    content = "".join(text_parts) if text_parts else ""
                else:
                    content = raw_content if isinstance(raw_content, str) else ""
                
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
            reasoning_content = self._normalize_stream_reasoning_content(
                getattr(message, 'reasoning_content', None)
            )
        except Exception:
            reasoning_content = None
        
        try:
            content = self._normalize_stream_content(getattr(message, 'content', ''))
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
            # DashScope's GenerationResponse doesn't include a top-level
            # ``model`` field, so ``response_dict.get('model')`` is almost
            # always None. The provider layer (which has access to the
            # configured model name via ``self.config.model``) fills this
            # in after the fact — see ``BailianProvider.send``.
            model=response_dict.get('model', None),
            usage=usage,
            request_id=response_dict.get('request_id', None),
        )
    
    def _build_chunk_metadata(self, chunk: Any) -> Dict[str, Any]:
        """Build metadata from DashScope streaming chunk."""
        metadata = {
            "provider": "bailian",
            # DashScope chunks also lack a top-level model field. The
            # provider layer fills it in after streaming (see send_stream).
        }
        
        # Add request_id if available
        if hasattr(chunk, 'request_id'):
            metadata["request_id"] = chunk.request_id
        
        return metadata


class BailianProvider(Provider):
    """Bailian provider implementation using DashScope SDK.
    
    Supports two DashScope APIs depending on the model:
    - Generation.call(): text-only models (qwen-plus, qwen-max, qwen-turbo, etc.)
    - MultiModalConversation.call(): multimodal models (qwen-vl-*, qwen3.6-*, etc.)
    """
    
    # Model name patterns that require MultiModalConversation API.
    # "-vl" covers the whole vision series (qwen-vl, qwen2-vl, qwen2.5-vl,
    # qwen3-vl); "qwen3." covers dotted multimodal text models like
    # qwen3.6-plus.
    _MULTIMODAL_MODEL_PATTERNS = ["-vl", "qwen3."]
    
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

    def _normalize_error(self, error: BaseException) -> ProviderError:
        """Precisely map DashScope SDK exceptions to ProviderError.

        DashScope has two error pathways:
        1. Exceptions raised directly (e.g. InputRequired, RequestFailure,
           AuthenticationError, TimeoutException, ServiceUnavailableError)
        2. Error responses wrapped in GenerationResponse with non-200
           status_code — these are handled by _check_response_error()
           which raises ProviderError directly.
        """
        if isinstance(error, ProviderError):
            error.provider = error.provider or self.provider_name
            error.model = error.model or self.config.model
            error.base_url = error.base_url or getattr(self.config, "base_url", None)
            return error

        from dashscope.common.error import (
            AuthenticationError as DashScopeAuthError,
            RequestFailure,
            TimeoutException,
            ServiceUnavailableError,
            InvalidParameter,
            InvalidInput,
            UnsupportedModel,
            InputRequired,
            ModelRequired,
        )

        base_url = getattr(self.config, "base_url", None)

        if isinstance(error, DashScopeAuthError):
            return ProviderError(
                f"BailianProvider auth error: {error}",
                provider=self.provider_name,
                model=self.config.model,
                base_url=base_url,
                status_code=401,
                error_type=ErrorType.AUTH_ERROR,
                raw_error=error,
            )

        if isinstance(error, TimeoutException):
            return ProviderError(
                f"BailianProvider timeout: {error}",
                provider=self.provider_name,
                model=self.config.model,
                base_url=base_url,
                status_code=None,
                error_type=ErrorType.TIMEOUT,
                raw_error=error,
            )

        if isinstance(error, ServiceUnavailableError):
            return ProviderError(
                f"BailianProvider service unavailable: {error}",
                provider=self.provider_name,
                model=self.config.model,
                base_url=base_url,
                status_code=503,
                error_type=ErrorType.SERVER_ERROR,
                raw_error=error,
            )

        if isinstance(error, RequestFailure):
            # RequestFailure carries http_code from the API
            http_code = getattr(error, "http_code", None)
            return ProviderError(
                f"BailianProvider request failure: {error}",
                provider=self.provider_name,
                model=self.config.model,
                base_url=base_url,
                status_code=http_code,
                error_type=ErrorType.from_status_code(http_code),
                raw_error=error,
            )

        if isinstance(error, (InvalidParameter, InvalidInput, UnsupportedModel,
                              InputRequired, ModelRequired)):
            return ProviderError(
                f"BailianProvider bad request: {error}",
                provider=self.provider_name,
                model=self.config.model,
                base_url=base_url,
                status_code=400,
                error_type=ErrorType.BAD_REQUEST,
                raw_error=error,
            )

        # Fallback: unrecognized error
        return ProviderError(
            f"BailianProvider error: {error}",
            provider=self.provider_name,
            model=self.config.model,
            base_url=base_url,
            status_code=None,
            error_type=ErrorType.UNKNOWN,
            raw_error=error,
        )
    
    @staticmethod
    def _is_multimodal_model(model: str) -> bool:
        """Check if the model requires MultiModalConversation API.
        
        Models like qwen-vl-max, qwen3-vl-plus, qwen3.6-plus are multimodal
        and require the MultiModalConversation API instead of Generation API.
        """
        model_lower = model.lower()
        return any(pattern in model_lower for pattern in BailianProvider._MULTIMODAL_MODEL_PATTERNS)
    
    def send(
            self,
            messages: List[Message],
            stream: bool = False,
            **kwargs
    ):
        """Unified send method — branches on model type for correct API."""
        is_multimodal = self._is_multimodal_model(self.config.model)
        
        try:
            # Convert messages to provider format
            provider_messages = self.converter.to_provider_format(messages)
            
            # For multimodal models, convert content to DashScope multimodal format
            if is_multimodal:
                provider_messages = self._convert_to_multimodal_messages(provider_messages)
            
            if stream:
                return self._wrap_stream_errors(
                    self._patch_stream_model(
                        self._send_stream(provider_messages, is_multimodal=is_multimodal, **kwargs)
                    )
                )
            else:
                response = self._send_complete(provider_messages, is_multimodal=is_multimodal, **kwargs)
                result = self.converter.from_provider_response(response, is_multimodal=is_multimodal)
                # DashScope responses don't carry a top-level ``model``
                # field, so ``converter._build_metadata`` left it None.
                # The provider owns the configured model name, so patch
                # it here — downstream tooling (cost accounting,
                # chak.inspector's per-model stats table) relies on
                # metadata.model being populated.
                if result.metadata is not None and not result.metadata.model:
                    result.metadata.model = self.config.model
                self._ensure_provider_trace(result)
                return result
        
        except Exception as e:
            raise self._normalize_error(e) from e

    def _patch_stream_model(self, gen):
        """Wrap a chunk generator to back-fill ``metadata['model']``.

        DashScope streaming chunks don't include the model name, so
        ``_build_chunk_metadata`` produces ``{'provider': 'bailian'}``
        without a model. Yield each chunk with the configured model
        name spliced into its metadata dict so downstream consumers see
        a fully-populated ``provider/model`` identifier.
        """
        model = self.config.model
        for chunk in gen:
            md = getattr(chunk, "metadata", None)
            if isinstance(md, dict) and not md.get("model"):
                md["model"] = model
            yield chunk
    
    def _convert_to_multimodal_messages(self, provider_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert message content from OpenAI format to DashScope multimodal format."""
        converted = []
        for msg in provider_messages:
            converted_msg = dict(msg)
            converted_msg["content"] = self.converter._convert_content_to_multimodal(msg.get("content"))
            converted.append(converted_msg)
        return converted
    
    @staticmethod
    def _check_response_error(response) -> None:
        """Check if a DashScope response contains an error and raise if so.

        DashScope returns HTTP 200 even for errors, with status_code in body.
        We check response.status_code (the API-level status, not HTTP status).

        The raised ProviderError includes both status_code and error_type so
        that is_retryable_provider_error() can make precise failover decisions.
        """
        status_code = getattr(response, 'status_code', None)
        if status_code is not None and status_code != 200:
            code = getattr(response, 'code', 'Unknown')
            message = getattr(response, 'message', 'Unknown error')
            raise ProviderError(
                f"DashScope API error (status={status_code}, code={code}): {message}",
                status_code=status_code,
                error_type=ErrorType.from_status_code(status_code),
            )
    
    def _send_complete(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, **kwargs):
        """Send non-streaming request using DashScope SDK.
        
        Args:
            messages: Already converted provider-format messages
            is_multimodal: If True, use MultiModalConversation API
            **kwargs: Additional parameters including:
                - tools: list of tool definitions (optional)
                - reasoning: dict with reasoning config (optional)
                - temperature, top_p, max_tokens, etc.
        
        Returns:
            DashScope response (GenerationResponse or MultiModalConversationResponse)
        """
        if is_multimodal:
            return self._send_multimodal_complete(messages, **kwargs)
        else:
            return self._send_generation_complete(messages, **kwargs)
    
    def _send_generation_complete(self, messages: List[Dict[str, Any]], **kwargs):
        """Send non-streaming request via Generation API (text-only models)."""
        # Apply reasoning parameters
        self._apply_reasoning_params(kwargs)
        
        # Build DashScope parameters
        params = {
            "api_key": self.config.api_key,
            "model": self.config.model,
            "messages": messages,
            "result_format": "message",
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
        
        # Call DashScope Generation API
        response = Generation.call(**params)
        
        # Check for API errors (DashScope returns HTTP 200 even on errors)
        self._check_response_error(response)
        
        return response
    
    def _send_multimodal_complete(self, messages: List[Dict[str, Any]], **kwargs):
        """Send non-streaming request via MultiModalConversation API."""
        import dashscope
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        
        params = {
            "api_key": self.config.api_key,
            "model": self.config.model,
            "messages": messages,
        }
        
        # Add tools / tool_choice (for structured output via tool calling)
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs and kwargs["tool_choice"]:
            params["tool_choice"] = kwargs["tool_choice"]
        
        # Add optional parameters supported by MultiModalConversation
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        
        # Call DashScope MultiModalConversation API
        response = MultiModalConversation.call(**params)
        
        # Check for API errors
        self._check_response_error(response)
        
        return response
    
    def _send_stream(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, **kwargs) -> Iterator[Any]:
        """Send streaming request using DashScope SDK.
        
        Args:
            messages: Already converted provider-format messages
            is_multimodal: If True, use MultiModalConversation streaming
            **kwargs: Additional parameters including tools
        
        Returns:
            Iterator of DashScope streaming chunks
        """
        if is_multimodal:
            return self._send_multimodal_stream(messages, **kwargs)
        else:
            return self._send_generation_stream(messages, **kwargs)
    
    def _send_generation_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        """Send streaming request via Generation API (text-only models)."""
        # Apply reasoning parameters
        self._apply_reasoning_params(kwargs)
        
        # Build DashScope parameters
        params = {
            "api_key": self.config.api_key,
            "model": self.config.model,
            "messages": messages,
            "result_format": "message",
            "stream": True,
            "incremental_output": True,
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
        
        return responses
    
    def _send_multimodal_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        """Send streaming request via MultiModalConversation API."""
        import dashscope
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        
        params = {
            "api_key": self.config.api_key,
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "incremental_output": True,
        }
        
        # Add tools / tool_choice (for structured output via tool calling)
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs and kwargs["tool_choice"]:
            params["tool_choice"] = kwargs["tool_choice"]
        
        # Add optional parameters
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        
        # Call DashScope MultiModalConversation streaming API
        responses = MultiModalConversation.call(**params)
        
        return responses
    
    async def _asend_stream(self, messages: List[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        """Async streaming — falls back to sync streaming for now."""
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
