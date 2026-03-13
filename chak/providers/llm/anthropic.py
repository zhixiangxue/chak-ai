"""
Anthropic Claude Provider - Native SDK

Uses the official Anthropic SDK for full feature support including:
- Prompt caching (cache_control: {"type": "ephemeral"})
- Extended thinking (thinking parameter)
- Native tool use (input_schema instead of OpenAI parameters)
- Streaming with proper event handling

Official documentation: https://docs.anthropic.com/

Supported models:
- Claude 4: claude-opus-4-5, claude-sonnet-4-5
- Claude 3.7: claude-3-7-sonnet-20250219
- Claude 3.5: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- Claude 3: claude-3-opus-20240229, claude-3-haiku-20240307
"""
import json
from typing import Optional, Dict, Any, List, Iterator

import anthropic
from pydantic import field_validator

from .base import Provider, BaseProviderConfig, BaseMessageConverter
from ...exceptions import ProviderError
from ...message import (
    Message, AIMessage, UnifiedStreamChunk,
    ToolCallDelta, ChatCompletionMessageToolCall, Function,
)
from ...metadata import Metadata, Usage
from ...schemas import Reasoning, Cache


def _cache_control(cache: Cache) -> Dict[str, Any]:
    """Translate a Cache into an Anthropic cache_control dict.

    Anthropic supports two TTL durations: 5 minutes (default) and 1 hour.
    ttl >= 3600 seconds maps to '1h'; anything else uses the 5-minute default.
    """
    ctrl: Dict[str, Any] = {"type": "ephemeral"}
    if cache.ttl >= 3600:
        ctrl["ttl"] = "1h"
    return ctrl


class AnthropicConfig(BaseProviderConfig):
    """Configuration for Anthropic native SDK provider."""
    base_url: Optional[str] = None  # SDK handles endpoint internally
    cache: Optional[Cache] = None  # Prompt caching settings

    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        return v or None


class AnthropicMessageConverter(BaseMessageConverter):
    """Converter for Anthropic native Messages API format.

    Stateful: tracks block-index → tool-call-index mapping across streaming
    events so that ToolCallDelta indices are always 0-based within tool calls.
    """

    def __init__(self):
        # Stream state: maps content-block index to tool-call list index
        self._block_to_tool_index: Dict[int, int] = {}
        self._next_tool_index: int = 0

    def _reset_stream_state(self) -> None:
        """Reset per-message stream tracking state."""
        self._block_to_tool_index = {}
        self._next_tool_index = 0

    # ------------------------------------------------------------------ #
    # to_provider_format                                                   #
    # ------------------------------------------------------------------ #

    def to_provider_format(self, messages: List[Message]) -> Dict[str, Any]:
        """Convert chak messages to Anthropic native format.

        Returns:
            {"system": str | None, "messages": List[Dict]}
        """
        system: Optional[str] = None
        provider_messages: List[Dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "system":
                # Anthropic takes system as a separate top-level param
                system = msg.content if isinstance(msg.content, str) else ""
                i += 1

            elif msg.role == "tool":
                # Group consecutive ToolMessage objects into a single user
                # message that holds tool_result content blocks.
                tool_results: List[Dict[str, Any]] = []
                while i < len(messages) and messages[i].role == "tool":
                    tm = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tm.tool_call_id,
                        "content": tm.content or "",
                    })
                    i += 1
                provider_messages.append({
                    "role": "user",
                    "content": tool_results,
                })

            elif msg.role == "assistant":
                content_blocks: List[Dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            input_dict = json.loads(tc.function.arguments)
                        except (json.JSONDecodeError, TypeError):
                            input_dict = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": input_dict,
                        })
                if not content_blocks:
                    content_blocks = [{"type": "text", "text": ""}]
                provider_messages.append({"role": "assistant", "content": content_blocks})
                i += 1

            elif msg.role == "user":
                content = msg.content
                if isinstance(content, str):
                    blocks: List[Dict[str, Any]] = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    blocks = self._convert_multimodal(content)
                else:
                    blocks = [{"type": "text", "text": ""}]
                provider_messages.append({"role": "user", "content": blocks})
                i += 1

            else:
                i += 1

        return {"system": system, "messages": provider_messages}

    def _convert_multimodal(self, content_parts: List[Dict]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style multimodal content parts to Anthropic blocks."""
        blocks: List[Dict[str, Any]] = []
        for part in content_parts:
            part_type = part.get("type", "")
            if part_type == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    header, data = url.split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    })
                else:
                    blocks.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
        return blocks or [{"type": "text", "text": ""}]

    # ------------------------------------------------------------------ #
    # from_provider_response (non-streaming)                               #
    # ------------------------------------------------------------------ #

    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert Anthropic Messages API response to AIMessage."""
        content = ""
        reasoning_content: Optional[str] = None
        tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content += block.text
            elif block_type == "thinking":
                # Extended thinking block
                reasoning_content = block.thinking
            elif block_type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(ChatCompletionMessageToolCall(
                    id=block.id,
                    type="function",
                    function=Function(
                        name=block.name,
                        arguments=json.dumps(block.input),
                    ),
                ))

        return AIMessage(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            metadata=self._build_metadata(response),
        )

    # ------------------------------------------------------------------ #
    # from_provider_chunk (streaming)                                      #
    # ------------------------------------------------------------------ #

    def from_provider_chunk(self, chunk: Any) -> UnifiedStreamChunk:
        """Convert an Anthropic streaming event to UnifiedStreamChunk.

        Anthropic SSE event types handled:
        - message_start            → reset stream state, capture input usage
        - content_block_start      → detect new text / tool_use block
        - content_block_delta      → text_delta / input_json_delta / thinking_delta
        - message_delta            → stop_reason → finish_reason mapping
        - message_stop             → is_final
        - content_block_stop       → ignored
        """
        content = ""
        reasoning_content: Optional[str] = None
        tool_calls_delta: List[ToolCallDelta] = []
        is_final = False
        finish_reason: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

        chunk_type = getattr(chunk, "type", None)

        if chunk_type == "message_start":
            self._reset_stream_state()
            # Capture initial input-token count
            usage = getattr(chunk, "message", None)
            if usage:
                usage = getattr(usage, "usage", None)
            if usage:
                metadata = self._usage_to_metadata(usage)

        elif chunk_type == "content_block_start":
            block = chunk.content_block
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                # Assign a 0-based tool-call index (independent of block index)
                tool_idx = self._next_tool_index
                self._block_to_tool_index[chunk.index] = tool_idx
                self._next_tool_index += 1
                tool_calls_delta.append(ToolCallDelta(
                    index=tool_idx,
                    id=block.id,
                    type="function",
                    function_name=block.name,
                    function_arguments="",
                ))

        elif chunk_type == "content_block_delta":
            delta = chunk.delta
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                content = delta.text or ""
            elif delta_type == "thinking_delta":
                reasoning_content = delta.thinking or ""
            elif delta_type == "input_json_delta":
                tool_idx = self._block_to_tool_index.get(chunk.index, 0)
                tool_calls_delta.append(ToolCallDelta(
                    index=tool_idx,
                    function_arguments=delta.partial_json or "",
                ))

        elif chunk_type == "message_delta":
            delta = chunk.delta
            stop_reason = getattr(delta, "stop_reason", None)
            if stop_reason == "end_turn":
                finish_reason = "stop"
                is_final = True
            elif stop_reason == "tool_use":
                # Map to OpenAI-compatible finish reason for ToolManager
                finish_reason = "tool_calls"
                is_final = True
            elif stop_reason is not None:
                finish_reason = stop_reason
                is_final = True
            # message_delta carries output usage
            usage = getattr(chunk, "usage", None)
            if usage:
                metadata = self._usage_to_metadata(usage)

        elif chunk_type == "message_stop":
            is_final = True

        return UnifiedStreamChunk(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls_delta=tool_calls_delta,
            finish_reason=finish_reason,
            is_final=is_final,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Metadata helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_metadata(self, response: Any) -> Metadata:
        """Build Metadata from a complete Anthropic Messages response."""
        usage = None
        raw = getattr(response, "usage", None)
        if raw:
            inp = int(getattr(raw, "input_tokens", 0) or 0)
            out = int(getattr(raw, "output_tokens", 0) or 0)
            cache_create = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
            cache_read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
            usage = Usage(
                prompt_tokens=inp,
                completion_tokens=out,
                total_tokens=inp + out,
                cache_creation_input_tokens=cache_create,
                cache_read_input_tokens=cache_read,
            )
        return Metadata(
            provider="anthropic",
            model=getattr(response, "model", None),
            usage=usage,
            finish_reason=getattr(response, "stop_reason", None),
        )

    def _usage_to_metadata(self, usage: Any) -> Dict[str, Any]:
        """Build metadata dict from an Anthropic usage object."""
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        cache_create = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        return {
            "provider": "anthropic",
            "usage": {
                "prompt_tokens": inp,
                "completion_tokens": out,
                "total_tokens": inp + out,
                "cache_creation_input_tokens": cache_create,
                "cache_read_input_tokens": cache_read,
            },
        }


class AnthropicProvider(Provider):
    """Anthropic Claude provider using the official Anthropic SDK.

    Supports:
    - Non-streaming / streaming chat
    - Tool use (converted from OpenAI format at call time)
    - Extended thinking (via 'reasoning' param)
    - Prompt caching (pass cache_control in message content blocks)
    """

    DEFAULT_MAX_TOKENS = 8192

    def __init__(self, config: AnthropicConfig, converter: AnthropicMessageConverter = None):
        self.config = config
        self.converter = converter or AnthropicMessageConverter()
        self._client: Optional[anthropic.Anthropic] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Anthropic SDK client."""
        kwargs: Dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = anthropic.Anthropic(**kwargs)

    # ------------------------------------------------------------------ #
    # Tool format conversion                                               #
    # ------------------------------------------------------------------ #

    def _convert_tools(self, openai_tools: List[Dict]) -> List[Dict[str, Any]]:
        """Convert OpenAI function-calling format to Anthropic tool format.

        OpenAI: {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
        """
        result = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                result.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {
                        "type": "object",
                        "properties": {},
                    }),
                })
        return result

    # ------------------------------------------------------------------ #
    # Shared param builder                                                 #
    # ------------------------------------------------------------------ #

    def _build_params(self, provider_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Build Anthropic API request params from provider_data and kwargs."""
        system = provider_data.get("system")
        messages = provider_data.get("messages", [])

        # Pop params that need special handling
        tools = kwargs.pop("tools", None)
        kwargs.pop("timeout", None)      # handled at client level
        kwargs.pop("stream_options", None)  # OpenAI-only, not needed here

        # Apply reasoning / extended thinking
        self._apply_reasoning_params(kwargs)

        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", self.DEFAULT_MAX_TOKENS),
        }

        # System prompt: optionally wrap in content block for cache_control
        if system:
            if self.config.cache and self.config.cache.system_prompt:
                params["system"] = [
                    {"type": "text", "text": system,
                     "cache_control": _cache_control(self.config.cache)}
                ]
            else:
                params["system"] = system

        # Tools: convert format, optionally inject cache_control on last entry
        if tools:
            converted = self._convert_tools(tools)
            if self.config.cache and self.config.cache.tools and converted:
                converted[-1]["cache_control"] = _cache_control(self.config.cache)
            params["tools"] = converted

        # Pass through recognised Anthropic kwargs
        _allowed = {"temperature", "top_p", "top_k", "stop_sequences", "thinking"}
        for key in list(kwargs.keys()):
            if key in _allowed:
                params[key] = kwargs.pop(key)

        return params

    # ------------------------------------------------------------------ #
    # send / _send_complete / _send_stream                                 #
    # ------------------------------------------------------------------ #

    def send(self, messages: List[Message], stream: bool = False, **kwargs):
        """Unified send method (overrides base to handle dict provider_data)."""
        try:
            provider_data = self.converter.to_provider_format(messages)
            if stream:
                return self._send_stream(provider_data, **kwargs)
            else:
                response = self._send_complete(provider_data, **kwargs)
                return self.converter.from_provider_response(response)
        except anthropic.APIError as e:
            raise ProviderError(f"AnthropicProvider error: {e}") from e
        except Exception as e:
            raise ProviderError(f"AnthropicProvider error: {e}") from e

    def _send_complete(self, provider_data: Dict[str, Any], **kwargs) -> Any:
        """Send a non-streaming request to the Anthropic Messages API."""
        params = self._build_params(provider_data, **kwargs)
        return self._client.messages.create(**params)

    def _send_stream(self, provider_data: Dict[str, Any], **kwargs) -> Iterator[Any]:
        """Send a streaming request; returns a raw SSE event iterator."""
        params = self._build_params(provider_data, **kwargs)
        # stream=True → returns Stream[RawMessageStreamEvent]
        return self._client.messages.create(**params, stream=True)

    # ------------------------------------------------------------------ #
    # Reasoning / extended thinking                                        #
    # ------------------------------------------------------------------ #

    def _apply_reasoning_params(self, kwargs: dict) -> None:
        """Transform chak's unified reasoning param to Anthropic thinking format.

        Anthropic extended thinking:
            thinking = {"type": "enabled", "budget_tokens": int}
        """
        reasoning = kwargs.pop("reasoning", None)
        if not reasoning:
            return

        if isinstance(reasoning, Reasoning):
            reasoning_dict = reasoning.model_dump(exclude_none=True)
        elif isinstance(reasoning, dict):
            reasoning_dict = reasoning
        else:
            return

        budget: Optional[int] = reasoning_dict.get("budget")
        if not budget:
            effort = reasoning_dict.get("effort", "medium")
            effort_map = {"low": 2000, "medium": 8000, "high": 16000}
            budget = effort_map.get(effort, 8000)

        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": int(budget),
        }
