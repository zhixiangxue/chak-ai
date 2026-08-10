"""
OpenAI Provider

Official OpenAI API provider.
Official documentation: https://platform.openai.com/docs/api-reference

Supported models:
- GPT-4 series: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini
- GPT-3.5 series: gpt-3.5-turbo
- O1 series: o1, o1-mini, o1-preview
"""
from types import SimpleNamespace
from typing import Optional, Any, Union, Dict, List, get_args

import re

from pydantic import field_validator
from openai.types.responses import (
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseOutputItemAddedEvent,
    ResponseCompletedEvent,
    ResponseStreamEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
)

from .openai_compat import OpenAICompatibleMessageConverter, OpenAICompatibleProvider
from .base import BaseProviderConfig
from ...message import AIMessage, MessageChunk, ReasoningChunk, ToolCallDelta, UnifiedStreamChunk
from ...schemas import Reasoning, Cache


class OpenAIConfig(BaseProviderConfig):
    """Configuration for OpenAI provider."""
    base_url: Optional[str] = "https://api.openai.com/v1"
    cache: Optional[Cache] = None  # Prompt caching settings

    @field_validator('base_url', mode='before')
    @classmethod
    def set_default_base_url(cls, v):
        """Set default base URL for OpenAI."""
        return v or "https://api.openai.com/v1"


class OpenAIMessageConverter(OpenAICompatibleMessageConverter):
    """Converter for OpenAI message formats."""

    def from_provider_response(self, response: Any) -> AIMessage:
        """Handle both Chat Completions and Responses API responses for OpenAI."""
        if hasattr(response, "choices") and response.choices:
            # Chat Completions style - use base implementation
            return super().from_provider_response(response)

        # Responses API style response
        content, reasoning_content, metadata = self._from_responses_response(response)
        return AIMessage(
            content=content,
            reasoning_content=reasoning_content,
            metadata=metadata,
        )
    
    def from_provider_chunk(self, chunk: Any) -> UnifiedStreamChunk:
        """Convert OpenAI streaming chunk to UnifiedStreamChunk.
        
        Handles both:
        - Chat Completions API chunks (delta-based, has 'choices' attribute)
        - Responses API streaming events (event-based, no 'choices' attribute)
        """
        # Distinguish by checking for 'choices' attribute
        # Chat Completions chunks have 'choices', Responses API events don't
        if hasattr(chunk, 'choices'):
            # Chat Completions chunk handling
            return super().from_provider_chunk(chunk)
        else:
            # Responses API event handling
            return self._from_responses_event(chunk)
    
    def _from_responses_event(self, event: Any) -> UnifiedStreamChunk:
        """Handle OpenAI Responses API streaming events.

        Event types and flow:
        1. response.created: Response object created
        2. response.in_progress: Response is being generated
        3. response.output_item.added (reasoning): Reasoning started
        4. response.reasoning_summary_text.delta: Reasoning summary delta
        5. response.output_item.added (message): Answer message started
        6. response.output_text.delta: Text content delta (ANSWER CONTENT)
        7. response.output_item.added (function_call): Tool call started
        8. response.function_call_arguments.delta: Tool args delta (TOOL CONTENT)
        9. response.completed: Response generation completed

        Note: Reasoning raw content is encrypted and not streamed.
        However, reasoning summary (when requested) IS streamed as deltas.
        """
        # Handle reasoning summary text delta events (REASONING CONTENT)
        if isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
            return UnifiedStreamChunk(
                content="",
                reasoning_content=event.delta,
                is_final=False,
            )

        # Handle answer text delta events (ANSWER CONTENT)
        if isinstance(event, ResponseTextDeltaEvent):
            return UnifiedStreamChunk(
                content=event.delta,
                reasoning_content=None,
                is_final=False,
            )

        # Handle function-call argument streaming (TOOL CONTENT)
        if isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            return UnifiedStreamChunk(
                content="",
                reasoning_content=None,
                tool_calls_delta=[ToolCallDelta(
                    index=event.output_index,
                    id=None,
                    type=None,
                    function_name=None,
                    function_arguments=event.delta,
                )],
                is_final=False,
            )

        # Handle output item added events
        if isinstance(event, ResponseOutputItemAddedEvent):
            item_type = getattr(event.item, 'type', None)

            # Reasoning item added - reasoning started
            if item_type == 'reasoning':
                return UnifiedStreamChunk(content="", reasoning_content="", is_final=False)

            # Message item added - answer message started
            if item_type == 'message':
                return UnifiedStreamChunk(content="", reasoning_content=None, is_final=False)

            # Function-call item added - emit tool call id + name
            if item_type == 'function_call':
                return UnifiedStreamChunk(
                    content="",
                    reasoning_content=None,
                    tool_calls_delta=[ToolCallDelta(
                        index=getattr(event, 'output_index', 0),
                        id=getattr(event.item, 'call_id', None),
                        type="function",
                        function_name=getattr(event.item, 'name', None),
                        function_arguments=None,
                    )],
                    is_final=False,
                )

        # Handle completion events
        if isinstance(event, ResponseCompletedEvent):
            metadata = self._build_metadata(event.response, choice=None)
            # Translate Responses status to Chat Completions finish_reason so
            # ToolManager's loop condition (finish_reason == "tool_calls") works.
            # Check whether any output item is a function_call.
            output_items = getattr(event.response, 'output', None) or []
            has_function_call = any(
                getattr(item, 'type', None) == 'function_call'
                if not isinstance(item, dict) else item.get('type') == 'function_call'
                for item in output_items
            )
            finish_reason = "tool_calls" if has_function_call else "stop"
            return UnifiedStreamChunk(
                content="",
                reasoning_content=None,
                is_final=True,
                finish_reason=finish_reason,
                metadata=metadata.model_dump() if metadata else None,
            )

        # For other events (created, in_progress, done, etc.), return empty chunk
        return UnifiedStreamChunk(content="", reasoning_content=None, is_final=False)

    def _from_responses_response(self, response: Any):
        """Handle OpenAI Responses API response.

        基于 2.txt Reasoning models 文档：
        - 最终答案文本来自 `response.output_text` 或 output 中 type="message" 的 output_text
        - 推理摘要来自 output 中 type="reasoning" 的 summary 数组
        """
        # 1) Final answer content
        content: str = ""
        if hasattr(response, "output_text") and response.output_text:
            content = response.output_text
        else:
            output_items = getattr(response, "output", None)
            if output_items:
                for item in output_items:
                    item_type = getattr(item, "type", None)
                    if item_type is None and isinstance(item, dict):
                        item_type = item.get("type")
                    if item_type == "message":
                        contents = getattr(item, "content", None)
                        if contents is None and isinstance(item, dict):
                            contents = item.get("content")
                        if contents:
                            texts: list[str] = []
                            for c in contents:
                                c_type = getattr(c, "type", None)
                                if c_type is None and isinstance(c, dict):
                                    c_type = c.get("type")
                                if c_type == "output_text":
                                    text = getattr(c, "text", None)
                                    if text is None and isinstance(c, dict):
                                        text = c.get("text")
                                    if isinstance(text, str):
                                        texts.append(text)
                            if texts:
                                content = "".join(texts)
                                break

        # 2) Reasoning summary content
        reasoning_content: Optional[str] = None
        output_items = getattr(response, "output", None)
        if output_items:
            for item in output_items:
                item_type = getattr(item, "type", None)
                if item_type is None and isinstance(item, dict):
                    item_type = item.get("type")
                if item_type == "reasoning":
                    summary_list = getattr(item, "summary", None)
                    if summary_list is None and isinstance(item, dict):
                        summary_list = item.get("summary")
                    if summary_list:
                        texts: list[str] = []
                        for s in summary_list:
                            text = getattr(s, "text", None)
                            if text is None and isinstance(s, dict):
                                text = s.get("text")
                            if isinstance(text, str):
                                texts.append(text)
                        if texts:
                            reasoning_content = "".join(texts)
                            break

        metadata = self._build_metadata(response, choice=None)
        return content, reasoning_content, metadata


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider implementation.

    Supports prompt caching via ``OpenAIConfig.cache``. Unlike Anthropic,
    OpenAI caching is automatic for prompts ≥ 1024 tokens. The cache config
    adds two things:
    - ``prompt_cache_key``: improves hit rate across requests sharing a prefix.
    - Explicit breakpoints on GPT-5.6+ models when ``cache.system_prompt`` is set.
    """

    # ------------------------------------------------------------------ #
    # Cache helpers                                                       #
    # ------------------------------------------------------------------ #

    def _apply_cache_params(self, messages: list, kwargs: dict) -> None:
        """Inject OpenAI prompt-caching parameters into kwargs.

        - ``cache.key`` → ``prompt_cache_key`` (routing hint, all models).
        - ``cache.system_prompt`` → wrap system message with explicit
          ``prompt_cache_breakpoint``. Only injected on GPT-5.6+ models;
          older models reject the parameter with HTTP 400
          (``prompt_cache_breakpoint is not supported on this model``)
          and are left untouched so automatic caching keeps working.
        """
        cache: Optional[Cache] = getattr(self.config, "cache", None)
        if cache is None:
            return

        # prompt_cache_key: pass-through routing key (supported on all models
        # gpt-4o and newer)
        if cache.key:
            kwargs["prompt_cache_key"] = cache.key

        # Explicit breakpoint on system prompt (Chat Completions API only,
        # GPT-5.6+ only). Silently skipped on older models — automatic
        # caching still works for them, so nothing is lost.
        if cache.system_prompt and self._is_gpt56_plus(self.config.model):
            self._inject_system_breakpoint(messages)

    @staticmethod
    def _is_gpt56_plus(model: str) -> bool:
        """Whether the model is GPT-5.6 or a later generation.

        Two GPT-5.6+ behaviors require this check:

        1. **Tool calling**: Chat Completions returns HTTP 400 for function
           tools on these models (error: "Function tools with
           reasoning_effort are not supported ... use /v1/responses").
           Empirically verified (2026-08) that gpt-5, gpt-5.1, and gpt-5.5
           do NOT have this restriction — only 5.6+ does. The Responses
           API is the required path for tool calling on 5.6+.

        2. **Prompt caching**: ``prompt_cache_breakpoint`` is only
           accepted on GPT-5.6+; older models reject it with HTTP 400.

        Matches: gpt-5.6, gpt-5.6-luna, gpt-5.6-mini, gpt-5.7, ..., gpt-6, ...
        """
        m = (model or "").lower()
        # GPT-5.6 through 5.9 (any patch variant like gpt-5.6-mini also matches)
        for minor in ("5.6", "5.7", "5.8", "5.9"):
            if m.startswith(f"gpt-{minor}"):
                return True
        # GPT-6 and later major versions
        match = re.match(r"gpt-(\d+)", m)
        if match and int(match.group(1)) >= 6:
            return True
        return False

    @staticmethod
    def _inject_system_breakpoint(messages: list) -> None:
        """Wrap the first system message's string content with a breakpoint.

        Mutates *messages* in place. Only converts plain-string ``content``
        to the structured list form; already-structured content is left as-is
        to avoid clobbering multimodal blocks.
        """
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue  # Already structured or empty — skip
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ]
            break  # Only the first system message

    # ------------------------------------------------------------------ #
    # Reasoning helpers                                                   #
    # ------------------------------------------------------------------ #

    def _apply_reasoning_params(self, kwargs: dict) -> None:
        """Apply reasoning parameters for OpenAI.
        
        OpenAI Responses API natively supports 'reasoning' parameter with format:
            reasoning = {"effort": "low|medium|high", "summary": "auto|detailed|concise"}
        
        This method transforms chak's Reasoning object to OpenAI's format.
        For Chat Completions API fallback, the parameter will be removed in exception handlers.
        """
        reasoning = kwargs.get('reasoning')
        if not reasoning:
            return
        
        # Convert Reasoning object to dict if needed
        if isinstance(reasoning, Reasoning):
            reasoning_dict = reasoning.model_dump(exclude_none=True)
        elif isinstance(reasoning, dict):
            reasoning_dict = reasoning
        else:
            # Unknown type, remove it
            kwargs.pop('reasoning', None)
            return
        
        # Build OpenAI reasoning parameter
        openai_reasoning = {}
        
        # Map effort (direct mapping)
        if 'effort' in reasoning_dict:
            openai_reasoning['effort'] = reasoning_dict['effort']
        
        # Map summary (chak uses "auto"/"enabled"/"disabled", OpenAI uses "auto"/"detailed"/"concise")
        if 'summary' in reasoning_dict:
            summary_value = reasoning_dict['summary']
            if summary_value == 'enabled':
                openai_reasoning['summary'] = 'auto'  # Use auto to get best available
            elif summary_value == 'auto':
                openai_reasoning['summary'] = 'auto'
            # 'disabled' means don't include summary parameter at all
        
        # Replace with OpenAI format
        if openai_reasoning:
            kwargs['reasoning'] = openai_reasoning
        else:
            # No valid reasoning config, remove it
            kwargs.pop('reasoning', None)
    
    # ------------------------------------------------------------------ #
    # Responses API adapter                                               #
    #                                                                    #
    # GPT-5.6+ models require the Responses API for tool calling because #
    # Chat Completions rejects function tools when the model's internal  #
    # reasoning_effort is active. The methods below convert between the  #
    # two API formats at the I/O boundary so the rest of chak (converter,#
    # ToolManager, Conversation) works unchanged.                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_responses_unsupported_error(error: Exception) -> bool:
        """Check if an error means the model can't use Responses API (for fallback)."""
        msg = str(error).lower()
        return any(p in msg for p in (
            "unsupported parameter", "not supported", "does not support",
        ))

    @staticmethod
    def _convert_tools_to_responses(tools: List[Dict]) -> List[Dict]:
        """Convert Chat Completions tool definitions to Responses API format.

        Chat Completions nests fields under a ``function`` key::

            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Responses API expects them flat at the top level::

            {"type": "function", "name": ..., "description": ..., "parameters": ...}
        """
        result: List[Dict] = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
                fn = tool["function"]
                converted: Dict[str, Any] = {"type": "function"}
                for key in ("name", "description", "parameters", "strict"):
                    if key in fn:
                        converted[key] = fn[key]
                result.append(converted)
            else:
                # Already flat (Responses format) or unknown type — pass through
                result.append(tool)
        return result

    @staticmethod
    def _convert_content_parts_to_responses(content: List[Dict]) -> List[Dict]:
        """Convert Chat Completions multimodal content parts to Responses API format.

        Chat Completions uses ``text`` and ``image_url`` part types, while the
        Responses API requires ``input_text`` and ``input_image`` respectively.

        Mapping:
        - ``{"type": "text", "text": "..."}`` → ``{"type": "input_text", "text": "..."}``
        - ``{"type": "image_url", "image_url": {"url": "...", "detail": "..."}}``
          → ``{"type": "input_image", "image_url": "...", "detail": "..."}``
        """
        result: List[Dict] = []
        for part in content:
            if not isinstance(part, dict):
                result.append(part)
                continue
            part_type = part.get("type", "")
            if part_type == "text":
                result.append({"type": "input_text", "text": part.get("text", "")})
            elif part_type == "image_url":
                image_url_info = part.get("image_url", {})
                if isinstance(image_url_info, dict):
                    image_url = image_url_info.get("url", "")
                    detail = image_url_info.get("detail", "auto")
                else:
                    image_url = str(image_url_info)
                    detail = "auto"
                result.append({
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": detail,
                })
            else:
                # Unknown type — pass through as-is
                result.append(part)
        return result

    @staticmethod
    def _convert_messages_to_responses_input(messages: List[Dict]) -> List[Dict]:
        """Convert Chat Completions messages to Responses API input items.

        Four structural differences require translation:

        1. Multimodal content parts need type mapping:
           ``text`` → ``input_text``, ``image_url`` → ``input_image``.
        2. Assistant message with ``tool_calls`` → split into a text item
           plus one ``function_call`` item per tool call.
        3. ``role="tool"`` message → ``function_call_output`` item.
        4. Regular string-content ``system`` / ``user`` / ``assistant``
           messages pass through unchanged.
        """
        input_items: List[Dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                input_items.append(msg)
                continue

            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "tool":
                # Tool result → function_call_output
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id or "",
                    "output": content if isinstance(content, str) else str(content),
                })
            elif role == "assistant" and tool_calls:
                # Assistant with tool calls → text (if any) + function_call items
                if content:
                    input_items.append({"role": "assistant", "content": content})
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}"),
                    })
            else:
                # Regular message — convert multimodal content parts if needed
                if isinstance(content, list):
                    content = OpenAIProvider._convert_content_parts_to_responses(content)
                input_items.append({"role": role, "content": content})

        return input_items

    @staticmethod
    def _wrap_responses_as_completion(response: Any) -> Any:
        """Wrap a Responses API response as a Chat Completions-compatible object.

        This is the key adapter trick: by giving the wrapped object a
        ``choices`` list with ``message.{content, tool_calls,
        reasoning_content}``, the base ``from_provider_response`` in
        ``OpenAICompatibleMessageConverter`` processes it through the
        existing Chat Completions code path — no converter changes needed.
        """
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[Any] = []

        output_items = getattr(response, "output", None) or []

        for item in output_items:
            item_type = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")

            if item_type == "message":
                contents = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
                if contents:
                    for c in contents:
                        c_type = getattr(c, "type", None) if not isinstance(c, dict) else c.get("type")
                        if c_type == "output_text":
                            text = getattr(c, "text", "") if not isinstance(c, dict) else c.get("text", "")
                            content_parts.append(text)

            elif item_type == "reasoning":
                summaries = getattr(item, "summary", None) if not isinstance(item, dict) else item.get("summary")
                if summaries:
                    for s in summaries:
                        text = getattr(s, "text", "") if not isinstance(s, dict) else s.get("text", "")
                        reasoning_parts.append(text)

            elif item_type == "function_call":
                name = getattr(item, "name", "") if not isinstance(item, dict) else item.get("name", "")
                call_id = getattr(item, "call_id", "") if not isinstance(item, dict) else item.get("call_id", "")
                arguments = getattr(item, "arguments", "{}") if not isinstance(item, dict) else item.get("arguments", "{}")
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(name=name, arguments=arguments),
                ))

        message = SimpleNamespace(
            content="".join(content_parts) if content_parts else None,
            reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls if tool_calls else None,
        )
        choice = SimpleNamespace(
            message=message,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

        return SimpleNamespace(
            choices=[choice],
            usage=getattr(response, "usage", None),
            model=getattr(response, "model", None),
            id=getattr(response, "id", None),
        )

    def _send_via_responses(self, model: str, messages: List[Dict], stream: bool, **kwargs):
        """Send via Responses API with automatic format conversion.

        Converts tool definitions and messages from Chat Completions
        format to Responses API format, calls the API, then wraps the
        non-streaming response back to CC format so the rest of chak
        (converter, ToolManager) processes it unchanged.

        For streaming, the raw event stream is returned directly — the
        converter's ``_from_responses_event`` handles each event.
        """
        # Convert tools from CC nested format to Responses flat format
        cc_tools = kwargs.pop('tools', None)
        if cc_tools:
            kwargs['tools'] = self._convert_tools_to_responses(cc_tools)

        # Convert tool_choice from CC nested format to Responses flat format.
        # CC:  {"type": "function", "function": {"name": "extract"}}
        # Responses: {"type": "function", "name": "extract"}
        tool_choice = kwargs.get('tool_choice')
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" \
                and "function" in tool_choice:
            fn = tool_choice["function"]
            kwargs['tool_choice'] = {"type": "function", "name": fn.get("name", "")}

        # Convert messages to Responses input format
        responses_input = self._convert_messages_to_responses_input(messages)

        # Convert reasoning params from chak format to OpenAI format
        self._apply_reasoning_params(kwargs)

        # Responses API also supports prompt_cache_key
        cache = getattr(self.config, "cache", None)
        if cache and cache.key:
            kwargs.setdefault("prompt_cache_key", cache.key)

        if stream:
            # Return raw event stream — converter handles event parsing
            return self._client.responses.create(
                model=model,
                input=responses_input,
                stream=True,
                **kwargs,
            )

        response = self._client.responses.create(
            model=model,
            input=responses_input,
            **kwargs,
        )
        # Wrap as Chat Completions format for downstream processing
        return self._wrap_responses_as_completion(response)

    # ------------------------------------------------------------------ #
    # Send methods (routing layer)                                        #
    # ------------------------------------------------------------------ #

    def _send_complete(self, messages, **kwargs):
        """Send non-streaming request.

        Routing logic:
        - GPT-5.6+ models → always Responses API (Chat Completions
          rejects function tools due to internal reasoning_effort).
        - Other models + reasoning param → try Responses API, fall
          back to Chat Completions if the model doesn't support it.
        - Everything else → Chat Completions API (unchanged).
        """
        model = self.config.model

        # GPT-5.6+: Responses API is the only option for tool calling
        if self._is_gpt56_plus(model):
            return self._send_via_responses(model, messages, stream=False, **kwargs)

        # Other models with reasoning: try Responses API, fall back gracefully
        if 'reasoning' in kwargs:
            try:
                return self._send_via_responses(model, messages, stream=False, **kwargs)
            except Exception as e:
                if self._is_responses_unsupported_error(e):
                    kwargs.pop('reasoning', None)  # Fall back without reasoning
                else:
                    raise

        # Chat Completions API (default path)
        self._apply_cache_params(messages, kwargs)
        self._apply_reasoning_params(kwargs)
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

    def _send_stream(self, messages, **kwargs):
        """Send streaming request.

        Same routing logic as ``_send_complete``.
        """
        model = self.config.model

        # GPT-5.6+: Responses API is the only option for tool calling
        if self._is_gpt56_plus(model):
            return self._send_via_responses(model, messages, stream=True, **kwargs)

        # Other models with reasoning: try Responses API, fall back gracefully
        if 'reasoning' in kwargs:
            try:
                return self._send_via_responses(model, messages, stream=True, **kwargs)
            except Exception as e:
                if self._is_responses_unsupported_error(e):
                    kwargs.pop('reasoning', None)  # Fall back without reasoning
                else:
                    raise

        # Chat Completions API streaming (default path)
        self._apply_cache_params(messages, kwargs)
        self._apply_reasoning_params(kwargs)

        if 'stream_options' not in kwargs:
            kwargs['stream_options'] = {"include_usage": True}

        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )
