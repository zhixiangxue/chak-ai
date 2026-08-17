"""
Google AI Provider - Native google-genai SDK

Uses the official google-genai SDK (https://github.com/googleapis/python-genai)
instead of the OpenAI-compatible endpoint. The compat layer mangles Gemini
specifics (thought_signature nesting, null delta indices); the native SDK
treats them as first-class citizens.

Supported models:
- Gemini 3.x: gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash
- Latest aliases: gemini-pro-latest, gemini-flash-latest
"""
import base64
import json
import mimetypes
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .base import BaseMessageConverter, BaseProviderConfig, Provider
from ...exceptions import ErrorType, ProviderError
from ...message import (
    AIMessage,
    ChatCompletionMessageToolCall,
    Function,
    Message,
    ToolCallDelta,
    UnifiedStreamChunk,
)
from ...metadata import Metadata, Usage
from ...schemas import Reasoning


class GoogleConfig(BaseProviderConfig):
    """Configuration for Google AI provider.

    base_url stays None by default (SDK targets AI Studio); it exists as an
    escape hatch for custom/proxy endpoints.
    """
    pass


class GoogleMessageConverter(BaseMessageConverter):
    """Converter between chak messages and the native Gemini Content model.

    Stateful in two ways (same pattern as AnthropicCompatibleMessageConverter):
    - ``_thought_signatures``: Gemini thinking models require every echoed
      functionCall part to carry the thought_signature the model returned
      with it. chak's tool-call objects have no slot for it, so the
      converter caches signature by call id on receive and re-injects it
      when the same call is sent back as history. The cache covers one
      provider instance, which spans the whole conversation.
    - ``_next_tool_index``: streaming tool-call counter, reset per stream,
      so ToolCallDelta indices stay 0-based for the manager's accumulator.
    """

    def __init__(self):
        self._thought_signatures: Dict[str, str] = {}
        self._call_names: Dict[str, str] = {}
        self._next_tool_index: int = 0
        # Gemini emits the function_call part and the finish_reason in
        # different chunks; remember that this stream carried a call so
        # the final chunk can still be normalized to "tool_calls".
        self._saw_function_call: bool = False

    def _reset_stream_state(self) -> None:
        """Reset per-stream tracking state."""
        self._next_tool_index = 0
        self._saw_function_call = False

    # ------------------------------------------------------------------ #
    # to_provider_format: chak messages -> Gemini contents                 #
    # ------------------------------------------------------------------ #

    def to_provider_format(self, messages: List[Message]) -> Dict[str, Any]:
        """Convert chak messages to {"system_instruction", "contents"}.

        Gemini requires strictly alternating user/model turns, so
        consecutive same-role chak messages are merged into one Content.
        """
        system_instruction: Optional[str] = None
        contents: List[genai_types.Content] = []

        for msg in messages:
            if msg.role == "system":
                # Gemini takes the system prompt as a separate param; keep
                # the last one if several are present.
                system_instruction = msg.content if isinstance(msg.content, str) else ""
                continue

            if msg.role == "assistant":
                role, parts = "model", self._assistant_parts(msg)
            elif msg.role == "tool":
                # function_response parts travel in a user turn
                role, parts = "user", self._tool_parts(msg)
            else:
                role, parts = "user", self._user_parts(msg)

            if not parts:
                # Gemini rejects Contents with empty part lists; a
                # placeholder keeps degenerate turns (empty assistant
                # reply, unreadable attachment) from breaking the request.
                parts = [genai_types.Part.from_text(text="(no content)")]

            # Merge into the previous Content when the role matches, to
            # satisfy the alternating-turn constraint.
            if contents and contents[-1].role == role:
                contents[-1].parts.extend(parts)
            else:
                contents.append(genai_types.Content(role=role, parts=parts))

        return {"system_instruction": system_instruction, "contents": contents}

    def _user_parts(self, msg: Message) -> List[genai_types.Part]:
        """Build parts for a user message (text or multimodal list)."""
        content = msg.content
        if content is None:
            return []
        if isinstance(content, str):
            return [genai_types.Part.from_text(text=content)]

        parts: List[genai_types.Part] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text") or ""
                if text:
                    parts.append(genai_types.Part.from_text(text=text))
            elif item_type == "image_url":
                part = self._media_part_from_url(
                    (item.get("image_url") or {}).get("url"), default_mime="image/png"
                )
                if part:
                    parts.append(part)
            elif item_type == "video":
                part = self._media_part_from_url(
                    (item.get("video") or {}).get("url"), default_mime="video/mp4"
                )
                if part:
                    parts.append(part)
            elif item_type == "input_audio":
                audio = item.get("input_audio") or {}
                data_uri = audio.get("data") or ""
                fmt = audio.get("format") or "mp3"
                part = self._media_part_from_data_uri(
                    data_uri, default_mime=f"audio/{fmt}"
                )
                if part:
                    parts.append(part)
        return parts

    def _assistant_parts(self, msg: Message) -> List[genai_types.Part]:
        """Build parts for an assistant message: text plus function calls."""
        parts: List[genai_types.Part] = []
        if isinstance(msg.content, str) and msg.content.strip():
            parts.append(genai_types.Part.from_text(text=msg.content))

        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            self._call_names[tc.id] = tc.function.name
            # Absorb signatures carried on the message itself (e.g. history
            # restored from persistence) so they survive a cache-less round.
            carried = (getattr(tc, "extra_content", None) or {}).get("google", {})
            if carried.get("thought_signature"):
                self._thought_signatures[tc.id] = carried["thought_signature"]
            signature = self._thought_signatures.get(tc.id)
            parts.append(genai_types.Part(
                function_call=genai_types.FunctionCall(
                    id=tc.id, name=tc.function.name, args=args
                ),
                # Missing signature (e.g. restored history) is sent as None;
                # Gemini only enforces it for signatures it issued itself.
                thought_signature=signature,
            ))
        return parts

    def _tool_parts(self, msg: Message) -> List[genai_types.Part]:
        """Build function_response parts for a tool result message.

        Gemini matches responses to calls by function NAME, but chak's
        ToolMessage only carries the call id, so the name is resolved
        through the cache populated from assistant turns.
        """
        call_id = getattr(msg, "tool_call_id", None) or ""
        name = self._call_names.get(call_id, call_id or "unknown_tool")
        content = msg.content
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        return [genai_types.Part.from_function_response(
            name=name, response={"result": content or ""}
        )]

    # ------------------------------------------------------------------ #
    # Media helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _media_part_from_data_uri(url: Optional[str], default_mime: str) -> Optional[genai_types.Part]:
        """Decode a data URI (or accept raw base64) into an inline part."""
        if not url:
            return None
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            mime = header[5:].partition(";")[0] or default_mime
        else:
            # Raw base64 without a data URI prefix
            mime, payload = default_mime, url
        try:
            data = base64.b64decode(payload)
        except Exception:
            return None
        return genai_types.Part.from_bytes(data=data, mime_type=mime)

    @classmethod
    def _media_part_from_url(cls, url: Optional[str], default_mime: str) -> Optional[genai_types.Part]:
        """Resolve an image/video URL: data URIs decode inline, remote URLs
        are downloaded (Gemini inline_data requires raw bytes)."""
        if not url:
            return None
        if url.startswith("data:"):
            return cls._media_part_from_data_uri(url, default_mime)
        if url.startswith(("http://", "https://")):
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            mime = resp.headers.get("content-type", "").split(";")[0].strip()
            if not mime:
                guessed, _ = mimetypes.guess_type(urlparse(url).path)
                mime = guessed or default_mime
            return genai_types.Part.from_bytes(data=resp.content, mime_type=mime)
        # Local filesystem path
        try:
            with open(url, "rb") as f:
                data = f.read()
        except OSError:
            return None
        guessed, _ = mimetypes.guess_type(url)
        return genai_types.Part.from_bytes(data=data, mime_type=guessed or default_mime)

    # ------------------------------------------------------------------ #
    # from_provider_response / from_provider_chunk                         #
    # ------------------------------------------------------------------ #

    def from_provider_response(self, response: Any) -> AIMessage:
        """Convert a complete GenerateContentResponse to an AIMessage."""
        candidate = self._first_candidate(response)
        content_text, reasoning_text, tool_calls = self._extract_parts(
            candidate.content.parts if candidate and candidate.content else []
        )
        return AIMessage(
            content=content_text,
            reasoning_content=reasoning_text or None,
            tool_calls=tool_calls or None,
            metadata=self._build_metadata(response, candidate),
        )

    def from_provider_chunk(self, chunk: Any) -> UnifiedStreamChunk:
        """Convert one streamed GenerateContentResponse to UnifiedStreamChunk.

        Unlike OpenAI deltas, Gemini stream chunks carry complete parts:
        text parts are incremental fragments, but a function_call part is
        always whole, so each maps to one complete ToolCallDelta.
        """
        candidate = self._first_candidate(chunk)
        parts = candidate.content.parts if candidate and candidate.content else []
        content_text, reasoning_text, deltas = self._extract_parts(parts, streaming=True)

        raw_finish = getattr(candidate, "finish_reason", None) if candidate else None
        # Gemini reports STOP even when the turn ends in a function call;
        # the manager's tool loop keys off the OpenAI-style "tool_calls"
        # value, so normalize it here at the provider boundary. The call
        # part and the finish reason arrive in different chunks, hence
        # the stateful _saw_function_call flag.
        tool_calls_present = bool(deltas) or self._saw_function_call
        finish_reason = self._normalize_finish_reason(raw_finish, tool_calls_present)
        metadata = self._build_chunk_metadata(chunk, raw_finish, tool_calls_present)
        return UnifiedStreamChunk(
            content=content_text,
            reasoning_content=reasoning_text or None,
            tool_calls_delta=deltas,
            finish_reason=finish_reason,
            is_final=bool(raw_finish),
            metadata=metadata,
        )

    @staticmethod
    def _first_candidate(response: Any) -> Any:
        candidates = getattr(response, "candidates", None)
        return candidates[0] if candidates else None

    def _extract_parts(self, parts: List[Any], streaming: bool = False):
        """Split Gemini parts into (text, reasoning, tool_calls|deltas).

        Thought parts are surfaced as reasoning_content; function_call
        parts produce tool calls and cache their thought_signature.
        """
        content_text = ""
        reasoning_text = ""
        tool_calls: List[ChatCompletionMessageToolCall] = []
        deltas: List[ToolCallDelta] = []

        for part in parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                # Native ids exist on recent models; synthesize one when
                # absent so downstream tool_call_id matching still works.
                call_id = getattr(fc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                self._call_names[call_id] = fc.name
                signature = getattr(part, "thought_signature", None)
                if signature:
                    # The SDK hands back decoded bytes; normalize to the
                    # base64 str form the Part constructor expects on input.
                    if isinstance(signature, bytes):
                        signature = base64.b64encode(signature).decode("ascii")
                    self._thought_signatures[call_id] = signature
                args_json = json.dumps(fc.args or {}, ensure_ascii=False)
                if streaming:
                    deltas.append(ToolCallDelta(
                        index=self._next_tool_index,
                        id=call_id,
                        type="function",
                        function_name=fc.name,
                        function_arguments=args_json,
                    ))
                    self._next_tool_index += 1
                    self._saw_function_call = True
                else:
                    tool_calls.append(ChatCompletionMessageToolCall(
                        id=call_id,
                        type="function",
                        function=Function(name=fc.name, arguments=args_json),
                    ))
            elif getattr(part, "thought", None) and part.text:
                reasoning_text += part.text
            elif part.text:
                content_text += part.text

        if streaming:
            return content_text, reasoning_text, deltas
        return content_text, reasoning_text, tool_calls

    @staticmethod
    def _normalize_finish_reason(raw_finish: Any, tool_calls_present: bool) -> Optional[str]:
        """Map Gemini finish reasons to OpenAI-style strings.

        Gemini has no dedicated tool-calls finish reason: a turn that ends
        in a function call still reports STOP. Downstream (manager tool
        loop) expects "tool_calls", so synthesize it from the parts.
        """
        if raw_finish is None:
            return None
        if tool_calls_present:
            return "tool_calls"
        name = getattr(raw_finish, "name", None) or str(raw_finish)
        return name.lower()

    # ------------------------------------------------------------------ #
    # Metadata / usage                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_usage(raw_usage: Any) -> Optional[Usage]:
        """Map Gemini usage into chak's four disjoint buckets.

        Gemini's prompt_token_count INCLUDES cached_content_token_count,
        so strip the cache subset to keep the canonical invariant
        total = pt + ct + cc + cr. Thoughts count as generated output.
        """
        if raw_usage is None:
            return None
        prompt = int(getattr(raw_usage, "prompt_token_count", 0) or 0)
        completion = int(getattr(raw_usage, "candidates_token_count", 0) or 0)
        cache_read = int(getattr(raw_usage, "cached_content_token_count", 0) or 0)
        thoughts = int(getattr(raw_usage, "thoughts_token_count", 0) or 0)
        fresh_prompt = max(prompt - cache_read, 0)
        total = fresh_prompt + completion + thoughts + cache_read
        return Usage(
            prompt_tokens=fresh_prompt,
            completion_tokens=completion + thoughts,
            total_tokens=total,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cache_read,
        )

    def _build_metadata(self, response: Any, candidate: Any) -> Metadata:
        tool_calls_present = bool(
            getattr(candidate, "content", None)
            and any(getattr(p, "function_call", None) for p in (candidate.content.parts or []))
        )
        return Metadata(
            provider="google",
            model=getattr(response, "model", None),
            usage=self._normalize_usage(getattr(response, "usage_metadata", None)),
            finish_reason=self._normalize_finish_reason(
                getattr(candidate, "finish_reason", None), tool_calls_present
            ),
        )

    def _build_chunk_metadata(self, chunk: Any, finish_reason: Any, tool_calls_present: bool = False) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "provider": "google",
            "model": getattr(chunk, "model", None),
            "finish_reason": self._normalize_finish_reason(finish_reason, tool_calls_present),
        }
        usage = self._normalize_usage(getattr(chunk, "usage_metadata", None))
        if usage is not None:
            metadata["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
            }
        return metadata


class GoogleProvider(Provider):
    """Google AI provider built on the official google-genai SDK."""

    def _initialize_client(self) -> None:
        """Initialize the google-genai client.

        api_version stays on v1beta: it exposes the newest Gemini features
        (thinking config, thought signatures) ahead of v1.
        """
        http_kwargs: Dict[str, Any] = {
            "timeout": int(self.config.timeout) * 1000,  # SDK expects milliseconds
            "api_version": "v1beta",
        }
        if self.config.base_url:
            http_kwargs["base_url"] = self.config.base_url
        self._client = genai.Client(
            api_key=self.config.api_key,
            http_options=genai_types.HttpOptions(**http_kwargs),
        )

    # ------------------------------------------------------------------ #
    # Request building                                                     #
    # ------------------------------------------------------------------ #

    def _build_config(self, provider_data: Dict[str, Any], **kwargs) -> genai_types.GenerateContentConfig:
        """Translate OpenAI-style kwargs into a native GenerateContentConfig."""
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        response_format = kwargs.pop("response_format", None)
        reasoning = kwargs.pop("reasoning", None)
        max_tokens = kwargs.pop("max_tokens", None)
        stop = kwargs.pop("stop", None)
        # OpenAI/compat-only params with no Gemini equivalent
        kwargs.pop("stream_options", None)
        kwargs.pop("timeout", None)
        kwargs.pop("parallel_tool_calls", None)

        config_kwargs: Dict[str, Any] = {}
        if provider_data.get("system_instruction") is not None:
            config_kwargs["system_instruction"] = provider_data["system_instruction"]

        if tools:
            declarations = []
            for tool in tools:
                if tool.get("type") != "function":
                    continue
                func = tool.get("function", {})
                declarations.append(genai_types.FunctionDeclaration(
                    name=func.get("name"),
                    description=func.get("description"),
                    # parameters_json_schema accepts a plain JSON schema
                    # dict without Gemini's Schema type wrapping.
                    parameters_json_schema=func.get("parameters") or {
                        "type": "object", "properties": {}
                    },
                ))
            if declarations:
                config_kwargs["tools"] = [genai_types.Tool(function_declarations=declarations)]

        if tool_choice is not None:
            config_kwargs["tool_config"] = self._convert_tool_choice(tool_choice)

        if response_format is not None:
            rf_type = response_format.get("type")
            if rf_type == "json_schema":
                schema = (response_format.get("json_schema") or {}).get("schema")
                if schema is not None:
                    config_kwargs["response_schema"] = schema
                config_kwargs["response_mime_type"] = "application/json"
            elif rf_type == "json_object":
                config_kwargs["response_mime_type"] = "application/json"

        if reasoning is not None:
            thinking = self._convert_reasoning(reasoning)
            if thinking is not None:
                config_kwargs["thinking_config"] = thinking

        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = int(max_tokens)
        if stop is not None:
            config_kwargs["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        # Forward sampling params that Gemini natively understands
        for key in ("temperature", "top_p", "top_k", "seed",
                    "presence_penalty", "frequency_penalty"):
            if key in kwargs:
                config_kwargs[key] = kwargs.pop(key)

        return genai_types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _convert_tool_choice(tool_choice: Any) -> genai_types.ToolConfig:
        """Map OpenAI tool_choice values to Gemini FunctionCallingConfig."""
        mode = "AUTO"
        allowed: Optional[List[str]] = None
        if isinstance(tool_choice, str):
            if tool_choice == "required":
                mode = "ANY"
            elif tool_choice == "none":
                mode = "NONE"
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                name = (tool_choice.get("function") or {}).get("name")
                mode = "ANY"
                allowed = [name] if name else None
        return genai_types.ToolConfig(
            function_calling_config=genai_types.FunctionCallingConfig(
                mode=mode, allowed_function_names=allowed
            )
        )

    @staticmethod
    def _convert_reasoning(reasoning: Any) -> Optional[genai_types.ThinkingConfig]:
        """Map chak's unified reasoning param to Gemini ThinkingConfig."""
        if isinstance(reasoning, Reasoning):
            reasoning_dict = reasoning.model_dump(exclude_none=True)
        elif isinstance(reasoning, dict):
            reasoning_dict = reasoning
        else:
            return None
        budget = reasoning_dict.get("budget")
        if budget is None:
            budget = {"low": 2000, "medium": 8000, "high": 16000}.get(
                reasoning_dict.get("effort", "medium"), 8000
            )
        return genai_types.ThinkingConfig(thinking_budget=int(budget))

    # ------------------------------------------------------------------ #
    # send / _send_complete / _send_stream                                 #
    # ------------------------------------------------------------------ #

    def _send_complete(self, provider_data: Dict[str, Any], **kwargs) -> Any:
        """Send a non-streaming request via the native SDK."""
        config = self._build_config(provider_data, **kwargs)
        return self._client.models.generate_content(
            model=self.config.model,
            contents=provider_data["contents"],
            config=config,
        )

    def _send_stream(self, provider_data: Dict[str, Any], **kwargs) -> Any:
        """Send a streaming request; returns the SDK chunk iterator."""
        self.converter._reset_stream_state()
        config = self._build_config(provider_data, **kwargs)
        return self._client.models.generate_content_stream(
            model=self.config.model,
            contents=provider_data["contents"],
            config=config,
        )

    # ------------------------------------------------------------------ #
    # Error mapping                                                        #
    # ------------------------------------------------------------------ #

    def _normalize_error(self, error: BaseException) -> ProviderError:
        """Map google-genai / httpx exceptions to ProviderError.

        The SDK raises ClientError (4xx) / ServerError (5xx), both
        subclasses of APIError carrying the HTTP status in ``.code``.
        """
        if isinstance(error, ProviderError):
            error.provider = error.provider or self.provider_name
            error.model = error.model or self.config.model
            error.base_url = error.base_url or getattr(self.config, "base_url", None)
            return error

        base_url = getattr(self.config, "base_url", None)

        if isinstance(error, genai_errors.APIError):
            status_code = getattr(error, "code", None)
            return ProviderError(
                f"{self.__class__.__name__} error: {getattr(error, 'message', str(error))}",
                provider=self.provider_name,
                model=self.config.model,
                base_url=base_url,
                status_code=status_code,
                error_type=ErrorType.from_status_code(status_code),
                raw_error=error,
            )

        if isinstance(error, httpx.TimeoutException):
            return ProviderError(
                f"{self.__class__.__name__} timeout: {error}",
                provider=self.provider_name,
                model=self.config.model,
                base_url=base_url,
                status_code=None,
                error_type=ErrorType.TIMEOUT,
                raw_error=error,
            )

        if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
            return ProviderError(
                f"{self.__class__.__name__} connection error: {error}",
                provider=self.provider_name,
                model=self.config.model,
                base_url=base_url,
                status_code=None,
                error_type=ErrorType.CONNECTION_ERROR,
                raw_error=error,
            )

        return ProviderError(
            f"{self.__class__.__name__} error: {error}",
            provider=self.provider_name,
            model=self.config.model,
            base_url=base_url,
            status_code=None,
            error_type=ErrorType.UNKNOWN,
            raw_error=error,
        )
