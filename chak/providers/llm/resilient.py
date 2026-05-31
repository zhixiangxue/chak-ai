from typing import Any, Dict, Iterator, List, Optional

from ...exceptions import ProviderError
from ...message import FailoverChunk, Message, UnifiedStreamChunk
from ...metadata import Metadata
from .base import BaseMessageConverter, Provider


class ResilientMessageConverter(BaseMessageConverter):
    """Pass-through converter for chunks already normalized by child providers."""

    def to_provider_format(self, messages: List[Message]) -> List[Message]:
        return messages

    def from_provider_response(self, response: Any) -> Message:
        return response

    def from_provider_chunk(self, chunk: Any) -> UnifiedStreamChunk:
        if isinstance(chunk, UnifiedStreamChunk):
            return chunk
        raise TypeError(f"Expected UnifiedStreamChunk, got {type(chunk).__name__}")


class ResilientProvider(Provider):
    """Provider proxy that retries a request through ordered fallback providers."""

    def __init__(self, primary_provider: Provider, fallback_providers: List[Provider]):
        self.primary_provider = primary_provider
        self.fallback_providers = list(fallback_providers)
        self.providers = [primary_provider, *self.fallback_providers]
        self.config = primary_provider.config
        self.converter = ResilientMessageConverter()
        self._client = None

    def _initialize_client(self):
        pass

    def _send_complete(self, messages: Any, **kwargs) -> Any:
        raise NotImplementedError("ResilientProvider uses send() directly")

    def _send_stream(self, messages: Any, **kwargs) -> Iterator[Any]:
        raise NotImplementedError("ResilientProvider uses send() directly")

    def send(self, messages: List[Message], stream: bool = False, **kwargs):
        if stream:
            return self._send_stream_with_fallback(messages, **kwargs)
        return self._send_nonstream_with_fallback(messages, **kwargs)

    def _send_nonstream_with_fallback(self, messages: List[Message], **kwargs) -> Message:
        failures: List[Dict[str, Any]] = []

        for index, provider in enumerate(self.providers):
            try:
                response = provider.send(messages=messages, stream=False, **kwargs)
                self._annotate_message(response, provider, failures)
                return response
            except Exception as error:
                failures.append(self._failure_record(provider, index, error))
                if not self._should_try_next(error, index):
                    raise self._build_provider_error(failures) from error

        raise self._build_provider_error(failures)

    def _send_stream_with_fallback(self, messages: List[Message], **kwargs) -> Iterator[Any]:
        failures: List[Dict[str, Any]] = []

        for index, provider in enumerate(self.providers):
            user_visible_yielded = False
            try:
                provider_chunks = provider.send(messages=messages, stream=True, **kwargs)
                for provider_chunk in provider_chunks:
                    unified_chunk = provider.converter.from_provider_chunk(provider_chunk)
                    unified_chunk.metadata = self._annotate_metadata_dict(unified_chunk.metadata, provider, failures)
                    if unified_chunk.content or unified_chunk.reasoning_content:
                        user_visible_yielded = True
                    yield unified_chunk
                return
            except Exception as error:
                failures.append(self._failure_record(provider, index, error))
                if not self._should_try_next(error, index):
                    raise self._build_provider_error(failures) from error

                next_provider = self.providers[index + 1]
                if user_visible_yielded:
                    yield FailoverChunk(
                        failed_provider=self._provider_name(provider),
                        next_provider=self._provider_name(next_provider),
                        error=str(error),
                    )

        raise self._build_provider_error(failures)

    def _should_try_next(self, error: Exception, index: int) -> bool:
        return index < len(self.providers) - 1 and is_retryable_provider_error(error)

    def _annotate_message(self, message: Message, provider: Provider, failures: List[Dict[str, Any]]) -> None:
        metadata = getattr(message, "metadata", None)
        if isinstance(metadata, Metadata):
            if not metadata.provider:
                metadata.provider = self._provider_name(provider)
            if metadata.model is None:
                metadata.model = self._model_name(provider)
            metadata.extra.update(self._failover_metadata(provider, failures))
        elif isinstance(metadata, dict):
            metadata.update(self._annotate_metadata_dict(metadata, provider, failures))

    def _annotate_metadata_dict(
        self,
        metadata: Optional[Dict[str, Any]],
        provider: Provider,
        failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        annotated = dict(metadata or {})
        annotated.setdefault("provider", self._provider_name(provider))
        annotated.setdefault("model", self._model_name(provider))
        annotated.update(self._failover_metadata(provider, failures))
        return annotated

    def _failover_metadata(self, provider: Provider, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "primary_provider": self._provider_name(self.primary_provider),
            "fallback_used": bool(failures),
            "failover_attempts": len(failures),
            "failed_providers": list(failures),
            "resolved_provider": self._provider_name(provider),
            "resolved_model": self._model_name(provider),
        }

    @staticmethod
    def _failure_record(provider: Provider, attempt_index: int, error: Exception) -> Dict[str, Any]:
        record = {
            "attempt_index": attempt_index,
            "provider": ResilientProvider._provider_name(provider),
            "model": ResilientProvider._model_name(provider),
            "base_url": ResilientProvider._base_url(provider),
            "error": str(error),
        }
        if isinstance(error, ProviderError):
            record["status_code"] = error.status_code
            record["error_type"] = error.error_type
        return record

    @staticmethod
    def _provider_name(provider: Provider) -> str:
        return str(getattr(provider.config, "provider_name", provider.__class__.__name__))

    @staticmethod
    def _model_name(provider: Provider) -> str:
        return str(getattr(provider.config, "model", ""))

    @staticmethod
    def _base_url(provider: Provider) -> str:
        base_url = getattr(provider.config, "base_url", None)
        return "" if base_url is None else str(base_url)

    @staticmethod
    def _build_provider_error(failures: List[Dict[str, Any]]) -> ProviderError:
        details = "; ".join(
            f"{item['provider']}/{item['model']}: {item['error']}" for item in failures
        )
        return ProviderError(
            f"All resilient provider attempts failed: {details}",
            error_type="all_attempts_failed",
        )


def is_retryable_provider_error(error: BaseException) -> bool:
    provider_error = _find_provider_error(error)
    if provider_error is not None:
        if provider_error.error_type in {"timeout", "connection_error", "rate_limit", "server_error"}:
            return True
        if provider_error.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        if provider_error.status_code is not None and 400 <= provider_error.status_code < 500:
            return False
        return False

    status_code = _find_status_code(error)
    if status_code is not None:
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        if 400 <= status_code < 500:
            return False

    for current in _iter_error_chain(error):
        name = type(current).__name__.lower()
        if any(token in name for token in ("timeout", "connection", "connect", "ratelimit")):
            return True

    return False


def _find_provider_error(error: BaseException) -> Optional[ProviderError]:
    for current in _iter_error_chain(error):
        if isinstance(current, ProviderError):
            return current
    return None


def _find_status_code(error: BaseException) -> Optional[int]:
    for current in _iter_error_chain(error):
        status = getattr(current, "status_code", None)
        if status is None and hasattr(current, "response"):
            status = getattr(getattr(current, "response"), "status_code", None)
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                return None
    return None


def _iter_error_chain(error: BaseException):
    current: Optional[BaseException] = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__
