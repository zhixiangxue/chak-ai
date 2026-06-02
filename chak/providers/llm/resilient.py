from typing import Any, Dict, Iterator, List, Optional

from ...exceptions import ProviderError, ErrorType, RETRYABLE_STATUS_CODES
from ...message import FailoverChunk, Message, UnifiedStreamChunk
from ...metadata import Metadata, ProviderTrace, FailureRecord
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
                        failed_provider=provider.provider_name,
                        next_provider=next_provider.provider_name,
                        error=str(error),
                    )

        raise self._build_provider_error(failures)

    def _should_try_next(self, error: Exception, index: int) -> bool:
        return index < len(self.providers) - 1 and is_retryable_provider_error(error)

    def _annotate_message(self, message: Message, provider: Provider, failures: List[Dict[str, Any]]) -> None:
        metadata = getattr(message, "metadata", None)
        if isinstance(metadata, Metadata):
            if not metadata.provider:
                metadata.provider = provider.provider_name
            if metadata.model is None:
                metadata.model = provider.model_name
            metadata.provider_trace = self._build_provider_trace(provider, failures)
        elif isinstance(metadata, dict):
            metadata.update(self._annotate_metadata_dict(metadata, provider, failures))

    def _annotate_metadata_dict(
        self,
        metadata: Optional[Dict[str, Any]],
        provider: Provider,
        failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        annotated = dict(metadata or {})
        annotated.setdefault("provider", provider.provider_name)
        annotated.setdefault("model", provider.model_name)
        trace = self._build_provider_trace(provider, failures)
        annotated["provider_trace"] = trace.model_dump()
        return annotated

    def _build_provider_trace(self, provider: Provider, failures: List[Dict[str, Any]]) -> ProviderTrace:
        return ProviderTrace(
            primary_provider=self.primary_provider.provider_name,
            primary_model=self.primary_provider.model_name,
            fallback_used=bool(failures),
            failover_attempts=len(failures),
            failed_providers=[FailureRecord(**f) for f in failures],
            resolved_provider=provider.provider_name,
            resolved_model=provider.model_name,
        )

    @staticmethod
    def _failure_record(provider: Provider, attempt_index: int, error: Exception) -> Dict[str, Any]:
        record = {
            "attempt_index": attempt_index,
            "provider": provider.provider_name,
            "model": provider.model_name,
            "base_url": ResilientProvider._base_url(provider),
            "error": str(error),
        }
        if isinstance(error, ProviderError):
            record["status_code"] = error.status_code
            record["error_type"] = error.error_type
        return record

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
            error_type=ErrorType.FAILOVER_EXHAUSTED,
        )


def is_retryable_provider_error(error: BaseException) -> bool:
    """Decide whether *error* should trigger a resilient failover.

    This function is the single gatekeeper for the retry-or-fail decision
    in :class:`ResilientProvider`.  It MUST receive a :class:`ProviderError`
    — all provider SDK errors are precisely mapped into ProviderError by
    each provider's ``_provider_error()`` override, so raw exceptions should
    never reach this function.

    Retryable (try next provider):
        - error_type in {"timeout", "connection_error", "rate_limit", "server_error"}
        - status_code in {408, 429, 500, 502, 503, 504}

    NOT retryable (raise immediately):
        - error_type == "unknown" (unrecognized error — be conservative)
        - error_type in {"auth_error", "bad_request", "not_found"}
        - Any 4xx status_code not in the retryable set
        - The error is not a ProviderError at all
    """
    if not isinstance(error, ProviderError):
        return False

    if error.error_type in ErrorType.RETRYABLE:
        return True
    if error.status_code in RETRYABLE_STATUS_CODES:
        return True
    if error.status_code is not None and 400 <= error.status_code < 500:
        return False
    return False


def _find_provider_error(error: BaseException) -> Optional[ProviderError]:
    """Walk the exception chain and return the first ProviderError found.

    Used by :func:`is_retryable_provider_error` (which first does a direct
    ``isinstance`` check) and by tests that need to verify the wrapped
    ProviderError inside a chained exception.
    """
    current: Optional[BaseException] = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderError):
            return current
        current = current.__cause__ or current.__context__
    return None
