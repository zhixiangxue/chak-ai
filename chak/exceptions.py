# src/chak/exceptions.py
"""
Exception hierarchy for chak.
"""
from typing import Optional


class ChakError(Exception):
    """Base exception for all chak errors."""
    pass


class ProviderError(ChakError):
    """Errors related to LLM providers."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        raw_error: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.status_code = status_code
        self.error_type = error_type
        self.raw_error = raw_error

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        message: Optional[str] = None,
    ) -> "ProviderError":
        if isinstance(error, ProviderError):
            error.provider = error.provider or provider
            error.model = error.model or model
            error.base_url = error.base_url or base_url
            return error

        status_code = _extract_status_code(error)
        error_type = _classify_error(error, status_code)
        return cls(
            message or str(error),
            provider=provider,
            model=model,
            base_url=base_url,
            status_code=status_code,
            error_type=error_type,
            raw_error=error,
        )


class ConfigError(ChakError):
    """Configuration-related errors."""
    pass


class ConversationNotFoundError(ChakError):
    """Requested conversation not found."""
    pass


class ContextError(ChakError):
    """Context management errors."""
    pass


class URIError(ChakError):
    """URI parsing and validation errors."""
    pass


def _extract_status_code(error: BaseException) -> Optional[int]:
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


def _classify_error(error: BaseException, status_code: Optional[int]) -> Optional[str]:
    if status_code is not None:
        if status_code == 400:
            return "bad_request"
        if status_code in {401, 403}:
            return "auth_error"
        if status_code == 404:
            return "not_found"
        if status_code == 408:
            return "timeout"
        if status_code == 409:
            return "conflict"
        if status_code == 425:
            return "too_early"
        if status_code == 429:
            return "rate_limit"
        if 500 <= status_code < 600:
            return "server_error"
        return "http_error"

    for current in _iter_error_chain(error):
        name = type(current).__name__.lower()
        if "timeout" in name:
            return "timeout"
        if "connection" in name or "connect" in name:
            return "connection_error"
        if "ratelimit" in name or "rate_limit" in name:
            return "rate_limit"

    return None


def _iter_error_chain(error: BaseException):
    current: Optional[BaseException] = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__