# src/chak/exceptions.py
"""
Exception hierarchy for chak.
"""
from typing import Optional


# =========================================================================
# Error type constants — single source of truth
# =========================================================================


class ErrorType:
    """Canonical error_type values for ProviderError.

    All provider ``_provider_error()`` overrides and the resilient failover
    logic MUST use these constants — never raw strings.
    """

    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    AUTH_ERROR = "auth_error"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    FAILOVER_EXHAUSTED = "failover_exhausted"
    UNKNOWN = "unknown"

    # Which error types should trigger resilient failover
    RETRYABLE = frozenset({TIMEOUT, CONNECTION_ERROR, RATE_LIMIT, SERVER_ERROR})

    @classmethod
    def from_status_code(cls, status_code: Optional[int]) -> str:
        """Map an HTTP status code to a canonical error_type string.

        This is a deterministic lookup — no heuristics.  Used by provider
        ``_provider_error()`` overrides after extracting the exact status code
        from the SDK exception.
        """
        if status_code is None:
            return cls.UNKNOWN
        if status_code == 400:
            return cls.BAD_REQUEST
        if status_code in {401, 403}:
            return cls.AUTH_ERROR
        if status_code == 404:
            return cls.NOT_FOUND
        if status_code == 408:
            return cls.TIMEOUT
        if status_code == 429:
            return cls.RATE_LIMIT
        if 500 <= status_code < 600:
            return cls.SERVER_ERROR
        return cls.UNKNOWN


# HTTP status codes that should trigger resilient failover
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


# =========================================================================
# Exception classes
# =========================================================================


class ChakError(Exception):
    """Base exception for all chak errors."""
    pass


class ProviderError(ChakError):
    """Errors related to LLM providers.

    Every SDK exception MUST be precisely mapped into a ProviderError by
    the owning provider's ``_provider_error()`` override.  The ``error_type``
    field drives resilient failover decisions, so it must be accurate.

    Use :class:`ErrorType` constants for the ``error_type`` field.
    """

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
        """Wrap a raw exception into a ProviderError.

        If *error* is already a ProviderError, merge in any missing metadata
        and return it unchanged.

        Otherwise, create a new ProviderError with error_type="unknown".
        Provider subclasses should NOT rely on this fallback — they should
        override ``_provider_error()`` for precise SDK-level mapping.
        """
        if isinstance(error, ProviderError):
            error.provider = error.provider or provider
            error.model = error.model or model
            error.base_url = error.base_url or base_url
            return error

        return cls(
            message or str(error),
            provider=provider,
            model=model,
            base_url=base_url,
            status_code=None,
            error_type=ErrorType.UNKNOWN,
            raw_error=error,
        )


# Backward-compatible alias (deprecated)
map_status_to_error_type = ErrorType.from_status_code


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