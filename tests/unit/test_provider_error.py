"""Test provider _normalize_error against real SDK exception classes.

These tests verify that real SDK exceptions (openai, anthropic) are correctly
normalized into ProviderError by the provider's _normalize_error method, and
that is_retryable_provider_error makes the correct failover decision.

Design principles
-----------------
1. All tests use REAL SDK exception classes with REAL httpx.Request/Response
   objects — NO Mock(). The exception structures have been verified against
   actual SDK source code.

2. Each test verifies the COMPLETE pipeline:
   SDK error → provider._normalize_error → status_code / error_type
   SDK error → provider._normalize_error → is_retryable_provider_error

3. The three SDK families tested:
   - **openai** (used by: openai, deepseek, azure, moonshot, siliconflow,
     volcengine, tencent, zhipu, minimax, iflytek, xai, mistral, ollama, vllm)
   - **anthropic** (used by: anthropic)
   - **dashscope** (used by: bailian — special: raises ProviderError directly)
"""
import httpx
import pytest

from chak.exceptions import ErrorType, ProviderError
from chak.message import AIMessage, UnifiedStreamChunk
from chak.providers.llm.base import (
    BaseMessageConverter,
    BaseProviderConfig,
    OpenAICompatibleProvider,
)
from chak.providers.llm.resilient import is_retryable_provider_error

pytestmark = pytest.mark.unit


# =============================================================================
# Test infrastructure: minimal provider stubs for calling _normalize_error
# =============================================================================


class StubConverter(BaseMessageConverter):
    """Minimal converter stub — not called in error path tests."""
    def to_provider_format(self, messages):
        return messages

    def from_provider_response(self, response):
        return response

    def from_provider_chunk(self, chunk):
        return chunk


class StubOpenAIProvider(OpenAICompatibleProvider):
    """Concrete OpenAI-compatible provider for testing _normalize_error."""
    def _initialize_client(self):
        pass  # skip real client creation

    def _send_complete(self, messages, **kwargs):
        return AIMessage(content="ok")

    def _send_stream(self, messages, **kwargs):
        yield UnifiedStreamChunk(content="ok")


def _make_openai_provider(provider_name="openai", model="gpt-4o-mini", base_url="https://api.openai.com/v1"):
    config = BaseProviderConfig(
        api_key="test-key",
        model=model,
        base_url=base_url,
        provider_name=provider_name,
    )
    return StubOpenAIProvider(config, StubConverter())


def _make_anthropic_provider(provider_name="anthropic", model="claude-haiku-4-5", base_url="https://api.anthropic.com"):
    """Create a minimal AnthropicProvider-like instance for testing _normalize_error.

    We reuse the AnthropicProvider class directly since its _normalize_error
    only depends on self.provider_name, self.config.model, self.config.base_url.
    """
    from chak.providers.llm.anthropic import AnthropicProvider, AnthropicConfig

    class StubAnthropicProvider(AnthropicProvider):
        def _initialize_client(self):
            pass  # skip real client

        def _send_complete(self, messages, **kwargs):
            return AIMessage(content="ok")

        def _send_stream(self, messages, **kwargs):
            yield UnifiedStreamChunk(content="ok")

    config = AnthropicConfig(
        api_key="test-key",
        model=model,
        base_url=base_url,
        provider_name=provider_name,
    )
    return StubAnthropicProvider(config, StubConverter())


# =============================================================================
# Helpers: build REAL httpx objects for SDK exception constructors
# =============================================================================


def _request(method: str = "POST", url: str = "https://api.example.com/v1/chat") -> httpx.Request:
    """Build a real httpx.Request — the same object the SDK creates at runtime."""
    return httpx.Request(method, url)


def _response(status_code: int, *, request: httpx.Request = None) -> httpx.Response:
    """Build a real httpx.Response with a status_code and linked request."""
    req = request or _request()
    return httpx.Response(status_code, request=req, json={"error": {"message": f"error {status_code}"}})


# =============================================================================
# openai SDK errors — used by 15+ providers via OpenAICompatibleProvider
# =============================================================================


class TestOpenAISDKErrors:
    """openai SDK exception → _normalize_error → is_retryable."""

    @pytest.fixture
    def provider(self):
        return _make_openai_provider()

    # -- APIStatusError (has status_code + response) -------------------------

    @pytest.mark.parametrize(
        "status_code,error_type,retryable",
        [
            (429, ErrorType.RATE_LIMIT, True),
            (500, ErrorType.SERVER_ERROR, True),
            (503, ErrorType.SERVER_ERROR, True),
            (400, ErrorType.BAD_REQUEST, False),
            (401, ErrorType.AUTH_ERROR, False),
            (403, ErrorType.AUTH_ERROR, False),
            (404, ErrorType.NOT_FOUND, False),
        ],
    )
    def test_api_status_error(self, provider, status_code, error_type, retryable):
        """Real openai.APIStatusError with real httpx.Response."""
        from openai import APIStatusError

        raw = APIStatusError(
            f"HTTP {status_code}",
            response=_response(status_code),
            body={"error": {"message": f"error {status_code}"}},
        )

        error = provider._normalize_error(raw)

        assert error.status_code == status_code
        assert error.error_type == error_type
        assert error.provider == "openai"
        assert error.model == "gpt-4o-mini"
        assert error.base_url == "https://api.openai.com/v1"
        assert error.raw_error is raw
        assert is_retryable_provider_error(error) is retryable

    # -- Subclass: RateLimitError --------------------------------------------

    def test_rate_limit_error_subclass(self, provider):
        """openai.RateLimitError is a subclass of APIStatusError, status_code=429."""
        from openai import RateLimitError

        raw = RateLimitError(
            "Rate limit exceeded",
            response=_response(429),
            body={"error": {"message": "rate limit"}},
        )

        error = provider._normalize_error(raw)

        assert error.status_code == 429
        assert error.error_type == ErrorType.RATE_LIMIT
        assert is_retryable_provider_error(error) is True

    def test_authentication_error_subclass(self, provider):
        """openai.AuthenticationError is a subclass of APIStatusError, status_code=401."""
        from openai import AuthenticationError

        raw = AuthenticationError(
            "Invalid API key",
            response=_response(401),
            body={"error": {"message": "invalid key"}},
        )

        error = provider._normalize_error(raw)

        assert error.status_code == 401
        assert error.error_type == ErrorType.AUTH_ERROR
        assert is_retryable_provider_error(error) is False

    # -- APIConnectionError (no status_code) ---------------------------------

    def test_connection_error(self, provider):
        """openai.APIConnectionError — no status_code, error_type='connection_error'."""
        from openai import APIConnectionError

        raw = APIConnectionError(request=_request())

        error = provider._normalize_error(raw)

        assert error.status_code is None
        assert error.error_type == ErrorType.CONNECTION_ERROR
        assert is_retryable_provider_error(error) is True

    # -- APITimeoutError (no status_code) ------------------------------------

    def test_timeout_error(self, provider):
        """openai.APITimeoutError — no status_code, error_type='timeout'."""
        from openai import APITimeoutError

        raw = APITimeoutError(request=_request())

        error = provider._normalize_error(raw)

        assert error.status_code is None
        assert error.error_type == ErrorType.TIMEOUT
        assert is_retryable_provider_error(error) is True

    # -- Unrecognized error → UNKNOWN (not retryable) ------------------------

    def test_unrecognized_error_is_unknown(self, provider):
        """Non-SDK exception → error_type='unknown', NOT retryable."""
        raw = ValueError("something unexpected")

        error = provider._normalize_error(raw)

        assert error.status_code is None
        assert error.error_type == ErrorType.UNKNOWN
        assert error.raw_error is raw
        assert is_retryable_provider_error(error) is False


# =============================================================================
# anthropic SDK errors
# =============================================================================


class TestAnthropicSDKErrors:
    """anthropic SDK exception → _normalize_error → is_retryable."""

    @pytest.fixture
    def provider(self):
        return _make_anthropic_provider()

    @pytest.mark.parametrize(
        "status_code,error_type,retryable",
        [
            (429, ErrorType.RATE_LIMIT, True),
            (500, ErrorType.SERVER_ERROR, True),
            (503, ErrorType.SERVER_ERROR, True),
            (529, ErrorType.SERVER_ERROR, True),  # Anthropic-specific: overloaded
            (400, ErrorType.BAD_REQUEST, False),
            (401, ErrorType.AUTH_ERROR, False),
            (404, ErrorType.NOT_FOUND, False),
        ],
    )
    def test_api_status_error(self, provider, status_code, error_type, retryable):
        """Real anthropic.APIStatusError with real httpx.Response."""
        from anthropic import APIStatusError

        raw = APIStatusError(
            f"HTTP {status_code}",
            response=_response(status_code),
            body=None,
        )

        error = provider._normalize_error(raw)

        assert error.status_code == status_code
        assert error.error_type == error_type
        assert error.provider == "anthropic"
        assert error.model == "claude-haiku-4-5"
        assert error.raw_error is raw
        assert is_retryable_provider_error(error) is retryable

    def test_connection_error(self, provider):
        """anthropic.APIConnectionError — connection_error, retryable."""
        from anthropic import APIConnectionError

        raw = APIConnectionError(request=_request())

        error = provider._normalize_error(raw)

        assert error.status_code is None
        assert error.error_type == ErrorType.CONNECTION_ERROR
        assert is_retryable_provider_error(error) is True

    def test_timeout_error(self, provider):
        """anthropic.APITimeoutError — timeout, retryable."""
        from anthropic import APITimeoutError

        raw = APITimeoutError(request=_request())

        error = provider._normalize_error(raw)

        assert error.status_code is None
        assert error.error_type == ErrorType.TIMEOUT
        assert is_retryable_provider_error(error) is True


# =============================================================================
# DeepSeek — uses openai SDK with different provider_name
# =============================================================================


def test_deepseek_rate_limit():
    """DeepSeek 429 → rate_limit, retryable."""
    from openai import APIStatusError

    provider = _make_openai_provider(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )

    raw = APIStatusError(
        "rate limited",
        response=_response(429),
        body=None,
    )

    error = provider._normalize_error(raw)

    assert error.status_code == 429
    assert error.error_type == ErrorType.RATE_LIMIT
    assert error.provider == "deepseek"
    assert error.model == "deepseek-v4-flash"
    assert error.base_url == "https://api.deepseek.com"
    assert is_retryable_provider_error(error) is True


# =============================================================================
# Bailian (DashScope) — raises ProviderError directly
# =============================================================================


class TestBailianErrors:
    """Bailian / DashScope error handling.

    DashScope is unique: it returns HTTP 200 even on API errors. The real
    error status is in the response body's ``status_code`` field.
    ``_check_response_error`` detects this and raises a ``ProviderError``
    directly (NOT an SDK exception).

    Then ``Provider.send()`` catches it, calls ``_normalize_error(e)`` →
    which reuses the existing ProviderError instance (the isinstance check).
    """

    def test_bailian_rate_limit_reuses_provider_error(self):
        """_check_response_error raises ProviderError(status_code=429).

        _normalize_error reuses the instance →
        is_retryable_provider_error sees status_code=429 → True.
        """
        original = ProviderError(
            "DashScope API error (status=429, code=Throttling): rate limited",
            status_code=429,
            error_type=ErrorType.RATE_LIMIT,
        )

        provider = _make_openai_provider(provider_name="bailian", model="qwen-plus")
        wrapped = provider._normalize_error(original)

        # _normalize_error reuses the instance (same object)
        assert wrapped is original
        assert wrapped.status_code == 429
        assert wrapped.provider == "bailian"
        assert wrapped.model == "qwen-plus"
        assert is_retryable_provider_error(wrapped) is True

    def test_bailian_bad_request_not_retryable(self):
        """_check_response_error raises ProviderError(status_code=400)."""
        original = ProviderError(
            "DashScope API error (status=400, code=InvalidParameter): bad request",
            status_code=400,
            error_type=ErrorType.BAD_REQUEST,
        )

        provider = _make_openai_provider(provider_name="bailian", model="qwen-plus")
        wrapped = provider._normalize_error(original)

        assert wrapped is original
        assert wrapped.status_code == 400
        assert is_retryable_provider_error(wrapped) is False

    def test_bailian_server_error_retryable(self):
        """_check_response_error raises ProviderError(status_code=500)."""
        original = ProviderError(
            "DashScope API error (status=500, code=InternalError): server error",
            status_code=500,
            error_type=ErrorType.SERVER_ERROR,
        )

        provider = _make_openai_provider(provider_name="bailian", model="qwen-plus")
        wrapped = provider._normalize_error(original)

        assert wrapped is original
        assert wrapped.status_code == 500
        assert is_retryable_provider_error(wrapped) is True


# =============================================================================
# ProviderError.from_exception: reuse existing ProviderError instances
# =============================================================================


def test_from_exception_reuses_provider_error():
    """When input is already a ProviderError, return the SAME object."""
    original = ProviderError("wrapped", status_code=503, error_type=ErrorType.SERVER_ERROR)

    error = ProviderError.from_exception(
        original,
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )

    assert error is original
    assert error.provider == "deepseek"
    assert error.model == "deepseek-v4-flash"
    assert error.base_url == "https://api.deepseek.com"
    assert error.status_code == 503
    assert error.error_type == ErrorType.SERVER_ERROR


def test_from_exception_fills_only_missing_fields():
    """from_exception does NOT overwrite existing provider/model/base_url."""
    original = ProviderError(
        "wrapped",
        provider="original-prov",
        model="original-model",
        status_code=429,
        error_type=ErrorType.RATE_LIMIT,
    )

    error = ProviderError.from_exception(
        original,
        provider="new-prov",
        model="new-model",
        base_url="https://new.example.com",
    )

    # Existing fields are preserved, missing fields are filled
    assert error.provider == "original-prov"   # NOT overwritten
    assert error.model == "original-model"      # NOT overwritten
    assert error.base_url == "https://new.example.com"  # filled because was None
    assert error.status_code == 429


def test_from_exception_unknown_error_wraps_as_unknown():
    """Non-ProviderError → new ProviderError with error_type='unknown'."""
    raw = RuntimeError("something broke")

    error = ProviderError.from_exception(
        raw, provider="test", model="test-model"
    )

    assert error.error_type == ErrorType.UNKNOWN
    assert error.provider == "test"
    assert error.model == "test-model"
    assert error.raw_error is raw
    assert is_retryable_provider_error(error) is False


# =============================================================================
# End-to-end pipeline: SDK error → _normalize_error → resilient decision
# =============================================================================


class TestEndToEndPipeline:
    """Verify the full pipeline that ResilientProvider relies on.

    ResilientProvider catches exceptions from child providers, which are
    already ProviderError (because Provider.send() wraps via _normalize_error).
    Then is_retryable_provider_error decides whether to fail over.
    """

    def test_rate_limit_triggers_failover(self):
        """429 from openai SDK → _normalize_error → is_retryable=True."""
        from openai import RateLimitError

        provider = _make_openai_provider()
        raw = RateLimitError(
            "Rate limit exceeded",
            response=_response(429),
            body={"error": {"message": "rate limit"}},
        )

        error = provider._normalize_error(raw)
        assert is_retryable_provider_error(error) is True

    def test_auth_error_stops_failover(self):
        """401 from openai SDK → _normalize_error → is_retryable=False."""
        from openai import AuthenticationError

        provider = _make_openai_provider()
        raw = AuthenticationError(
            "Invalid API key",
            response=_response(401),
            body={"error": {"message": "invalid key"}},
        )

        error = provider._normalize_error(raw)
        assert is_retryable_provider_error(error) is False

    def test_connection_error_triggers_failover(self):
        """APIConnectionError → _normalize_error → is_retryable=True."""
        from openai import APIConnectionError

        provider = _make_openai_provider()
        raw = APIConnectionError(request=_request())

        error = provider._normalize_error(raw)
        assert is_retryable_provider_error(error) is True

    def test_anthropic_overloaded_triggers_failover(self):
        """529 (Overloaded) from anthropic → server_error, retryable."""
        from anthropic import APIStatusError

        provider = _make_anthropic_provider()
        raw = APIStatusError(
            "Overloaded",
            response=_response(529),
            body=None,
        )

        error = provider._normalize_error(raw)

        assert error.status_code == 529
        assert error.error_type == ErrorType.SERVER_ERROR
        assert is_retryable_provider_error(error) is True

    def test_anthropic_timeout_triggers_failover(self):
        """APITimeoutError from anthropic → timeout, retryable."""
        from anthropic import APITimeoutError

        provider = _make_anthropic_provider()
        raw = APITimeoutError(request=_request())

        error = provider._normalize_error(raw)

        assert error.error_type == ErrorType.TIMEOUT
        assert is_retryable_provider_error(error) is True
