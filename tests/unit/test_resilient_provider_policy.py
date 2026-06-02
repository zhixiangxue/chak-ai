from types import SimpleNamespace

import pytest

from chak.exceptions import ProviderError
from chak.providers.llm.resilient import ResilientProvider, is_retryable_provider_error

pytestmark = pytest.mark.unit


# =============================================================================
# ProviderError with status_code
# =============================================================================


@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
def test_retryable_status_codes(status_code):
    """ProviderError with a retryable status_code → True."""
    error = ProviderError("temporary", status_code=status_code)

    assert is_retryable_provider_error(error) is True


@pytest.mark.parametrize("status_code", [400, 401, 402, 403, 404, 405, 406, 409, 410, 422, 425])
def test_non_retryable_client_status_codes(status_code):
    """ProviderError with a 4xx status_code outside retryable set → False."""
    error = ProviderError("config error", status_code=status_code)

    assert is_retryable_provider_error(error) is False


# =============================================================================
# ProviderError with error_type
# =============================================================================


@pytest.mark.parametrize("error_type", ["timeout", "connection_error", "rate_limit", "server_error"])
def test_retryable_error_types(error_type):
    """ProviderError with a retryable error_type → True."""
    error = ProviderError("temporary", error_type=error_type)

    assert is_retryable_provider_error(error) is True


@pytest.mark.parametrize(
    "error_type",
    [
        "bad_request",
        "auth_error",
        "not_found",
        "conflict",
        "too_early",
        "http_error",
        "unknown_error",
    ],
)
def test_non_retryable_error_types(error_type):
    """ProviderError with a non-retryable error_type → False."""
    error = ProviderError("permanent", error_type=error_type)

    assert is_retryable_provider_error(error) is False


# =============================================================================
# Edge cases
# =============================================================================


def test_unknown_provider_error_is_not_retryable():
    """ProviderError with no status_code AND no error_type → False."""
    error = ProviderError("unknown")

    assert is_retryable_provider_error(error) is False


def test_error_type_wins_over_status_code():
    """retryable error_type ('rate_limit') wins over non-retryable status_code (400)."""
    error = ProviderError("rate limited", status_code=400, error_type="rate_limit")

    assert is_retryable_provider_error(error) is True


def test_status_code_500_without_error_type_is_retryable():
    """ProviderError with status_code=500 but no error_type → True.

    Simulates bailian _check_response_error path after the fix.
    """
    error = ProviderError("server error", status_code=500)

    assert is_retryable_provider_error(error) is True


def test_status_code_429_without_error_type_is_retryable():
    """ProviderError with status_code=429 but no error_type → True.

    Simulates bailian rate-limit after fix.
    """
    error = ProviderError("rate limited", status_code=429)

    assert is_retryable_provider_error(error) is True


def test_status_code_400_without_error_type_is_not_retryable():
    """ProviderError with status_code=400 but no error_type → False.

    Simulates bailian bad request after fix.
    """
    error = ProviderError("bad request", status_code=400)

    assert is_retryable_provider_error(error) is False


def test_non_provider_error_is_not_retryable():
    """ValueError (not a ProviderError) → False.

    Non-ProviderError exceptions should never reach this function in
    practice (all SDK errors are wrapped), but the function must handle
    this case safely.
    """
    assert is_retryable_provider_error(ValueError("something broke")) is False


# =============================================================================
# failure_record
# =============================================================================


def test_failure_record_includes_standardized_error_facts():
    provider = SimpleNamespace(
        provider_name="openai",
        model_name="gpt-4o-mini",
        config=SimpleNamespace(
            base_url="http://127.0.0.1:9/v1",
        ),
    )
    error = ProviderError("timeout", status_code=None, error_type="timeout")

    record = ResilientProvider._failure_record(provider, 1, error)

    assert record["attempt_index"] == 1
    assert record["provider"] == "openai"
    assert record["model"] == "gpt-4o-mini"
    assert record["base_url"] == "http://127.0.0.1:9/v1"
    assert record["status_code"] is None
    assert record["error_type"] == "timeout"
