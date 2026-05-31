from types import SimpleNamespace

import pytest

from chak.exceptions import ProviderError
from chak.providers.llm.resilient import ResilientProvider, is_retryable_provider_error

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("status_code", [408, 409, 425, 429, 500, 502, 503, 504])
def test_retryable_status_codes(status_code):
    error = ProviderError("temporary", status_code=status_code)

    assert is_retryable_provider_error(error) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_non_retryable_client_status_codes(status_code):
    error = ProviderError("config error", status_code=status_code)

    assert is_retryable_provider_error(error) is False


@pytest.mark.parametrize("error_type", ["timeout", "connection_error", "rate_limit", "server_error"])
def test_retryable_error_types(error_type):
    error = ProviderError("temporary", error_type=error_type)

    assert is_retryable_provider_error(error) is True


def test_unknown_provider_error_is_not_retryable():
    error = ProviderError("unknown")

    assert is_retryable_provider_error(error) is False


def test_failure_record_includes_standardized_error_facts():
    provider = SimpleNamespace(
        config=SimpleNamespace(
            provider_name="openai",
            model="gpt-4o-mini",
            base_url="http://127.0.0.1:9/v1",
        )
    )
    error = ProviderError("timeout", status_code=None, error_type="timeout")

    record = ResilientProvider._failure_record(provider, 1, error)

    assert record["attempt_index"] == 1
    assert record["provider"] == "openai"
    assert record["model"] == "gpt-4o-mini"
    assert record["base_url"] == "http://127.0.0.1:9/v1"
    assert record["status_code"] is None
    assert record["error_type"] == "timeout"
