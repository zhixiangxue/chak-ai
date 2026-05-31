import pytest

from chak.exceptions import ProviderError

pytestmark = pytest.mark.unit


class StatusCodeError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class Response:
    def __init__(self, status_code):
        self.status_code = status_code


class ResponseError(Exception):
    def __init__(self, status_code):
        super().__init__(f"response {status_code}")
        self.response = Response(status_code)


class TimeoutLikeError(Exception):
    pass


TimeoutLikeError.__name__ = "ReadTimeout"


@pytest.mark.parametrize(
    "status_code,error_type",
    [
        (400, "bad_request"),
        (401, "auth_error"),
        (403, "auth_error"),
        (404, "not_found"),
        (408, "timeout"),
        (409, "conflict"),
        (425, "too_early"),
        (429, "rate_limit"),
        (500, "server_error"),
        (503, "server_error"),
    ],
)
def test_provider_error_classifies_status_code(status_code, error_type):
    raw_error = StatusCodeError(status_code)

    error = ProviderError.from_exception(
        raw_error,
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
    )

    assert error.provider == "openai"
    assert error.model == "gpt-4o-mini"
    assert error.base_url == "https://api.openai.com/v1"
    assert error.status_code == status_code
    assert error.error_type == error_type
    assert error.raw_error is raw_error


def test_provider_error_extracts_response_status_code():
    error = ProviderError.from_exception(ResponseError(429))

    assert error.status_code == 429
    assert error.error_type == "rate_limit"


def test_provider_error_classifies_timeout_without_status_code():
    raw_error = TimeoutLikeError("request timed out")

    error = ProviderError.from_exception(raw_error)

    assert error.status_code is None
    assert error.error_type == "timeout"
    assert error.raw_error is raw_error


def test_provider_error_reuses_existing_instance_and_fills_missing_facts():
    original = ProviderError("wrapped", status_code=503, error_type="server_error")

    error = ProviderError.from_exception(
        original,
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
    )

    assert error is original
    assert error.provider == "deepseek"
    assert error.model == "deepseek-chat"
    assert error.base_url == "https://api.deepseek.com"
    assert error.status_code == 503
    assert error.error_type == "server_error"
