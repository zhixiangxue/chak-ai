import pytest

from chak.exceptions import ProviderError
from chak.message import AIMessage, HumanMessage, UnifiedStreamChunk
from chak.providers.llm.base import BaseMessageConverter, BaseProviderConfig, Provider

pytestmark = pytest.mark.unit


class EchoConverter(BaseMessageConverter):
    def to_provider_format(self, messages):
        return messages

    def from_provider_response(self, response):
        return response

    def from_provider_chunk(self, chunk):
        return chunk


class CompleteErrorProvider(Provider):
    def _initialize_client(self):
        pass

    def _send_complete(self, messages, **kwargs):
        raise RuntimeError("complete failed")

    def _send_stream(self, messages, **kwargs):
        yield UnifiedStreamChunk(content="unused")


class StreamErrorProvider(Provider):
    def _initialize_client(self):
        pass

    def _send_complete(self, messages, **kwargs):
        return AIMessage(content="ok")

    def _send_stream(self, messages, **kwargs):
        yield UnifiedStreamChunk(content="first")
        raise RuntimeError("stream failed")


def make_config():
    return BaseProviderConfig(
        api_key="test-key",
        model="test-model",
        base_url="https://example.test/v1",
        provider_name="test-provider",
    )


def test_send_wraps_nonstream_error_with_provider_facts():
    provider = CompleteErrorProvider(make_config(), EchoConverter())

    with pytest.raises(ProviderError) as exc_info:
        provider.send([HumanMessage(content="hello")])

    error = exc_info.value
    assert error.provider == "test-provider"
    assert error.model == "test-model"
    assert error.base_url == "https://example.test/v1"
    assert isinstance(error.raw_error, RuntimeError)


def test_send_wraps_stream_iteration_error_with_provider_facts():
    provider = StreamErrorProvider(make_config(), EchoConverter())

    stream = provider.send([HumanMessage(content="hello")], stream=True)
    first = next(stream)

    assert first.content == "first"
    with pytest.raises(ProviderError) as exc_info:
        next(stream)

    error = exc_info.value
    assert error.provider == "test-provider"
    assert error.model == "test-model"
    assert error.base_url == "https://example.test/v1"
    assert isinstance(error.raw_error, RuntimeError)


def test_send_keeps_successful_stream_chunks_unchanged():
    provider = StreamErrorProvider(make_config(), EchoConverter())
    stream = provider._wrap_stream_errors(iter([UnifiedStreamChunk(content="ok")]))

    assert next(stream).content == "ok"
