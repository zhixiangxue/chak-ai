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


class RecordingProvider(Provider):
    """Captures the kwargs that reach ``_send_complete`` / ``_send_stream``."""

    def _initialize_client(self):
        self.last_kwargs = None

    def _send_complete(self, messages, **kwargs):
        self.last_kwargs = kwargs
        return AIMessage(content="ok")

    def _send_stream(self, messages, **kwargs):
        self.last_kwargs = kwargs
        yield UnifiedStreamChunk(content="ok")


def test_construction_time_extra_params_are_forwarded():
    # Extra fields (extra='allow') act as default request params.
    config = BaseProviderConfig(
        api_key="k", model="m", temperature=0.11, top_p=0.22
    )
    provider = RecordingProvider(config, EchoConverter())

    provider.send([HumanMessage(content="hi")])

    assert provider.last_kwargs["temperature"] == 0.11
    assert provider.last_kwargs["top_p"] == 0.22


def test_per_call_kwargs_override_construction_defaults():
    config = BaseProviderConfig(
        api_key="k", model="m", temperature=0.11, top_p=0.22
    )
    provider = RecordingProvider(config, EchoConverter())

    provider.send([HumanMessage(content="hi")], temperature=0.9)

    # Per-call wins; untouched default stays applied.
    assert provider.last_kwargs["temperature"] == 0.9
    assert provider.last_kwargs["top_p"] == 0.22


def test_declared_config_fields_are_not_forwarded():
    # Declared fields (api_key/model/base_url/...) must never leak onto the wire.
    config = BaseProviderConfig(
        api_key="k", model="m", base_url="https://x/v1", temperature=0.5
    )
    provider = RecordingProvider(config, EchoConverter())

    provider.send([HumanMessage(content="hi")])

    assert provider.last_kwargs == {"temperature": 0.5}


def test_default_params_forwarded_in_streaming():
    config = BaseProviderConfig(api_key="k", model="m", temperature=0.3)
    provider = RecordingProvider(config, EchoConverter())

    list(provider.send([HumanMessage(content="hi")], stream=True))

    assert provider.last_kwargs["temperature"] == 0.3
