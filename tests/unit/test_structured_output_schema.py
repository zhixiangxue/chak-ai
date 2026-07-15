from typing import Dict, List

import pytest
from pydantic import BaseModel

import chak.conversation as conversation_module
from chak import Conversation
from chak.message import AIMessage, ChatCompletionMessageToolCall, Function

pytestmark = pytest.mark.unit


class Program(BaseModel):
    name: str


class Requirement(BaseModel):
    """An eligibility requirement."""
    key: str
    title: str


class CapturingProvider:
    def __init__(self):
        self.calls = []

    def send(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        tool_name = kwargs["tools"][0]["function"]["name"]
        return AIMessage(
            content="",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name=tool_name,
                        arguments='{"items": [{"name": "Alpha"}, {"name": "Beta"}]}',
                    ),
                )
            ],
        )


@pytest.fixture
def capturing_provider(monkeypatch):
    provider = CapturingProvider()

    def create_provider(provider_name, config_dict, category):
        return provider

    monkeypatch.setattr(conversation_module, "create_provider", create_provider)
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("returns_type", [list[Program], List[Program]])
async def test_structured_output_list_return_uses_object_schema_and_unwraps(capturing_provider, returns_type):
    conv = Conversation("openai/gpt-4o-mini", api_key="test-key")

    result = await conv.asend("Return two programs.", returns=returns_type)

    assert result == [Program(name="Alpha"), Program(name="Beta")]

    schema = capturing_provider.calls[0]["tools"][0]["function"]["parameters"]
    assert schema["type"] == "object"
    assert schema["required"] == ["items"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["items"]["type"] == "array"


# --------------------------------------------------------------------------- #
# Tool-name emission (regression: previously all structured outputs advertised
# themselves as generic ``ExtractedData`` because the RootModel probe used
# ``hasattr(model, '__pydantic_generic_metadata__')``, which is True for every
# Pydantic v2 BaseModel. The fix uses ``issubclass(model, RootModel)`` instead
# so that plain BaseModel subclasses keep their own class name and docstring.)
# --------------------------------------------------------------------------- #


class _ScalarProvider:
    """Returns a valid tool_call payload for a plain BaseModel schema."""

    def __init__(self):
        self.calls = []

    def send(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        tool_name = kwargs["tools"][0]["function"]["name"]
        return AIMessage(
            content="",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name=tool_name,
                        arguments='{"key": "k", "title": "t"}',
                    ),
                )
            ],
        )


@pytest.fixture
def scalar_provider(monkeypatch):
    provider = _ScalarProvider()
    monkeypatch.setattr(
        conversation_module,
        "create_provider",
        lambda provider_name, config_dict, category: provider,
    )
    return provider


@pytest.mark.asyncio
async def test_basemodel_returns_uses_class_name_as_tool_name(scalar_provider):
    conv = Conversation("openai/gpt-4o-mini", api_key="test-key")

    result = await conv.asend("Give me a requirement.", returns=Requirement)

    assert result == Requirement(key="k", title="t")

    fn = scalar_provider.calls[0]["tools"][0]["function"]
    # Regression: must be the user-defined class name, not "ExtractedData".
    assert fn["name"] == "Requirement"
    # Docstring should propagate so the model has semantic context.
    assert "eligibility requirement" in fn["description"].lower()
    # And tool_choice must force this specific function.
    assert scalar_provider.calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "Requirement"},
    }


@pytest.mark.asyncio
async def test_list_returns_uses_generic_extracted_data_name(capturing_provider):
    conv = Conversation("openai/gpt-4o-mini", api_key="test-key")

    await conv.asend("Return two programs.", returns=List[Program])

    fn_name = capturing_provider.calls[0]["tools"][0]["function"]["name"]
    # List[...] is wrapped via ``create_model('ExtractedData', ...)``, which
    # is a plain BaseModel (not a RootModel). The generated name comes from
    # the wrapper's own ``__name__``, so it should be "ExtractedData".
    assert fn_name == "ExtractedData"


class _DictProvider:
    """Returns a Dict[str, Program]-shaped RootModel payload."""

    def __init__(self):
        self.calls = []

    def send(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        tool_name = kwargs["tools"][0]["function"]["name"]
        return AIMessage(
            content="",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name=tool_name,
                        arguments='{"alpha": {"name": "A"}}',
                    ),
                )
            ],
        )


@pytest.fixture
def dict_provider(monkeypatch):
    provider = _DictProvider()
    monkeypatch.setattr(
        conversation_module,
        "create_provider",
        lambda provider_name, config_dict, category: provider,
    )
    return provider


@pytest.mark.asyncio
async def test_dict_returns_uses_generic_extracted_data_name(dict_provider):
    conv = Conversation("openai/gpt-4o-mini", api_key="test-key")

    await conv.asend("Return a program map.", returns=Dict[str, Program])

    fn_name = dict_provider.calls[0]["tools"][0]["function"]["name"]
    # Dict[...] is wrapped as ``RootModel[...]``; the RootModel branch of
    # ``_generate_tool_schema_from_model`` must engage and produce the
    # generic "ExtractedData" name (the RootModel's own __name__ would be
    # unstable across Python versions).
    assert fn_name == "ExtractedData"


# --------------------------------------------------------------------------- #
# Envelope-unwrap fallback
#
# Observed in production with deepseek/deepseek-v4-pro on complex nested
# schemas under long contexts: the provider sporadically wraps the tool
# payload inside an envelope, e.g. ``{"requirement": {...}}`` or
# ``{"data": {...}, "file": "unknown"}``. Chak recovers by trying each
# top-level dict value once direct validation fails.
# --------------------------------------------------------------------------- #


class _WrappingProvider:
    """Return a payload whose top level is an envelope over the real fields."""

    def __init__(self, arguments_json: str):
        self._arguments = arguments_json
        self.calls = []

    def send(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        tool_name = kwargs["tools"][0]["function"]["name"]
        return AIMessage(
            content="",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(name=tool_name, arguments=self._arguments),
                )
            ],
        )


def _install(monkeypatch, provider):
    monkeypatch.setattr(
        conversation_module,
        "create_provider",
        lambda provider_name, config_dict, category: provider,
    )


@pytest.mark.asyncio
async def test_envelope_unwrap_single_key_matching_schema_title(monkeypatch):
    # Mirrors the field observation: DeepSeek returned
    #   {"requirement": {"key": ..., "title": ...}}
    # even though the tool function name was "ExtractedData". The wrap key
    # came from the Pydantic JSON Schema ``title``, not the function name.
    provider = _WrappingProvider('{"requirement": {"key": "k", "title": "t"}}')
    _install(monkeypatch, provider)

    # chak uses loguru with a stdout sink attached at import time. capsys and
    # caplog don't observe it, so we attach a purpose-built sink that appends
    # to a list, capturing exactly the warnings emitted during this test.
    from chak.utils.logger import logger as chak_logger

    captured_logs: list[str] = []
    handler_id = chak_logger.add(lambda msg: captured_logs.append(str(msg)), level="WARNING")
    try:
        conv = Conversation("openai/gpt-4o-mini", api_key="test-key")
        result = await conv.asend("Give me a requirement.", returns=Requirement)
    finally:
        chak_logger.remove(handler_id)

    assert result == Requirement(key="k", title="t")
    # Exactly one send: the fallback must not trigger a retry when it
    # recovers on the first attempt.
    assert len(provider.calls) == 1
    # A warning must surface so operators see the model is misbehaving.
    assert any("envelope wrap" in line for line in captured_logs), (
        f"expected an 'envelope wrap' warning; captured: {captured_logs!r}"
    )


@pytest.mark.asyncio
async def test_envelope_unwrap_metadata_style_multi_key(monkeypatch):
    # Second observed shape: ``{"data": {...}, "file": "unknown"}`` -- a
    # multi-key envelope where only one value is a dict. The unwrap loop
    # must still find the payload.
    provider = _WrappingProvider(
        '{"data": {"key": "k", "title": "t"}, "file": "unknown"}'
    )
    _install(monkeypatch, provider)

    conv = Conversation("openai/gpt-4o-mini", api_key="test-key")
    result = await conv.asend("Give me a requirement.", returns=Requirement)

    assert result == Requirement(key="k", title="t")


@pytest.mark.asyncio
async def test_envelope_unwrap_does_not_swallow_genuine_shape_errors(monkeypatch):
    # If neither the direct payload nor any dict-valued top-level entry
    # validates, the loop must give up and re-raise so chak's built-in
    # retry can feed the error back to the LLM. We assert this by using a
    # payload where every branch is genuinely wrong; the provider is only
    # invoked once here (the retry itself is tested elsewhere), so we just
    # check that ``asend`` swallows the failure into ``None`` (per chak's
    # documented contract for structured output).
    provider = _WrappingProvider('{"unrelated": {"foo": 1}, "note": "nope"}')
    _install(monkeypatch, provider)

    conv = Conversation("openai/gpt-4o-mini", api_key="test-key")
    result = await conv.asend("Give me a requirement.", returns=Requirement)

    # chak.asend swallows structured-output failures into None (see
    # conversation.py's except handler around _asend_with_structured_output).
    assert result is None
    # Retries happen: 3 attempts by default. All 3 hit the same shape,
    # so the provider is called 3 times.
    assert len(provider.calls) == 3
