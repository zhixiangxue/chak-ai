"""Unit tests for the native google-genai provider converter.

Offline only: exercises message conversion, thought_signature caching,
usage normalization, and OpenAI-style kwarg translation without any
network access.
"""
import json
from types import SimpleNamespace

import pytest

from chak.message import (
    AIMessage,
    ChatCompletionMessageToolCall,
    Function,
    HumanMessage,
    SystemMessage,
    ToolCallDelta,
    ToolMessage,
)
from chak.providers.llm.google import (
    GoogleConfig,
    GoogleMessageConverter,
    GoogleProvider,
)

try:
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    pytest.skip("google-genai not installed", allow_module_level=True)


TINY_RED_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFElEQVR4nGP4"
    "z8DwnxjMMKqQvgoBksPHOas6/LEAAAAASUVORK5CYII="
)


@pytest.fixture
def converter():
    return GoogleMessageConverter()


def _fake_response(parts, usage=None, finish=None, model="gemini-test"):
    """Build a duck-typed GenerateContentResponse."""
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=parts), finish_reason=finish
    )
    return SimpleNamespace(
        candidates=[candidate], usage_metadata=usage, model=model
    )


# ─── to_provider_format ─────────────────────────────────────────────────

def test_basic_conversion_and_system_instruction(converter):
    out = converter.to_provider_format([
        SystemMessage(content="You are helpful."),
        HumanMessage(content="Hi"),
        AIMessage(content="Hello!"),
    ])
    assert out["system_instruction"] == "You are helpful."
    contents = out["contents"]
    assert [c.role for c in contents] == ["user", "model"]
    assert contents[0].parts[0].text == "Hi"
    assert contents[1].parts[0].text == "Hello!"


def test_consecutive_same_role_messages_merge(converter):
    """Gemini requires alternating turns; consecutive user messages
    (e.g. tool results after a user turn) must merge into one Content."""
    out = converter.to_provider_format([
        HumanMessage(content="a"),
        HumanMessage(content="b"),
    ])
    assert len(out["contents"]) == 1
    assert len(out["contents"][0].parts) == 2


def test_tool_roundtrip_injects_signature_and_function_name(converter):
    """The converter must echo the cached thought_signature on the
    function_call part and resolve the function NAME for the response."""
    # Simulate a model response carrying a signature
    converter._thought_signatures["call_1"] = "c2lnbmF0dXJl"
    converter._call_names["call_1"] = "add_numbers"

    assistant = AIMessage(
        content="",
        tool_calls=[ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function=Function(name="add_numbers", arguments='{"a": 7, "b": 5}'),
        )],
    )
    tool = ToolMessage(content="12", tool_call_id="call_1")
    out = converter.to_provider_format([HumanMessage(content="add"), assistant, tool])

    model_turn = out["contents"][1]
    fc_part = model_turn.parts[-1]
    assert fc_part.function_call.name == "add_numbers"
    assert fc_part.function_call.args == {"a": 7, "b": 5}
    # SDK decodes the base64 signature back to raw bytes on read
    assert fc_part.thought_signature == b"signature"

    # function_response travels in a user turn and uses the NAME, not the id
    resp_turn = out["contents"][2]
    assert resp_turn.role == "user"
    fr_part = resp_turn.parts[0]
    assert fr_part.function_response.name == "add_numbers"
    assert fr_part.function_response.response == {"result": "12"}


def test_multimodal_data_uri_image(converter):
    out = converter.to_provider_format([
        HumanMessage(content=[
            {"type": "text", "text": "What color?"},
            {"type": "image_url", "image_url": {"url": TINY_RED_PNG}},
        ]),
    ])
    parts = out["contents"][0].parts
    assert parts[0].text == "What color?"
    assert parts[1].inline_data.mime_type == "image/png"
    assert parts[1].inline_data.data  # decoded bytes present


def test_empty_content_placeholder(converter):
    """Empty assistant turns must not produce an empty Content."""
    out = converter.to_provider_format([
        HumanMessage(content="hi"),
        AIMessage(content=""),
    ])
    assert out["contents"][1].parts[0].text == "(no content)"


# ─── from_provider_response ─────────────────────────────────────────────

def test_response_splits_thought_text_and_tool_calls(converter):
    parts = [
        genai_types.Part(text="thinking about it", thought=True),
        genai_types.Part(function_call=genai_types.FunctionCall(
            id="call_1", name="add_numbers", args={"a": 7, "b": 5}
        ), thought_signature="c2lnbmF0dXJl"),
    ]
    usage = SimpleNamespace(
        prompt_token_count=100, candidates_token_count=20,
        cached_content_token_count=30, thoughts_token_count=5,
    )
    msg = converter.from_provider_response(
        _fake_response(parts, usage=usage, finish="STOP")
    )
    assert msg.content == ""
    assert msg.reasoning_content == "thinking about it"
    assert msg.tool_calls and msg.tool_calls[0].id == "call_1"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"a": 7, "b": 5}
    # Signature cached for the echo round (normalized to base64 str)
    assert converter._thought_signatures["call_1"] == "c2lnbmF0dXJl"
    assert converter._call_names["call_1"] == "add_numbers"
    # Disjoint-bucket usage contract: cache stripped from prompt,
    # thoughts folded into completion
    assert msg.metadata.usage.prompt_tokens == 70
    assert msg.metadata.usage.completion_tokens == 25
    assert msg.metadata.usage.cache_read_input_tokens == 30
    assert msg.metadata.usage.total_tokens == 70 + 25 + 30
    assert msg.metadata.provider == "google"


def test_response_synthesizes_missing_call_id(converter):
    parts = [genai_types.Part(function_call=genai_types.FunctionCall(
        name="add_numbers", args={}
    ))]
    msg = converter.from_provider_response(_fake_response(parts))
    assert msg.tool_calls[0].id.startswith("call_")


# ─── from_provider_chunk (streaming) ────────────────────────────────────

def test_chunk_text_and_finish(converter):
    chunk = _fake_response([genai_types.Part(text="Hel")])
    unified = converter.from_provider_chunk(chunk)
    assert unified.content == "Hel"
    assert not unified.is_final

    final = _fake_response([], finish="STOP")
    unified_final = converter.from_provider_chunk(final)
    assert unified_final.is_final
    # Normalized to OpenAI-style lowercase finish reasons
    assert unified_final.finish_reason == "stop"


def test_chunk_finish_normalized_to_tool_calls(converter):
    """Gemini reports STOP for tool-calling turns; the converter must
    synthesize 'tool_calls' so the manager's loop keeps running."""
    converter._reset_stream_state()
    call_chunk = _fake_response([genai_types.Part(
        function_call=genai_types.FunctionCall(id="c1", name="add", args={})
    )])
    converter.from_provider_chunk(call_chunk)
    # The finish reason arrives in a later chunk without the call part
    final = _fake_response([], finish="STOP")
    unified = converter.from_provider_chunk(final)
    assert unified.finish_reason == "tool_calls"


def test_chunk_function_call_produces_zero_based_deltas(converter):
    converter._reset_stream_state()
    chunk1 = _fake_response([genai_types.Part(
        function_call=genai_types.FunctionCall(id="c1", name="add", args={"a": 1}),
        thought_signature="c2lnMQ==",
    )])
    chunk2 = _fake_response([genai_types.Part(
        function_call=genai_types.FunctionCall(id="c2", name="sub", args={"b": 2}),
    )])
    d1 = converter.from_provider_chunk(chunk1).tool_calls_delta
    d2 = converter.from_provider_chunk(chunk2).tool_calls_delta
    assert isinstance(d1[0], ToolCallDelta)
    assert d1[0].index == 0 and d1[0].function_name == "add"
    assert d2[0].index == 1 and d2[0].function_name == "sub"
    assert json.loads(d1[0].function_arguments) == {"a": 1}
    # Signatures cached for the later echo round (normalized to base64 str)
    assert converter._thought_signatures == {"c1": "c2lnMQ=="}


# ─── GoogleProvider param translation ───────────────────────────────────

@pytest.fixture
def provider():
    return GoogleProvider(
        config=GoogleConfig(api_key="test-key", model="gemini-test"),
        converter=GoogleMessageConverter(),
    )


def test_convert_tool_choice_mapping(provider):
    auto = provider._convert_tool_choice("auto")
    assert str(auto.function_calling_config.mode).endswith("AUTO")
    required = provider._convert_tool_choice("required")
    assert str(required.function_calling_config.mode).endswith("ANY")
    none = provider._convert_tool_choice("none")
    assert str(none.function_calling_config.mode).endswith("NONE")
    named = provider._convert_tool_choice(
        {"type": "function", "function": {"name": "add_numbers"}}
    )
    assert str(named.function_calling_config.mode).endswith("ANY")
    assert named.function_calling_config.allowed_function_names == ["add_numbers"]


def test_build_config_translates_openai_kwargs(provider):
    tools = [{
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    config = provider._build_config(
        {"system_instruction": "sys", "contents": []},
        tools=tools,
        tool_choice="required",
        response_format={"type": "json_schema", "json_schema": {
            "name": "City", "schema": {"type": "object"}
        }},
        max_tokens=123,
        temperature=0.2,
        stream_options={"include_usage": True},  # must be dropped silently
        timeout=60,                               # must be dropped silently
    )
    assert config.system_instruction == "sys"
    decl = config.tools[0].function_declarations[0]
    assert decl.name == "add_numbers"
    assert str(config.tool_config.function_calling_config.mode).endswith("ANY")
    assert config.response_mime_type == "application/json"
    assert config.response_schema == {"type": "object"}
    assert config.max_output_tokens == 123
    assert config.temperature == 0.2


def test_build_config_reasoning_budget(provider):
    config = provider._build_config(
        {"system_instruction": None, "contents": []},
        reasoning={"effort": "high"},
    )
    assert config.thinking_config.thinking_budget == 16000
