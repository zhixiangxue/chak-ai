import pytest

from chak.message import ConversationCompleteEvent, HumanMessage, MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent
from chak.providers.llm.bailian import BailianMessageConverter, BailianProvider
from chak.tools.manager import ToolManager

pytestmark = pytest.mark.unit


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def make_stream_chunk(content, *, reasoning_content=None, tool_calls=None, finish_reason=None):
    message = AttrDict({"content": content})
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    return AttrDict({
        "output": AttrDict({
            "choices": [
                AttrDict({
                    "message": message,
                    "finish_reason": finish_reason,
                })
            ]
        })
    })


def test_stream_chunk_list_content_is_normalized_to_empty_string():
    converter = BailianMessageConverter()
    tool_calls = [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]

    chunk = converter.from_provider_chunk(
        make_stream_chunk(content=[], tool_calls=tool_calls, finish_reason=None)
    )

    assert chunk.content == ""
    assert len(chunk.tool_calls_delta) == 1
    assert chunk.tool_calls_delta[0].id == "call_1"
    assert chunk.tool_calls_delta[0].function_name == "lookup"


def test_stream_chunk_list_reasoning_content_is_normalized_to_none():
    converter = BailianMessageConverter()

    chunk = converter.from_provider_chunk(
        make_stream_chunk(content="", reasoning_content=[])
    )

    assert chunk.content == ""
    assert chunk.reasoning_content is None


def test_stream_chunk_text_parts_content_is_joined_on_stop_frame():
    converter = BailianMessageConverter()

    chunk = converter.from_provider_chunk(
        make_stream_chunk(
            content=[{"text": "Final "}, {"text": "answer"}],
            finish_reason="stop",
        )
    )

    assert chunk.content == "Final answer"
    assert isinstance(chunk.content, str)
    assert chunk.finish_reason == "stop"
    assert chunk.is_final is True


class FakeProvider:
    def __init__(self):
        self.converter = BailianMessageConverter()
        self.calls = 0

    def send(self, *, messages, stream=False, tools=None):
        assert stream is True
        self.calls += 1
        if self.calls == 1:
            return iter([
                make_stream_chunk(
                    content=[],
                    tool_calls=[{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }],
                    finish_reason="tool_calls",
                )
            ])
        return iter([
            make_stream_chunk(
                content=[{"text": "Final answer after tool"}],
                finish_reason="stop",
            )
        ])


class FakeTool:
    name = "lookup"

    def to_openai_tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Lookup test data.",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def call(self, arguments, executor=None):
        return "tool result"


@pytest.mark.asyncio
async def test_event_tool_loop_emits_final_stop_frame_text_from_list_content():
    manager = ToolManager([FakeTool()], max_iterations=3)
    provider = FakeProvider()

    events = [
        event
        async for event in manager.execute_loop_with_events(
            provider=provider,
            messages=[HumanMessage(content="use the tool")],
            model_uri="bailian/qwen3.8-max",
        )
    ]

    text_chunks = [
        event
        for event in events
        if isinstance(event, MessageChunk) and event.content
    ]

    assert any(isinstance(event, ToolCallStartEvent) for event in events)
    assert any(isinstance(event, ToolCallSuccessEvent) for event in events)
    assert [chunk.content for chunk in text_chunks] == ["Final answer after tool"]
    assert isinstance(text_chunks[0].content, str)
    assert any(isinstance(event, ConversationCompleteEvent) for event in events)


@pytest.mark.parametrize(
    "model, expected",
    [
        # Vision series must route to MultiModalConversation API
        ("qwen-vl-max", True),
        ("qwen-vl-plus", True),
        ("qwen2-vl-72b-instruct", True),
        ("qwen2.5-vl-72b-instruct", True),
        ("qwen3-vl-plus", True),
        ("qwen3-vl-max", True),
        # Dotted multimodal text models
        ("qwen3.6-plus", True),
        # Text-only models stay on the Generation API
        ("qwen-plus", False),
        ("qwen-max", False),
        ("qwen-turbo", False),
    ],
)
def test_is_multimodal_model_routes_vision_series(model, expected):
    assert BailianProvider._is_multimodal_model(model) is expected
