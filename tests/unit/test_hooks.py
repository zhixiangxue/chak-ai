import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel

import chak.conversation as mod
from chak.message import AIMessage, MessageChunk

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers for mode-specific mocks
# ---------------------------------------------------------------------------

async def _mock_stream_impl(self, message, attachments, tool_executor, **kwargs):
    """Mock _asend_stream_impl: append user + AI message, yield one chunk."""
    self.messages.append(message)
    if attachments:
        self.attachments.extend(attachments)
    self.messages.append(AIMessage(content="mock stream"))
    yield MessageChunk(content="mock stream")


async def _mock_event_impl(self, message, attachments, tool_executor, **kwargs):
    """Mock _asend_with_events_impl: append user + AI, yield one event."""
    self.messages.append(message)
    if attachments:
        self.attachments.extend(attachments)
    self.messages.append(AIMessage(content="mock event"))
    yield MessageChunk(content="mock event")


async def _mock_tools_impl(self, messages, **kwargs):
    """Mock _asend_nonstream_with_tools: append AI response and return it."""
    result = AIMessage(content="mock tools")
    self.messages.append(result)
    return result


async def _mock_structured_impl(self, message, attachments, returns, **kwargs):
    """Mock _asend_with_structured_output: return a dummy model instance."""
    return returns(name="mock_structured")


async def _mock_extraction_loop(self, messages, returns, **kwargs):
    """Mock _run_extraction_loop: return a dummy model instance."""
    return returns(name="mock_extracted")


class DummyModel(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

def test_register_single_callback():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    cb = MagicMock()
    conv.hook.before_send(cb)
    assert len(conv.hook.before_send._callbacks) == 1
    assert conv.hook.before_send._callbacks[0] is cb


def test_register_list_of_callbacks():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    cb1, cb2 = MagicMock(), MagicMock()
    conv.hook.before_send([cb1, cb2])
    assert len(conv.hook.before_send._callbacks) == 2


def test_register_multiple_times_appends():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    cb1, cb2 = MagicMock(), MagicMock()
    conv.hook.before_send(cb1)
    conv.hook.before_send(cb2)
    assert len(conv.hook.before_send._callbacks) == 2


def test_register_non_callable_raises():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    with pytest.raises(TypeError):
        conv.hook.before_send("not_a_callable")


# ---------------------------------------------------------------------------
# before_send tests (non-streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_before_send_called_once():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    before_calls = []

    async def before(conv_, request, **kw):
        before_calls.append(request.content)

    conv.hook.before_send(before)
    await conv.asend("Hello")

    assert before_calls == ["Hello"]


@pytest.mark.asyncio
async def test_before_send_raise_aborts():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")

    async def before(conv_, request, **kw):
        raise RuntimeError("budget exceeded")

    conv.hook.before_send(before)

    with pytest.raises(RuntimeError, match="budget exceeded"):
        await conv.asend("Hello")


@pytest.mark.asyncio
async def test_before_send_multiple_executed_in_order():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    order = []

    async def first(conv_, request, **kw):
        order.append(1)

    async def second(conv_, request, **kw):
        order.append(2)

    conv.hook.before_send([first, second])
    await conv.asend("Hello")

    assert order == [1, 2]


# ---------------------------------------------------------------------------
# after_send tests (non-streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_after_send_called_with_request():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    after_calls = []

    async def after(conv_, request, **kw):
        after_calls.append(request.content)

    conv.hook.after_send(after)
    await conv.asend("Hello")

    assert after_calls == ["Hello"]


@pytest.mark.asyncio
async def test_after_send_has_response_in_conv_messages():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    after_responses = []

    async def after(conv_, request, **kw):
        after_responses.append(conv_.messages[-1].content)

    conv.hook.after_send(after)
    await conv.asend("Hello")

    assert after_responses == ["Hello from mock"]


@pytest.mark.asyncio
async def test_after_send_raise_propagates():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")

    async def after(conv_, request, **kw):
        raise RuntimeError("metrics failure")

    conv.hook.after_send(after)

    with pytest.raises(RuntimeError, match="metrics failure"):
        await conv.asend("Hello")


@pytest.mark.asyncio
async def test_empty_hooks_no_effect():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    result = await conv.asend("Hello")
    assert result is not None


# ---------------------------------------------------------------------------
# send_kwargs tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_before_send_receives_send_kwargs():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    received_kw = []

    async def before(conv_, request, **kw):
        received_kw.append(kw)

    conv.hook.before_send(before)
    await conv.asend("Hello", timeout=42)

    assert received_kw[0]["timeout"] == 42


@pytest.mark.asyncio
async def test_after_send_receives_send_kwargs():
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    received_kw = []

    async def after(conv_, request, **kw):
        received_kw.append(kw)

    conv.hook.after_send(after)
    await conv.asend("Hello", timeout=77)

    assert received_kw[0]["timeout"] == 77


# ---------------------------------------------------------------------------
# Sync send() hook tests
# ---------------------------------------------------------------------------

def test_sync_send_before_and_after_called():
    """sync send(): before_send + after_send both fire."""
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    trace = []

    async def before(conv_, request, **kw):
        trace.append(("before", request.content))

    async def after(conv_, request, **kw):
        trace.append(("after", request.content))

    conv.hook.before_send(before)
    conv.hook.after_send(after)
    conv.send("SyncHello")

    assert trace == [("before", "SyncHello"), ("after", "SyncHello")]


def test_sync_send_stream_before_and_after_called(monkeypatch):
    """sync send(stream=True): before_send fires; after_send fires after iteration."""
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    trace = []

    async def before(conv_, request, **kw):
        trace.append(("before", request.content))

    async def after(conv_, request, **kw):
        trace.append(("after", request.content))

    conv.hook.before_send(before)
    conv.hook.after_send(after)

    # Mock _send_stream to yield one chunk and append an AI response
    def _mock_sync_stream(self, messages, **kw):
        self.messages.append(AIMessage(content="mock sync stream"))
        yield MessageChunk(content="mock sync stream")

    monkeypatch.setattr(mod.Conversation, "_send_stream", _mock_sync_stream)

    # Consume stream to trigger after_send in finally
    chunks = list(conv.send("SyncStreamHello", stream=True))
    assert len(chunks) == 1
    assert chunks[0].content == "mock sync stream"

    assert trace == [("before", "SyncStreamHello"), ("after", "SyncStreamHello")]


# ---------------------------------------------------------------------------
# Async stream / event hook tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_stream_before_and_after_called(monkeypatch):
    """async asend(stream=True): before_send fires; after_send fires after stream done."""
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    trace = []

    async def before(conv_, request, **kw):
        trace.append(("before", request.content))

    async def after(conv_, request, **kw):
        trace.append(("after", request.content))

    conv.hook.before_send(before)
    conv.hook.after_send(after)
    monkeypatch.setattr(mod.Conversation, "_asend_stream_impl", _mock_stream_impl)

    chunks = []
    async for chunk in await conv.asend("AsyncStream", stream=True):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert trace == [("before", "AsyncStream"), ("after", "AsyncStream")]


@pytest.mark.asyncio
async def test_async_event_before_and_after_called(monkeypatch):
    """async asend(event=True): before_send fires; after_send fires after event stream."""
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    trace = []

    async def before(conv_, request, **kw):
        trace.append(("before", request.content))

    async def after(conv_, request, **kw):
        trace.append(("after", request.content))

    conv.hook.before_send(before)
    conv.hook.after_send(after)
    monkeypatch.setattr(mod.Conversation, "_asend_with_events_impl", _mock_event_impl)

    events = []
    async for evt in await conv.asend("AsyncEvent", event=True):
        events.append(evt)

    assert len(events) == 1
    assert trace == [("before", "AsyncEvent"), ("after", "AsyncEvent")]


# ---------------------------------------------------------------------------
# Async tool-calling hook tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_tools_before_and_after_called(monkeypatch):
    """async asend() with tools: hooks fire around tool-calling flow."""
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test",
                            tools=[lambda x: x])
    trace = []

    async def before(conv_, request, **kw):
        trace.append(("before", request.content))

    async def after(conv_, request, **kw):
        trace.append(("after", request.content))

    conv.hook.before_send(before)
    conv.hook.after_send(after)
    monkeypatch.setattr(mod.Conversation, "_asend_nonstream_with_tools", _mock_tools_impl)

    result = await conv.asend("ToolsHello")
    assert result.content == "mock tools"
    assert trace == [("before", "ToolsHello"), ("after", "ToolsHello")]


# ---------------------------------------------------------------------------
# Async structured-output hook tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_structured_no_tools_hooks_called(monkeypatch):
    """async asend(returns=Model) without tools: hooks fire."""
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    trace = []

    async def before(conv_, request, **kw):
        trace.append(("before", request.content))

    async def after(conv_, request, **kw):
        trace.append(("after", request.content))

    conv.hook.before_send(before)
    conv.hook.after_send(after)
    monkeypatch.setattr(mod.Conversation, "_asend_with_structured_output", _mock_structured_impl)

    result = await conv.asend("StructHello", returns=DummyModel)
    assert isinstance(result, DummyModel)
    assert result.name == "mock_structured"
    assert trace == [("before", "StructHello"), ("after", "StructHello")]


@pytest.mark.asyncio
async def test_async_structured_with_tools_hooks_called(monkeypatch):
    """async asend(returns=Model) with tools: hooks fire around two-phase flow."""
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test",
                            tools=[lambda x: x])
    trace = []

    async def before(conv_, request, **kw):
        trace.append(("before", request.content))

    async def after(conv_, request, **kw):
        trace.append(("after", request.content))

    conv.hook.before_send(before)
    conv.hook.after_send(after)
    monkeypatch.setattr(mod.Conversation, "_asend_nonstream_with_tools", _mock_tools_impl)
    monkeypatch.setattr(mod.Conversation, "_run_extraction_loop", _mock_extraction_loop)

    result = await conv.asend("StructToolsHello", returns=DummyModel)
    assert isinstance(result, DummyModel)
    assert result.name == "mock_extracted"
    assert trace == [("before", "StructToolsHello"), ("after", "StructToolsHello")]


# ---------------------------------------------------------------------------
# send_kwargs: stream / event modes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_send_kwargs_pass_through(monkeypatch):
    """Stream mode: send_kwargs includes stream=True and custom timeout."""
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    received_kw = []

    async def before(conv_, request, **kw):
        received_kw.append(kw)

    conv.hook.before_send(before)
    monkeypatch.setattr(mod.Conversation, "_asend_stream_impl", _mock_stream_impl)

    async for _ in await conv.asend("KwStream", stream=True, timeout=99):
        pass

    assert received_kw[0]["stream"] is True
    assert received_kw[0]["timeout"] == 99


@pytest.mark.asyncio
async def test_event_send_kwargs_pass_through(monkeypatch):
    """Event mode: send_kwargs includes event=True."""
    conv = mod.Conversation("openai/gpt-4o-mini", api_key="sk-test")
    received_kw = []

    async def before(conv_, request, **kw):
        received_kw.append(kw)

    conv.hook.before_send(before)
    monkeypatch.setattr(mod.Conversation, "_asend_with_events_impl", _mock_event_impl)

    async for _ in await conv.asend("KwEvent", event=True):
        pass

    assert received_kw[0]["event"] is True
