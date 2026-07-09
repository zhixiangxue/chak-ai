"""Live integration tests for lifecycle hooks across all send modes.

Verifies that before_send / after_send fire correctly with real LLM calls
for every core provider (deepseek/qwen/openai/claude/minimax) and every
send mode (non-stream/stream/event/tool/structured/structured+tool).
"""

import pytest
from pydantic import BaseModel

import chak
from chak.tools import wrap_tools

pytestmark = [pytest.mark.live, pytest.mark.hook]

# ── helpers ────────────────────────────────────────────────────────────────


class CityInfo(BaseModel):
    city: str
    country: str


async def _collect_hook_trace(conv, prompt, **send_kwargs):
    """Send *prompt* and return (before_trace, after_trace) lists."""
    before_trace = []
    after_trace = []

    async def before(conv_, request, **kw):
        before_trace.append({"content": request.content, "kw": dict(kw)})

    async def after(conv_, request, **kw):
        after_trace.append({"content": request.content, "kw": dict(kw)})

    conv.hook.before_send(before)
    conv.hook.after_send(after)
    await conv.asend(prompt, **send_kwargs)
    return before_trace, after_trace


async def _collect_stream_trace(conv, prompt, **send_kwargs):
    """Send *prompt* in stream mode and return (before, after) traces."""
    before_trace = []
    after_trace = []

    async def before(conv_, request, **kw):
        before_trace.append({"content": request.content, "kw": dict(kw)})

    async def after(conv_, request, **kw):
        after_trace.append({"content": request.content, "kw": dict(kw)})

    conv.hook.before_send(before)
    conv.hook.after_send(after)

    async for _ in await conv.asend(prompt, stream=True, **send_kwargs):
        pass
    return before_trace, after_trace


async def _collect_event_trace(conv, prompt, **send_kwargs):
    """Send *prompt* in event mode and return (before, after) traces."""
    before_trace = []
    after_trace = []

    async def before(conv_, request, **kw):
        before_trace.append({"content": request.content, "kw": dict(kw)})

    async def after(conv_, request, **kw):
        after_trace.append({"content": request.content, "kw": dict(kw)})

    conv.hook.before_send(before)
    conv.hook.after_send(after)

    async for _ in await conv.asend(prompt, event=True, **send_kwargs):
        pass
    return before_trace, after_trace


def _assert_hooks_fired(before_trace, after_trace, expected_content):
    """Common assertions: both hooks fired with correct request content."""
    assert len(before_trace) == 1, "before_send should fire once"
    assert len(after_trace) == 1, "after_send should fire once"
    assert before_trace[0]["content"] == expected_content
    assert after_trace[0]["content"] == expected_content


# ── non-streaming ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hook_basic(core_provider):
    """Non-streaming: before_send + after_send fire with correct content."""
    conv = chak.Conversation(
        core_provider.model_uri, api_key=core_provider.api_key, timeout=60
    )
    prompt = "Reply with exactly: hook test ok"
    before, after = await _collect_hook_trace(conv, prompt)

    _assert_hooks_fired(before, after, prompt)
    # after_send should see the assistant response in conversation
    assert conv.messages[-1].content


# ── streaming ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.streaming
async def test_hook_streaming(core_provider):
    """Streaming: before_send fires; after_send fires after stream exhausted."""
    conv = chak.Conversation(
        core_provider.model_uri, api_key=core_provider.api_key, timeout=60
    )
    prompt = "Reply with exactly: hook stream ok"
    before, after = await _collect_stream_trace(conv, prompt)

    _assert_hooks_fired(before, after, prompt)
    assert conv.messages[-1].content


# ── event stream ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.streaming
async def test_hook_event_stream(core_provider):
    """Event stream: before_send fires; after_send fires after event loop."""
    conv = chak.Conversation(
        core_provider.model_uri, api_key=core_provider.api_key, timeout=60
    )
    prompt = "Reply with exactly: hook event ok"
    before, after = await _collect_event_trace(conv, prompt)

    _assert_hooks_fired(before, after, prompt)
    assert conv.messages[-1].content


# ── tool calling ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.tools
async def test_hook_with_tools(core_provider):
    """Tool calling: hooks fire around tool-execution flow."""
    def add_numbers(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        return a + b

    conv = chak.Conversation(
        core_provider.model_uri,
        api_key=core_provider.api_key,
        tools=wrap_tools([add_numbers]),
        timeout=120,
    )
    prompt = (
        "You must call the add_numbers tool with a=2 and b=3. "
        "Do not answer directly before calling it."
    )
    before, after = await _collect_hook_trace(conv, prompt)

    _assert_hooks_fired(before, after, prompt)
    assert conv.messages[-1].content


# ── structured output (no tools) ───────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.structured
async def test_hook_structured_output(core_provider):
    """Structured output without tools: hooks fire around extraction."""
    conv = chak.Conversation(
        core_provider.model_uri, api_key=core_provider.api_key, timeout=90
    )
    prompt = "Return structured data for Paris, France."
    before, after = await _collect_hook_trace(
        conv, prompt, returns=CityInfo, timeout=90
    )

    _assert_hooks_fired(before, after, prompt)


# ── structured output + tools ──────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.structured
@pytest.mark.tools
async def test_hook_structured_with_tools(core_provider):
    """Structured output with tools: hooks fire around two-phase flow."""
    def lookup_city(name: str) -> str:
        """Return a fake city lookup result."""
        return f"City: {name}, Country: France, Population: 2M"

    conv = chak.Conversation(
        core_provider.model_uri,
        api_key=core_provider.api_key,
        tools=wrap_tools([lookup_city]),
        timeout=120,
    )
    prompt = (
        "Look up information about Paris using the lookup_city tool, "
        "then return structured data."
    )
    before, after = await _collect_hook_trace(
        conv, prompt, returns=CityInfo, timeout=120
    )

    _assert_hooks_fired(before, after, prompt)


# ── send_kwargs passthrough ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hook_send_kwargs_passthrough(core_provider):
    """Verify hook callbacks receive correct send_kwargs."""
    conv = chak.Conversation(
        core_provider.model_uri, api_key=core_provider.api_key, timeout=60
    )
    before_kw = []

    async def before(conv_, request, **kw):
        before_kw.append(dict(kw))

    conv.hook.before_send(before)
    await conv.asend("Reply with exactly: hook kw ok", timeout=99)

    assert len(before_kw) == 1
    assert before_kw[0]["timeout"] == 99
    assert before_kw[0]["stream"] is False
    assert before_kw[0]["event"] is False
