import pytest
from unittest.mock import MagicMock

import chak.conversation as mod

pytestmark = pytest.mark.unit


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
