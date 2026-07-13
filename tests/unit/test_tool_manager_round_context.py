"""Unit tests for the round_context_fn hook wired into ToolManager.

Verifies that the tool loop calls ``round_context_fn`` before every
provider.send() and uses its return value as the messages sent to the
provider, without disturbing the loop's own append-only history.
"""

import asyncio
from typing import List
from unittest.mock import MagicMock

import pytest

from chak.message import (
    AIMessage,
    ChatCompletionMessageToolCall,
    Function,
    HumanMessage,
    ToolMessage,
)
from chak.tools.manager import ToolManager
from chak.tools.native.function import NativeFunctionTool


pytestmark = pytest.mark.unit


def _tool_call(call_id: str, name: str, args: str = "{}") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=args),
    )


def _make_provider(responses: List[AIMessage]) -> MagicMock:
    """Return a MagicMock provider whose .send() returns ``responses`` in order."""
    provider = MagicMock()
    provider.send.side_effect = list(responses)
    return provider


def test_execute_loop_invokes_round_context_fn_each_iteration():
    """round_context_fn should be called once per LLM round with the
    correct round_index and current append-only history."""

    # Simple native function tool for the loop to call.
    def echo(text: str) -> str:  # pragma: no cover - never actually runs (mock)
        return text

    manager = ToolManager([NativeFunctionTool(echo)])

    # Two rounds: first returns tool_calls, second returns final answer.
    tool_call = _tool_call("call-1", "echo", '{"text": "hi"}')
    response_round0 = AIMessage(content="", tool_calls=[tool_call])
    response_round1 = AIMessage(content="done")

    provider = _make_provider([response_round0, response_round1])

    seen_calls: list = []

    def round_fn(current_messages: List, round_index: int) -> List:
        # Record what the loop passed us so we can assert on it later.
        seen_calls.append((round_index, [type(m).__name__ for m in current_messages]))
        # Return an obviously-different list to prove it's what the loop sends.
        return current_messages + [HumanMessage(content=f"marker-{round_index}")]

    initial = [HumanMessage(content="please echo hi")]

    final_msg, new_messages = asyncio.run(
        manager.execute_loop(
            provider=provider,
            messages=initial,
            model_uri="mock/echo",
            round_context_fn=round_fn,
        )
    )

    # round_context_fn should have been called twice, indexes 0 and 1.
    assert [idx for idx, _ in seen_calls] == [0, 1]

    # Round 0: sees just the initial HumanMessage.
    assert seen_calls[0][1] == ["HumanMessage"]
    # Round 1: append-only history now contains AIMessage(tool_calls) + ToolMessage.
    assert seen_calls[1][1] == ["HumanMessage", "AIMessage", "ToolMessage"]

    # The provider must have received the compressed lists (i.e. those the
    # callback returned), not the raw current_messages.
    for i, call in enumerate(provider.send.call_args_list):
        sent = call.kwargs["messages"]
        marker = sent[-1]
        assert isinstance(marker, HumanMessage) and marker.content == f"marker-{i}"

    # The loop's own append-only history (returned to Conversation) is
    # unaffected by compression — round marker HumanMessages are NOT in it.
    assert not any(
        isinstance(m, HumanMessage) and str(m.content).startswith("marker-")
        for m in new_messages
    )
    # And the final AIMessage is exactly what the second provider.send returned.
    assert final_msg.content == "done"


def test_execute_loop_without_round_context_fn_is_backwards_compatible():
    """When ``round_context_fn`` is omitted the loop behaves exactly as before:
    the append-only ``current_messages`` list is what provider.send receives."""

    def echo(text: str) -> str:  # pragma: no cover - mocked provider
        return text

    manager = ToolManager([NativeFunctionTool(echo)])
    provider = _make_provider([AIMessage(content="just text")])
    initial = [HumanMessage(content="hello")]

    final_msg, _ = asyncio.run(
        manager.execute_loop(
            provider=provider,
            messages=initial,
            model_uri="mock/echo",
        )
    )

    # Provider saw exactly the initial list, unmodified.
    sent = provider.send.call_args_list[0].kwargs["messages"]
    assert [type(m).__name__ for m in sent] == ["HumanMessage"]
    assert sent[0].content == "hello"
    assert final_msg.content == "just text"
