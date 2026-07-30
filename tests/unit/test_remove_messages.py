"""Unit tests for Conversation.remove_message / remove_turn.

Both public methods delegate to the same private _remove_turn: deletion is
always turn-scoped so the assistant(tool_calls) <-> tool result pairing can
never be broken by a partial removal.
"""
import pytest

from chak import Conversation
from chak.message import (
    AIMessage,
    ChatCompletionMessageToolCall,
    Function,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

pytestmark = pytest.mark.unit


def _make_conv() -> Conversation:
    return Conversation("openai/gpt-4o-mini", api_key="test-key")


def _tool_call(call_id: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name="pdf-read_pages", arguments="{}"),
    )


def _build_history(conv: Conversation) -> None:
    """Two turns: a plain QA turn and a tool-calling turn, plus a system message."""
    conv.messages.extend([
        SystemMessage(content="You are helpful.", turn_id=None),
        # Turn 1: plain QA
        HumanMessage(content="hi", turn_id="t1"),
        AIMessage(content="hello", turn_id="t1"),
        # Turn 2: tool-calling turn (4 messages)
        HumanMessage(content="read the pdf", turn_id="t2"),
        AIMessage(content="", tool_calls=[_tool_call("call_1")], turn_id="t2"),
        ToolMessage(content="page text", tool_call_id="call_1", turn_id="t2"),
        AIMessage(content="here is the summary", turn_id="t2"),
    ])


# ---------------------------------------------------------------------------
# remove_turn
# ---------------------------------------------------------------------------

def test_remove_turn_removes_whole_turn_in_order():
    conv = _make_conv()
    _build_history(conv)

    removed = conv.remove_turn("t2")

    assert [m.role for m in removed] == ["user", "assistant", "tool", "assistant"]
    assert len(conv.messages) == 3  # system + turn 1
    assert all(m.turn_id != "t2" for m in conv.messages)


def test_remove_turn_keeps_system_messages_even_with_matching_turn_id():
    conv = _make_conv()
    conv.messages.extend([
        SystemMessage(content="sys", turn_id="t1"),
        HumanMessage(content="hi", turn_id="t1"),
        AIMessage(content="hello", turn_id="t1"),
    ])

    removed = conv.remove_turn("t1")

    assert [m.role for m in removed] == ["user", "assistant"]
    assert len(conv.messages) == 1
    assert conv.messages[0].role == "system"


def test_remove_turn_unknown_id_raises():
    conv = _make_conv()
    _build_history(conv)

    with pytest.raises(ValueError, match="Turn with id 'nope' not found"):
        conv.remove_turn("nope")
    assert len(conv.messages) == 7  # untouched


def test_remove_turn_preserves_messages_list_identity():
    # The tool loop holds a reference to conv.messages (history=...), so
    # removal must mutate in place instead of rebinding the attribute.
    conv = _make_conv()
    _build_history(conv)
    ref = conv.messages

    conv.remove_turn("t1")

    assert conv.messages is ref


# ---------------------------------------------------------------------------
# remove_message
# ---------------------------------------------------------------------------

def test_remove_message_by_human_anchor_removes_whole_turn():
    conv = _make_conv()
    _build_history(conv)
    human_t2 = next(m for m in conv.messages if m.role == "user" and m.turn_id == "t2")

    removed = conv.remove_message(human_t2.id)

    assert len(removed) == 4
    assert all(m.turn_id != "t2" for m in conv.messages)


def test_remove_message_by_ai_anchor_removes_whole_turn():
    conv = _make_conv()
    _build_history(conv)
    final_ai = conv.messages[-1]
    assert final_ai.role == "assistant"

    removed = conv.remove_message(final_ai.id)

    assert len(removed) == 4
    assert len(conv.messages) == 3


def test_remove_message_unknown_id_raises():
    conv = _make_conv()
    _build_history(conv)

    with pytest.raises(ValueError, match="not found"):
        conv.remove_message("no-such-id")


def test_remove_message_tool_message_anchor_raises():
    conv = _make_conv()
    _build_history(conv)
    tool_msg = next(m for m in conv.messages if m.role == "tool")

    with pytest.raises(ValueError, match="role 'tool'"):
        conv.remove_message(tool_msg.id)
    assert len(conv.messages) == 7  # untouched


def test_remove_message_system_message_anchor_raises():
    conv = _make_conv()
    _build_history(conv)
    sys_msg = next(m for m in conv.messages if m.role == "system")

    with pytest.raises(ValueError, match="role 'system'"):
        conv.remove_message(sys_msg.id)


def test_remove_message_without_turn_id_raises():
    conv = _make_conv()
    conv.messages.append(HumanMessage(content="orphan", turn_id=None))
    orphan = conv.messages[-1]

    with pytest.raises(ValueError, match="no turn_id"):
        conv.remove_message(orphan.id)
    assert len(conv.messages) == 1  # untouched


# ---------------------------------------------------------------------------
# Structural safety: no orphan tool_calls / tool results after removal
# ---------------------------------------------------------------------------

def test_removal_never_orphans_tool_pairs():
    conv = _make_conv()
    _build_history(conv)

    conv.remove_turn("t2")

    tool_call_ids = {
        tc.id
        for m in conv.messages
        if m.role == "assistant" and m.tool_calls
        for tc in m.tool_calls
    }
    tool_result_ids = {
        m.tool_call_id for m in conv.messages if m.role == "tool"
    }
    assert tool_call_ids == tool_result_ids == set()


def test_turns_property_reflects_removal():
    conv = _make_conv()
    _build_history(conv)
    assert conv.turns == ["t1", "t2"]

    conv.remove_turn("t1")

    assert conv.turns == ["t2"]
