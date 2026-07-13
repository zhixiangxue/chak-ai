import pytest

from chak.context.handlers import (
    BaseContextHandler,
    FIFOContextHandler,
    LRUContextHandler,
    NoopContextHandler,
)
from chak.message import (
    AIMessage,
    ChatCompletionMessageToolCall,
    Function,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

pytestmark = pytest.mark.unit


def conversation_messages():
    return [
        SystemMessage(content="system"),
        HumanMessage(content="u1"),
        AIMessage(content="a1"),
        HumanMessage(content="u2"),
        AIMessage(content="a2"),
        HumanMessage(content="u3"),
        AIMessage(content="a3"),
    ]


def _tool_call(call_id: str, name: str = "read_pdf") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments="{}"),
    )


# ------------------------------------------------------------------
# Built-in handlers: handle_turn (renamed from handle)
# ------------------------------------------------------------------


def test_noop_context_handler_returns_original_messages():
    messages = conversation_messages()

    result = NoopContextHandler().handle_turn(messages, conversation_id="conv")

    assert result is messages


def test_fifo_context_handler_keeps_system_and_recent_turns():
    result = FIFOContextHandler(keep_recent_turns=2).handle_turn(
        conversation_messages(), conversation_id="conv"
    )

    assert [message.content for message in result] == ["system", "u2", "a2", "u3", "a3"]


def test_lru_context_handler_keeps_system_and_recent_messages():
    result = LRUContextHandler(keep_recent=2).handle_turn(
        conversation_messages(), conversation_id="conv"
    )

    assert [message.content for message in result] == ["system", "u3", "a3"]


def test_fifo_rejects_invalid_keep_recent_turns():
    with pytest.raises(ValueError):
        FIFOContextHandler(keep_recent_turns=0)


# ------------------------------------------------------------------
# Deprecation: `handle` shim on the base class
# ------------------------------------------------------------------


def test_calling_handle_emits_deprecation_warning():
    handler = NoopContextHandler()
    messages = conversation_messages()

    with pytest.warns(DeprecationWarning):
        result = handler.handle(messages, conversation_id="conv")

    assert result is messages


def test_legacy_handle_override_is_bridged_to_handle_turn():
    """Third-party handlers that still override the old ``handle`` should
    keep working: BaseContextHandler.__init_subclass__ aliases their
    override onto ``handle_turn`` so framework internals hit it."""

    with pytest.warns(DeprecationWarning):

        class LegacyHandler(BaseContextHandler):
            def handle(self, messages, *, conversation_id):
                # Emit a sentinel we can detect from the caller.
                return [SystemMessage(content="from-legacy")]

    handler = LegacyHandler()

    # Framework path uses handle_turn — must reach the user's implementation.
    result = handler.handle_turn(
        conversation_messages(), conversation_id="conv"
    )
    assert [m.content for m in result] == ["from-legacy"]


# ------------------------------------------------------------------
# handle_round default: no-op via call_for_round
# ------------------------------------------------------------------


def test_call_for_round_defaults_to_noop():
    handler = NoopContextHandler()  # inherits default handle_round
    messages = conversation_messages()

    result = handler.call_for_round(
        messages, conversation_id="conv", round_index=3
    )

    assert [m.content for m in result] == [m.content for m in messages]


def test_handle_round_override_receives_round_index():
    seen_indices = []

    class RecordingHandler(NoopContextHandler):
        def handle_round(self, messages, *, conversation_id, round_index):
            seen_indices.append(round_index)
            return messages

    handler = RecordingHandler()
    handler.call_for_round(
        conversation_messages(), conversation_id="conv", round_index=0
    )
    handler.call_for_round(
        conversation_messages(), conversation_id="conv", round_index=2
    )

    assert seen_indices == [0, 2]


# ------------------------------------------------------------------
# Enhanced integrity: reverse-orphan sweep
# ------------------------------------------------------------------


def test_integrity_drops_ai_message_missing_tool_response():
    """If an AIMessage(tool_calls=[a, b]) is answered only by ToolMessage(a),
    the AIMessage and its partial answer both go — otherwise Anthropic
    would 400 on the missing tool_result for `b`.
    """
    ai = AIMessage(
        content="calling tools",
        tool_calls=[_tool_call("a"), _tool_call("b")],
    )
    tool_a = ToolMessage(content="answer a", tool_call_id="a")

    handler = NoopContextHandler()
    repaired = handler._ensure_tool_call_integrity(
        [HumanMessage(content="u1"), ai, tool_a]
    )

    # AIMessage and its single answer are both dropped; only the human survives.
    assert [type(m).__name__ for m in repaired] == ["HumanMessage"]


def test_integrity_drops_orphan_tool_message_when_ai_missing():
    """Forward orphan sweep still works: a ToolMessage without a preceding
    AIMessage(tool_calls) is silently dropped."""
    orphan = ToolMessage(content="stale answer", tool_call_id="ghost")

    handler = NoopContextHandler()
    repaired = handler._ensure_tool_call_integrity(
        [HumanMessage(content="u"), orphan, AIMessage(content="a")]
    )

    assert [type(m).__name__ for m in repaired] == ["HumanMessage", "AIMessage"]


def test_integrity_keeps_valid_tool_cycle():
    """Pairing is preserved: intact tool cycles survive both sweeps."""
    ai = AIMessage(
        content="",
        tool_calls=[_tool_call("x")],
    )
    tool_x = ToolMessage(content="x-result", tool_call_id="x")

    handler = NoopContextHandler()
    repaired = handler._ensure_tool_call_integrity(
        [HumanMessage(content="u"), ai, tool_x, AIMessage(content="final")]
    )

    assert [type(m).__name__ for m in repaired] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]


def test_call_for_round_repairs_partial_cycle_offload():
    """A handler that offloads whole cycles at round scope keeps message
    sequence legal; even if it accidentally leaves a lone AIMessage, the
    integrity guard still returns a legal sequence."""

    class BrokenPruner(NoopContextHandler):
        # Simulates a buggy handler that drops a ToolMessage but forgets
        # its AIMessage parent.
        def handle_round(self, messages, *, conversation_id, round_index):
            return [m for m in messages if not isinstance(m, ToolMessage)]

    ai = AIMessage(content="", tool_calls=[_tool_call("q")])
    tool_q = ToolMessage(content="answer", tool_call_id="q")
    messages = [HumanMessage(content="u"), ai, tool_q]

    result = BrokenPruner().call_for_round(
        messages, conversation_id="conv", round_index=1
    )

    # AIMessage would have been an orphan (missing tool_result); guard
    # rail drops it and leaves only the HumanMessage.
    assert [type(m).__name__ for m in result] == ["HumanMessage"]
