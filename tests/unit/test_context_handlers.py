import pytest

from chak.context.handlers import FIFOContextHandler, LRUContextHandler, NoopContextHandler
from chak.message import AIMessage, HumanMessage, SystemMessage

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


def test_noop_context_handler_returns_original_messages():
    messages = conversation_messages()

    result = NoopContextHandler().handle(messages, conversation_id="conv")

    assert result is messages


def test_fifo_context_handler_keeps_system_and_recent_turns():
    result = FIFOContextHandler(keep_recent_turns=2).handle(conversation_messages(), conversation_id="conv")

    assert [message.content for message in result] == ["system", "u2", "a2", "u3", "a3"]


def test_lru_context_handler_keeps_system_and_recent_messages():
    result = LRUContextHandler(keep_recent=2).handle(conversation_messages(), conversation_id="conv")

    assert [message.content for message in result] == ["system", "u3", "a3"]


def test_fifo_rejects_invalid_keep_recent_turns():
    with pytest.raises(ValueError):
        FIFOContextHandler(keep_recent_turns=0)
