import pytest

from chak.message import AIMessage, FailoverChunk, HumanMessage, MessageChunk, ReasoningChunk, SystemMessage

pytestmark = pytest.mark.unit


def test_stream_chunk_types_keep_expected_fields():
    message_chunk = MessageChunk(content="hello", is_final=False, metadata={"provider": "test"})
    reasoning_chunk = ReasoningChunk(content="thinking", is_final=False)
    failover_chunk = FailoverChunk(failed_provider="openai", next_provider="deepseek", error="timeout")

    assert message_chunk.content == "hello"
    assert message_chunk.metadata["provider"] == "test"
    assert reasoning_chunk.content == "thinking"
    assert failover_chunk.failed_provider == "openai"
    assert failover_chunk.next_provider == "deepseek"
    assert failover_chunk.is_final is False


def test_message_roles_are_stable():
    assert HumanMessage(content="hi").role == "user"
    assert AIMessage(content="hello").role == "assistant"
    assert SystemMessage(content="system").role == "system"
