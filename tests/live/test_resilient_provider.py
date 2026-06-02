"""Live tests for resilient provider failover.

These tests verify the complete resilient failover pipeline:
  primary (unreachable) → fallback_1 (unreachable) → fallback_2 (deepseek, reachable)

Design:
- The first two providers use an unreachable endpoint (127.0.0.1:9) with a
  short timeout (2s) to force fast connection failures.
- The final fallback (deepseek) uses a real endpoint with a generous timeout
  to confirm successful recovery.
"""
import os

import pytest

import chak
from chak.exceptions import ErrorType, ProviderError

pytestmark = [pytest.mark.live]

# Deliberately unreachable endpoint — forces immediate connection error
UNREACHABLE_BASE = "http://127.0.0.1:9"

# All error types that should trigger failover (from ErrorType.RETRYABLE)
# Unreachable endpoints may produce timeout, connection_error, or server_error (e.g. 502)
RETRYABLE_ERROR_TYPES = ErrorType.RETRYABLE


@pytest.fixture
def resilient_conv():
    """Conversation with two unreachable providers + one reachable fallback."""
    missing = [
        name
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        pytest.skip(f"Missing required keys: {', '.join(missing)}")

    return chak.Conversation(
        f"anthropic@{UNREACHABLE_BASE}:claude-haiku-4-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=2,  # short timeout to fail fast on unreachable endpoint
        fallbacks=[
            {
                "model_uri": f"openai@{UNREACHABLE_BASE}/v1:gpt-4o-mini",
                "api_key": os.environ["OPENAI_API_KEY"],
                "timeout": 2,
            },
            {
                "model_uri": "deepseek/deepseek-v4-flash",
                "api_key": os.environ["DEEPSEEK_API_KEY"],
                "timeout": 60,
            },
        ],
    )


@pytest.fixture
def deepseek_conv():
    """Direct (non-resilient) conversation for baseline tests."""
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY required")

    return chak.Conversation(
        "deepseek/deepseek-v4-flash",
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )


# ------------------------------------------------------------------ #
# Non-streaming failover                                               #
# ------------------------------------------------------------------ #

def test_nonstream_failover_to_deepseek(resilient_conv):
    """Primary + fallback_1 fail → fallback_2 (deepseek) succeeds."""
    response = resilient_conv.send(
        "Reply with exactly: resilient ok", timeout=60
    )
    trace = response.metadata.provider_trace

    # Content was produced
    assert response.content
    assert "resilient" in response.content.lower()

    # Failover metadata
    assert trace.fallback_used is True
    assert trace.failover_attempts == 2
    assert trace.primary_provider == "anthropic"
    assert trace.primary_model == "claude-haiku-4-5"
    assert trace.resolved_provider == "deepseek"
    assert trace.resolved_model == "deepseek-v4-flash"

    # Failed provider details
    assert len(trace.failed_providers) == 2
    assert trace.failed_providers[0].attempt_index == 0
    assert trace.failed_providers[0].base_url == UNREACHABLE_BASE
    assert trace.failed_providers[0].error_type in RETRYABLE_ERROR_TYPES
    assert trace.failed_providers[1].attempt_index == 1
    assert trace.failed_providers[1].base_url == f"{UNREACHABLE_BASE}/v1"
    assert trace.failed_providers[1].error_type in RETRYABLE_ERROR_TYPES


# ------------------------------------------------------------------ #
# Streaming failover                                                   #
# ------------------------------------------------------------------ #

@pytest.mark.streaming
def test_streaming_failover_to_deepseek(resilient_conv):
    """Streaming: failover happens before any visible chunk is emitted."""
    chunks = list(resilient_conv.send(
        "Reply with exactly: resilient stream ok",
        stream=True, timeout=60,
    ))
    content = "".join(
        chunk.content for chunk in chunks if isinstance(chunk, chak.MessageChunk)
    )

    # Content integrity
    assert content.strip()
    assert "resilient" in content.lower()

    # No FailoverChunk should be visible (failover is transparent)
    assert not any(isinstance(chunk, chak.FailoverChunk) for chunk in chunks)

    # Final message stored in conversation history
    assert resilient_conv.messages[-1].content

    # Provider trace on final AI message
    trace = resilient_conv.messages[-1].metadata.provider_trace
    assert trace is not None
    assert trace.fallback_used is True
    assert trace.failover_attempts == 2
    assert trace.primary_provider == "anthropic"
    assert trace.resolved_provider == "deepseek"
    assert trace.resolved_model == "deepseek-v4-flash"

    # Failed providers carry retryable error types
    for fp in trace.failed_providers:
        assert fp.error_type in RETRYABLE_ERROR_TYPES


# ------------------------------------------------------------------ #
# All providers fail → ProviderError raised                            #
# ------------------------------------------------------------------ #

def test_all_providers_fail_raises():
    """When every provider is unreachable, ProviderError is raised."""
    missing = [
        name for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        pytest.skip(f"Missing required keys: {', '.join(missing)}")

    conv = chak.Conversation(
        f"anthropic@{UNREACHABLE_BASE}:claude-haiku-4-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=2,
        fallbacks=[
            {
                "model_uri": f"openai@{UNREACHABLE_BASE}/v1:gpt-4o-mini",
                "api_key": os.environ["OPENAI_API_KEY"],
                "timeout": 2,
            },
        ],
    )

    with pytest.raises(ProviderError) as exc_info:
        conv.send("hello", timeout=10)

    err = exc_info.value
    # The aggregate error wrapping all failures has its own error_type
    assert err.error_type == ErrorType.FAILOVER_EXHAUSTED


# ------------------------------------------------------------------ #
# Non-resilient (direct) provider_trace baseline                       #
# ------------------------------------------------------------------ #

def test_direct_provider_has_provider_trace(deepseek_conv):
    """Non-resilient calls must still populate provider_trace correctly."""
    response = deepseek_conv.send(
        "Reply with exactly: hello world", timeout=60
    )
    trace = response.metadata.provider_trace

    assert response.content
    assert trace is not None
    assert trace.fallback_used is False
    assert trace.failover_attempts == 0
    assert trace.failed_providers == []
    assert trace.primary_provider == "deepseek"
    assert trace.primary_model == "deepseek-v4-flash"
    assert trace.resolved_provider == "deepseek"
    assert trace.resolved_model == "deepseek-v4-flash"


# ------------------------------------------------------------------ #
# Conversation.get_provider_traces() across multiple turns             #
# ------------------------------------------------------------------ #

def test_get_provider_traces_multi_turn(resilient_conv):
    """get_provider_traces() returns one trace per AI response."""
    resilient_conv.send("Reply with exactly: first", timeout=60)
    resilient_conv.send("Reply with exactly: second", timeout=60)

    traces = resilient_conv.get_provider_traces()
    assert len(traces) == 2
    for trace in traces:
        assert trace.fallback_used is True
        assert trace.resolved_provider == "deepseek"
        assert trace.resolved_model == "deepseek-v4-flash"
        assert trace.failover_attempts == 2
