import os

import pytest

import chak

pytestmark = [pytest.mark.live]


def require_resilient_keys():
    missing = [
        name
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        pytest.skip(f"Missing required resilient provider keys: {', '.join(missing)}")


def make_resilient_conversation():
    require_resilient_keys()
    return chak.Conversation(
        "anthropic@http://127.0.0.1:9:claude-haiku-4-5",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=2,
        fallback_models=[
            {
                "model_uri": "openai@http://127.0.0.1:9/v1:gpt-4o-mini",
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


# ------------------------------------------------------------------ #
# Non-streaming failover                                               #
# ------------------------------------------------------------------ #

def test_resilient_provider_live_nonstream_fallback_to_deepseek():
    conv = make_resilient_conversation()

    response = conv.send("Reply with one short sentence containing: resilient ok.", timeout=60)
    trace = response.metadata.provider_trace

    assert response.content
    assert trace.fallback_used is True
    assert trace.failover_attempts == 2
    assert trace.primary_provider == "anthropic"
    assert trace.primary_model == "claude-haiku-4-5"
    assert trace.resolved_provider == "deepseek"
    assert trace.resolved_model == "deepseek-v4-flash"
    assert trace.failed_providers[0].attempt_index == 0
    assert trace.failed_providers[0].base_url == "http://127.0.0.1:9"
    assert trace.failed_providers[1].attempt_index == 1
    assert trace.failed_providers[1].base_url == "http://127.0.0.1:9/v1"
    assert trace.failed_providers[0].error_type
    assert trace.failed_providers[1].error_type


# ------------------------------------------------------------------ #
# Streaming failover (before visible chunk)                            #
# ------------------------------------------------------------------ #

@pytest.mark.streaming
def test_resilient_provider_live_streaming_fallback_to_deepseek_before_visible_chunk():
    conv = make_resilient_conversation()

    chunks = list(conv.send("Reply with one short sentence containing: resilient stream ok.", stream=True, timeout=60))
    content = "".join(chunk.content for chunk in chunks if isinstance(chunk, chak.MessageChunk))

    assert content.strip()
    assert not any(isinstance(chunk, chak.FailoverChunk) for chunk in chunks)
    assert conv.messages[-1].content

    # Verify provider_trace on the final AI message
    trace = conv.messages[-1].metadata.provider_trace
    assert trace is not None
    assert trace.fallback_used is True
    assert trace.primary_provider == "anthropic"
    assert trace.resolved_provider == "deepseek"


# ------------------------------------------------------------------ #
# Non-resilient (direct) provider_trace                                #
# ------------------------------------------------------------------ #

def test_direct_provider_has_provider_trace():
    """Non-resilient calls must still populate provider_trace."""
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY required")

    conv = chak.Conversation(
        "deepseek/deepseek-v4-flash",
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
    response = conv.send("Reply with one short sentence: hello world.", timeout=60)
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
# Conversation.get_provider_traces()                                   #
# ------------------------------------------------------------------ #

def test_get_provider_traces():
    require_resilient_keys()

    conv = make_resilient_conversation()
    conv.send("Reply with one short sentence: first.", timeout=60)
    conv.send("Reply with one short sentence: second.", timeout=60)

    traces = conv.get_provider_traces()
    assert len(traces) == 2
    for trace in traces:
        assert trace.fallback_used is True
        assert trace.resolved_provider == "deepseek"
