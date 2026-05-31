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
                "model_uri": "deepseek/deepseek-chat",
                "api_key": os.environ["DEEPSEEK_API_KEY"],
                "timeout": 60,
            },
        ],
    )


def test_resilient_provider_live_nonstream_fallback_to_deepseek():
    conv = make_resilient_conversation()

    response = conv.send("Reply with one short sentence containing: resilient ok.", timeout=60)
    metadata = response.metadata.extra

    assert response.content
    assert metadata["fallback_used"] is True
    assert metadata["failover_attempts"] == 2
    assert metadata["resolved_provider"] == "deepseek"
    assert metadata["failed_providers"][0]["attempt_index"] == 0
    assert metadata["failed_providers"][0]["base_url"] == "http://127.0.0.1:9"
    assert metadata["failed_providers"][1]["attempt_index"] == 1
    assert metadata["failed_providers"][1]["base_url"] == "http://127.0.0.1:9/v1"
    assert metadata["failed_providers"][0]["error_type"]
    assert metadata["failed_providers"][1]["error_type"]


@pytest.mark.streaming
def test_resilient_provider_live_streaming_fallback_to_deepseek_before_visible_chunk():
    conv = make_resilient_conversation()

    chunks = list(conv.send("Reply with one short sentence containing: resilient stream ok.", stream=True, timeout=60))
    content = "".join(chunk.content for chunk in chunks if isinstance(chunk, chak.MessageChunk))

    assert content.strip()
    assert not any(isinstance(chunk, chak.FailoverChunk) for chunk in chunks)
    assert conv.messages[-1].content
