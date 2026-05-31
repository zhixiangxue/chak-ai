import pytest

import chak

pytestmark = [pytest.mark.live, pytest.mark.streaming]


def test_core_provider_streaming_response(core_provider):
    conv = chak.Conversation(core_provider.model_uri, api_key=core_provider.api_key, timeout=60)

    chunks = list(conv.send("Reply with one short sentence containing: chak stream ok.", stream=True, timeout=60))
    content = "".join(chunk.content for chunk in chunks if isinstance(chunk, chak.MessageChunk))

    assert content.strip()
    assert any(isinstance(chunk, chak.MessageChunk) for chunk in chunks)
    assert conv.messages[-1].role == "assistant"
    assert conv.messages[-1].content
