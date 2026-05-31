import pytest

import chak

pytestmark = [pytest.mark.live]


def test_core_provider_sync_response(core_provider):
    conv = chak.Conversation(core_provider.model_uri, api_key=core_provider.api_key, timeout=60)

    response = conv.send("Reply with one short sentence containing: chak sync ok.", timeout=60)

    assert isinstance(response, chak.AIMessage)
    assert response.content
