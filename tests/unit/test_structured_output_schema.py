from typing import List

import pytest
from pydantic import BaseModel

import chak.conversation as conversation_module
from chak import Conversation
from chak.message import AIMessage, ChatCompletionMessageToolCall, Function

pytestmark = pytest.mark.unit


class Program(BaseModel):
    name: str


class CapturingProvider:
    def __init__(self):
        self.calls = []

    def send(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        tool_name = kwargs["tools"][0]["function"]["name"]
        return AIMessage(
            content="",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name=tool_name,
                        arguments='{"items": [{"name": "Alpha"}, {"name": "Beta"}]}',
                    ),
                )
            ],
        )


@pytest.fixture
def capturing_provider(monkeypatch):
    provider = CapturingProvider()

    def create_provider(provider_name, config_dict, category):
        return provider

    monkeypatch.setattr(conversation_module, "create_provider", create_provider)
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("returns_type", [list[Program], List[Program]])
async def test_structured_output_list_return_uses_object_schema_and_unwraps(capturing_provider, returns_type):
    conv = Conversation("openai/gpt-4o-mini", api_key="test-key")

    result = await conv.asend("Return two programs.", returns=returns_type)

    assert result == [Program(name="Alpha"), Program(name="Beta")]

    schema = capturing_provider.calls[0]["tools"][0]["function"]["parameters"]
    assert schema["type"] == "object"
    assert schema["required"] == ["items"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["items"]["type"] == "array"
