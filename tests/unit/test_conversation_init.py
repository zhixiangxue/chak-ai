from types import SimpleNamespace

import pytest

import chak.conversation as conversation_module
from chak import Conversation
from chak.providers.llm.resilient import ResilientProvider

pytestmark = pytest.mark.unit


@pytest.fixture
def provider_factory(monkeypatch):
    created = []

    def create_provider(provider_name, config_dict, category):
        provider = SimpleNamespace(
            provider_name=provider_name,
            config=SimpleNamespace(**config_dict),
            converter=SimpleNamespace(),
        )
        created.append((provider_name, config_dict, category, provider))
        return provider

    monkeypatch.setattr(conversation_module, "create_provider", create_provider)
    return created


def test_without_fallback_uses_single_provider_path(provider_factory):
    conv = Conversation("openai/gpt-4o-mini", api_key="primary-key", timeout=5)

    assert len(provider_factory) == 1
    provider_name, config, _, provider = provider_factory[0]
    assert provider_name == "openai"
    assert config["api_key"] == "primary-key"
    assert config["model"] == "gpt-4o-mini"
    assert config["timeout"] == 5
    assert conv.provider is provider


def test_fallbacks_create_resilient_provider_with_provider_name_in_config(provider_factory):
    conv = Conversation(
        "anthropic@http://127.0.0.1:9:claude-haiku-4-5",
        api_key="anthropic-key",
        timeout=2,
        fallbacks=[
            {"model_uri": "openai@http://127.0.0.1:9/v1:gpt-4o-mini", "api_key": "openai-key", "timeout": 3},
            {"model_uri": "deepseek/deepseek-chat", "api_key": "deepseek-key"},
        ],
    )

    assert isinstance(conv.provider, ResilientProvider)
    assert len(provider_factory) == 3
    assert provider_factory[0][1]["provider_name"] == "anthropic"
    assert provider_factory[1][1]["provider_name"] == "openai"
    assert provider_factory[2][1]["provider_name"] == "deepseek"
    assert provider_factory[0][1]["base_url"] == "http://127.0.0.1:9"
    assert provider_factory[1][1]["base_url"] == "http://127.0.0.1:9/v1"
    assert provider_factory[2][1]["api_key"] == "deepseek-key"


@pytest.mark.parametrize("fallbacks", [["openai/gpt-4o-mini"], [123]])
def test_fallbacks_must_be_dict_list(fallbacks, provider_factory):
    with pytest.raises(TypeError, match="fallback model spec must be a dict"):
        Conversation("deepseek/deepseek-chat", api_key="key", fallbacks=fallbacks)


def test_fallback_requires_model_uri(provider_factory):
    with pytest.raises(ValueError, match="fallback model spec requires 'model_uri'"):
        Conversation("deepseek/deepseek-chat", api_key="key", fallbacks=[{"api_key": "backup"}])


def test_fallback_nested_kwargs_must_be_dict(provider_factory):
    with pytest.raises(TypeError, match="fallback model 'kwargs' must be a dict"):
        Conversation(
            "deepseek/deepseek-chat",
            api_key="key",
            fallbacks=[{"model_uri": "openai/gpt-4o-mini", "kwargs": "bad"}],
        )
