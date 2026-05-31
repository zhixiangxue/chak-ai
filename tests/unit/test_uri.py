import pytest

from chak.exceptions import URIError
from chak.utils.uri import build, parse

pytestmark = pytest.mark.unit


def test_parse_simple_provider_model_uri():
    parsed = parse("deepseek/deepseek-chat")

    assert parsed["provider"] == "deepseek"
    assert parsed["model"] == "deepseek-chat"
    assert parsed["base_url"] is None
    assert parsed["params"] == {}


def test_parse_full_uri_preserves_base_url_path():
    parsed = parse("openai@http://127.0.0.1:9/v1:gpt-4o-mini?temperature=0.1")

    assert parsed["provider"] == "openai"
    assert parsed["base_url"] == "http://127.0.0.1:9/v1"
    assert parsed["model"] == "gpt-4o-mini"
    assert parsed["params"]["temperature"] == "0.1"


def test_build_uses_default_base_url_placeholder():
    assert build("openai", "gpt-4o-mini") == "openai@~:gpt-4o-mini"


@pytest.mark.parametrize("uri", ["", "openai", "openai/", "/gpt-4o-mini"])
def test_invalid_uri_raises(uri):
    with pytest.raises(URIError):
        parse(uri)
