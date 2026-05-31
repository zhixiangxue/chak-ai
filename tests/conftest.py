import os
from pathlib import Path
from typing import Dict

import pytest
from pydantic import BaseModel

try:
    import dotenv
except ImportError:  # pragma: no cover
    dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if dotenv is not None:
    dotenv.load_dotenv(PROJECT_ROOT / ".env")


class ProviderCase(BaseModel):
    name: str
    model_uri: str
    api_key_env: str

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")


CORE_PROVIDERS: Dict[str, ProviderCase] = {
    "deepseek": ProviderCase(
        name="deepseek",
        model_uri="deepseek/deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "qwen": ProviderCase(
        name="qwen",
        model_uri="bailian/qwen-plus",
        api_key_env="BAILIAN_API_KEY",
    ),
    "openai": ProviderCase(
        name="openai",
        model_uri="openai/gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    "claude": ProviderCase(
        name="claude",
        model_uri="anthropic/claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
    ),
}


def _provider_param(name: str):
    return pytest.param(CORE_PROVIDERS[name], id=name, marks=getattr(pytest.mark, name))


@pytest.fixture(params=[_provider_param("deepseek"), _provider_param("qwen"), _provider_param("openai"), _provider_param("claude")])
def core_provider(request) -> ProviderCase:
    provider = request.param
    if not provider.api_key:
        pytest.skip(f"{provider.api_key_env} is not set")
    return provider


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT
