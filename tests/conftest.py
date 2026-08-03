import os
from pathlib import Path
from typing import Dict, List, Union

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


CORE_PROVIDERS: Dict[str, Union[ProviderCase, List[ProviderCase]]] = {
    "deepseek": ProviderCase(
        name="deepseek",
        model_uri="deepseek/deepseek-v4-pro",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "qwen": ProviderCase(
        name="qwen",
        model_uri="bailian/qwen-plus",
        api_key_env="BAILIAN_API_KEY",
    ),
    "zhipu": ProviderCase(
        name="zhipu",
        model_uri="zhipu/glm-5.2",
        api_key_env="ZHIPU_API_KEY",
    ),
    "openai": [
        ProviderCase(
            name="openai",
            model_uri="openai/gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
        ),
        ProviderCase(
            name="openai",
            model_uri="openai/gpt-5.6-luna",
            api_key_env="OPENAI_API_KEY",
        ),
    ],
    "claude": ProviderCase(
        name="claude",
        model_uri="anthropic/claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "minimax": ProviderCase(
        name="minimax",
        model_uri="minimax@https://api.minimax.io/anthropic:MiniMax-M3",
        api_key_env="MINIMAX_API_KEY",
    ),
}


def _normalize_cases(value: Union[ProviderCase, List[ProviderCase]]) -> List[ProviderCase]:
    """Allow each provider key to map to one or more ProviderCase entries."""
    if isinstance(value, list):
        return value
    return [value]


# Build the flat list of parametrized cases. Each key in CORE_PROVIDERS can
# hold a single ProviderCase or a list of them (e.g. openai has gpt-4o-mini
# and gpt-5.6-luna). The test id includes the model name when a provider has
# multiple cases, so output stays readable: "openai[gpt-4o-mini]",
# "openai[gpt-5.6-luna]".
_all_params: list = []
for _mark_name, _value in CORE_PROVIDERS.items():
    for _case in _normalize_cases(_value):
        _case_id = _mark_name
        # When a provider has multiple cases, append model name to disambiguate
        if len(_normalize_cases(_value)) > 1:
            _model_short = _case.model_uri.split("/")[-1]
            _case_id = f"{_mark_name}-{_model_short}"
        _all_params.append(
            pytest.param(_case, id=_case_id, marks=getattr(pytest.mark, _mark_name))
        )


@pytest.fixture(params=_all_params)
def core_provider(request) -> ProviderCase:
    """Parametrized fixture running tests across all core providers.

    Usage:
        pytest tests/live/ -k deepseek       # run only deepseek
        pytest tests/live/ -k 'claude or minimax'  # run claude + minimax
        pytest tests/live/ -k openai          # run all openai models
        pytest tests/live/                     # run all providers
    """
    provider = request.param
    if not provider.api_key:
        pytest.skip(f"{provider.api_key_env} is not set")
    return provider


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT
