"""Test fixtures shared across unit tests."""
import sys
import types

from unittest.mock import MagicMock

import pytest
import chak.conversation as mod
from chak.message import AIMessage


@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    """Replace create_provider with a mock so no real LLM calls are made."""
    mock = MagicMock()
    mock.send.return_value = AIMessage(content="Hello from mock")
    monkeypatch.setattr(mod, "create_provider", MagicMock(return_value=mock))
    return mock
