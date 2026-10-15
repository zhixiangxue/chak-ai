"""Offline checks for the opt-in local code example."""

import argparse
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from examples.sem_code_mcp import run


@pytest.mark.asyncio
async def test_lookup_scopes_repository_and_selects_only_find(tmp_path, capsys):
    tool = SimpleNamespace(call=AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text="parse_config")], isError=False
    )))
    with patch("examples.sem_code_mcp.which", return_value="/bin/sem"), patch(
        "examples.sem_code_mcp.Server"
    ) as server:
        server.return_value.tools = AsyncMock(return_value=[tool])
        await run(argparse.Namespace(repo=str(tmp_path), query="parse_config", context=True))
        server.assert_called_once_with(
            command="/bin/sem", args=["mcp"],
            env={"SEM_REPO": str(tmp_path.resolve()), "SEM_NO_NETWORK": "1", "SEM_NO_TELEMETRY": "1"},
        )
        server.return_value.tools.assert_awaited_once_with(["sem_find"])
        tool.call.assert_awaited_once_with(
            {"query": "parse_config", "mode": "context", "token_budget": 1200}, retry=False
        )
        assert "parse_config" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [True, False])
async def test_missing_tool_or_server_error_fails(tmp_path, missing):
    tool = SimpleNamespace(call=AsyncMock(return_value=SimpleNamespace(content=[], isError=True)))
    with patch("examples.sem_code_mcp.which", return_value="/bin/sem"), patch(
        "examples.sem_code_mcp.Server"
    ) as server:
        server.return_value.tools = AsyncMock(return_value=[] if missing else [tool])
        with pytest.raises(RuntimeError, match="advertise" if missing else "failed"):
            await run(argparse.Namespace(repo=str(tmp_path), query="parse_config", context=False))
