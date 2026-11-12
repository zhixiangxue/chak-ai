"""Live tests for the boxd provider of the std Sandbox tool.

Each call creates a real isolated boxd machine and deletes it afterwards.

Run:
    BOXD_API_KEY=... pytest tests/live/test_sandbox_boxd.py -m boxd

Optional: BOXD_ORG, BOXD_REGION, BOXD_SANDBOX_NAME_PREFIX.
"""

import asyncio
import hashlib
import os
from datetime import UTC, datetime, timedelta

import pytest

from chak.tools.std import Sandbox

boxd = pytest.importorskip("boxd")

pytestmark = [
    pytest.mark.live,
    pytest.mark.boxd,
    pytest.mark.skipif(not os.getenv("BOXD_API_KEY"), reason="BOXD_API_KEY not set"),
]


@pytest.fixture
def created(monkeypatch):
    """Record the id of every machine the tool creates."""
    ids = []
    original = boxd.resources.machines.Machines.create

    def spy(self, *args, **kwargs):
        machine = original(self, *args, **kwargs)
        ids.append(machine.id)
        print(f"created machine {machine.id} ({machine.name})")
        return machine

    monkeypatch.setattr(boxd.resources.machines.Machines, "create", spy)
    return ids


def _sandbox(timeout=180):
    return Sandbox(
        provider="boxd",
        timeout=timeout,
        name_prefix=os.getenv("BOXD_SANDBOX_NAME_PREFIX", "chak-sandbox-test-"),
    )


def _assert_deleted(ids):
    assert ids
    with boxd.Boxd(api_key=os.environ["BOXD_API_KEY"], region=os.getenv("BOXD_REGION") or None) as client:
        for machine_id in ids:
            with pytest.raises(boxd.errors.NotFoundError):
                client.machines.get(machine_id)


def test_boxd_sandbox_runs_multi_file_project(created):
    big = "x" * (1024 * 1024)
    out = asyncio.run(_sandbox().call({
        "shell_script": (
            "pip install httpx -q\n"
            "cd /project && python main.py\n"
            "sha256sum /project/data/big.txt | cut -d' ' -f1"
        ),
        "files": [
            {"path": "/project/main.py", "content": "from util import greet\nimport httpx\nprint(greet(), httpx.__name__)\n"},
            {"path": "/project/util.py", "content": "def greet():\n    return 'hello from boxd'\n"},
            {"path": "/project/data/big.txt", "content": big},
        ],
    }))
    print(out)
    assert "hello from boxd httpx" in out
    assert hashlib.sha256(big.encode()).hexdigest() in out
    assert "[exit code" not in out
    _assert_deleted(created)


def test_boxd_sandbox_reports_failure(created):
    out = asyncio.run(_sandbox().call({"shell_script": "echo before\necho oops >&2\nexit 7\necho after"}))
    print(out)
    assert "before" in out
    assert "[stderr]\noops" in out
    assert "[exit code 7]" in out
    assert "after" not in out
    _assert_deleted(created)


def test_boxd_sandbox_times_out(created):
    out = asyncio.run(_sandbox(timeout=5).call({"shell_script": "sleep 60"}))
    print(out)
    assert out.startswith("Error running sandbox:")
    assert "timed out" in out
    _assert_deleted(created)


def test_boxd_sandbox_is_isolated_with_delete_after(created, monkeypatch):
    seen = []
    original = boxd.resources.machines.Machines.wait_until_ready

    def spy(self, machine_id, **kwargs):
        machine = original(self, machine_id, **kwargs)
        seen.append((machine.networking.isolated, machine.delete_at, datetime.now(UTC)))
        return machine

    monkeypatch.setattr(boxd.resources.machines.Machines, "wait_until_ready", spy)
    started = datetime.now(UTC)
    out = asyncio.run(_sandbox().call({"shell_script": "echo hi"}))
    assert out == "hi"
    [(isolated, delete_at, ready_at)] = seen
    assert isolated is True
    # a fixed timer of timeout + 600 seconds, counted from creation
    assert delete_at is not None
    assert started + timedelta(seconds=780 - 30) <= delete_at <= ready_at + timedelta(seconds=780 + 30)
    _assert_deleted(created)


def test_boxd_sandbox_bad_key():
    out = asyncio.run(Sandbox(provider="boxd", api_key="not-a-real-key").call({"shell_script": "echo hi"}))
    print(out)
    assert out.startswith("Error running sandbox:")
