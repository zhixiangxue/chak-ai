"""Unit tests for the boxd provider of the std Sandbox tool (fake boxd client)."""

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from chak.tools.std import Sandbox

pytestmark = pytest.mark.unit


class _FakeMachines:
    def __init__(self, calls, exec_results):
        self.calls = calls
        self.exec_results = list(exec_results)
        self.fs = SimpleNamespace(write=self._write)

    def create(self, name, **kwargs):
        self.calls.append(("create", name, kwargs))
        return SimpleNamespace(id="m-123", name=name)

    def wait_until_ready(self, machine_id, **kwargs):
        self.calls.append(("ready", machine_id))

    def _write(self, machine_id, path, data, **kwargs):
        self.calls.append(("write", machine_id, path, data))

    def exec(self, machine_id, command, **kwargs):
        self.calls.append(("exec", machine_id, command, kwargs))
        result = self.exec_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def delete(self, machine_id, **kwargs):
        self.calls.append(("delete", machine_id))


def _ok(stdout="", stderr="", exit_code=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)


@pytest.fixture
def fake_boxd(monkeypatch):
    """Install a fake `boxd` module; returns (calls, set_exec_results, client_kwargs)."""
    calls = []
    client_kwargs = {}
    state = {"results": [_ok("hello\n")]}

    class FakeBoxd:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)
            self.machines = _FakeMachines(calls, state["results"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            calls.append(("close",))

    module = types.ModuleType("boxd")
    module.Boxd = FakeBoxd
    monkeypatch.setitem(sys.modules, "boxd", module)

    def set_results(*results):
        state["results"] = list(results)

    return calls, set_results, client_kwargs


def _ops(calls):
    return [c[0] for c in calls]


def test_default_provider_is_e2b_and_reads_e2b_key(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "e2b-key")
    monkeypatch.setenv("BOXD_API_KEY", "boxd-key")
    sb = Sandbox()
    assert sb._provider == "e2b"
    assert sb._api_key == "e2b-key"


def test_boxd_provider_reads_boxd_env(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "e2b-key")
    monkeypatch.setenv("BOXD_API_KEY", "boxd-key")
    monkeypatch.setenv("BOXD_ORG", "my-org")
    monkeypatch.setenv("BOXD_REGION", "fra")
    sb = Sandbox(provider="boxd")
    assert sb._api_key == "boxd-key"
    assert sb._org == "my-org"
    assert sb._region == "fra"
    assert "provider=boxd" in repr(sb)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown sandbox provider"):
        Sandbox(provider="nope")


def test_boxd_missing_key_returns_error(monkeypatch, fake_boxd):
    monkeypatch.delenv("BOXD_API_KEY", raising=False)
    calls, _, _ = fake_boxd
    out = asyncio.run(Sandbox(provider="boxd").call({"shell_script": "echo hi"}))
    assert out.startswith("Error: boxd API key not found")
    assert calls == []


def test_boxd_missing_package_returns_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "boxd", None)
    out = asyncio.run(Sandbox(provider="boxd", api_key="k").call({"shell_script": "echo hi"}))
    assert out == "Error: 'boxd' package is not installed. Run: pip install boxd"


def test_boxd_runs_task_on_isolated_machine_and_deletes_it(fake_boxd):
    calls, set_results, client_kwargs = fake_boxd
    set_results(_ok(), _ok("200\n"))
    sb = Sandbox(provider="boxd", api_key="k", org="acme", region="nyc", timeout=30, name_prefix="t-")

    out = asyncio.run(sb.call({
        "shell_script": "python /project/main.py",
        "files": [{"path": "/project/main.py", "content": "print(200)"}],
    }))

    assert out == "200"
    assert client_kwargs == {"api_key": "k", "region": "nyc"}
    assert _ops(calls) == ["create", "ready", "exec", "write", "write", "exec", "delete", "close"]

    _, name, create_kwargs = calls[0]
    assert name.startswith("t-") and len(name) == len("t-") + 12
    assert create_kwargs == {"org": "acme", "isolated": True, "auto_destroy_timeout": 630}

    # machine is addressed by the id create returned
    assert calls[1] == ("ready", "m-123")
    assert calls[-2] == ("delete", "m-123")

    # parent directories are created before files are written
    mkdir = calls[2][2]
    assert "/project" in mkdir and "sudo mkdir -p" in mkdir

    assert calls[3] == ("write", "m-123", "/project/main.py", "print(200)")
    _, _, script_path, script = calls[4]
    assert script_path == "/tmp/_chak_script.sh"
    assert script.startswith("#!/bin/bash\nset -e\n")
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in script
    assert script.endswith("python /project/main.py")

    _, _, command, exec_kwargs = calls[5]
    assert command == "bash /tmp/_chak_script.sh"
    assert exec_kwargs == {"timeout": 30}


def test_boxd_without_files_skips_mkdir(fake_boxd):
    calls, set_results, _ = fake_boxd
    set_results(_ok("hi\n"))
    out = asyncio.run(Sandbox(provider="boxd", api_key="k").call({"shell_script": "echo hi"}))
    assert out == "hi"
    assert _ops(calls) == ["create", "ready", "write", "exec", "delete", "close"]


def test_boxd_reports_stderr_and_exit_code(fake_boxd):
    _, set_results, _ = fake_boxd
    set_results(_ok("out\n", "boom\n", 3))
    out = asyncio.run(Sandbox(provider="boxd", api_key="k").call({"shell_script": "exit 3"}))
    assert out == "out\n\n[stderr]\nboom\n\n[exit code 3]"


def test_boxd_deletes_machine_when_exec_fails(fake_boxd):
    calls, set_results, _ = fake_boxd
    set_results(RuntimeError("exec timed out after 5s"))
    out = asyncio.run(Sandbox(provider="boxd", api_key="k", timeout=5).call({"shell_script": "sleep 60"}))
    assert out == "Error running sandbox: exec timed out after 5s"
    assert ("delete", "m-123") in calls


def test_boxd_no_output(fake_boxd):
    _, set_results, _ = fake_boxd
    set_results(_ok())
    out = asyncio.run(Sandbox(provider="boxd", api_key="k").call({"shell_script": "true"}))
    assert out == "(no output)"


def test_boxd_mkdir_failure_returns_error_and_deletes(fake_boxd):
    calls, set_results, _ = fake_boxd
    set_results(_ok("", "mkdir: Permission denied\n", 1))
    out = asyncio.run(Sandbox(provider="boxd", api_key="k").call({
        "shell_script": "true",
        "files": [{"path": "/etc/x/a.txt", "content": "a"}],
    }))
    assert out == (
        "Error running sandbox: could not create directories (exit code 1): "
        "mkdir: Permission denied"
    )
    assert "write" not in _ops(calls)
    assert ("delete", "m-123") in calls


def test_boxd_create_failure_returns_error_without_delete(fake_boxd, monkeypatch):
    calls, _, _ = fake_boxd

    def boom(self, name, **kwargs):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(_FakeMachines, "create", boom)
    out = asyncio.run(Sandbox(provider="boxd", api_key="k").call({"shell_script": "echo hi"}))
    assert out == "Error running sandbox: quota exceeded"
    assert "delete" not in _ops(calls)
