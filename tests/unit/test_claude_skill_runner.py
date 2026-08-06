"""Unit tests for ClaudeSkill script runners."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from chak.tools.manager import ToolManager
from chak.tools.skills import ClaudeSkill, PyRunner


pytestmark = pytest.mark.unit


class DuckTool:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def call(self, arguments: dict, **kwargs) -> str:
        return "ok"


def _write_skill(tmp_path: Path, name: str = "calc") -> Path:
    skill_dir = tmp_path / name
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Calculate mortgage values\n---\n"
        "Use scripts/example.py for calculations.\n",
        encoding="utf-8",
    )
    (scripts_dir / "example.py").write_text(
        "import os\n"
        "import sys\n"
        "stdin = sys.stdin.read()\n"
        "print('cwd=' + os.path.basename(os.getcwd()))\n"
        "print('args=' + ','.join(sys.argv[1:]))\n"
        "print('stdin=' + stdin)\n",
        encoding="utf-8",
    )
    return skill_dir


def _tool_names(manager: ToolManager) -> list[str]:
    return [tool["function"]["name"] for tool in manager._get_openai_tools()]


def test_claude_skill_without_runner_keeps_read_file_only(tmp_path):
    skill = ClaudeSkill(str(_write_skill(tmp_path)))
    manager = ToolManager([skill])

    names = _tool_names(manager)
    assert "calc" in names
    assert "calc__read_file" in names
    assert "calc__run_python" not in names

    # No runner configured: legacy manifest hint (bash + global Python) is
    # preserved verbatim for backwards compatibility with existing callers
    # that still pair ClaudeSkill with a global Bash/Python tool.
    content = asyncio.run(skill.call({}))
    assert "calc__read_file" in content
    assert "use bash with 'python <skill_dir>/<path>'" in content


def test_claude_skill_with_pyrunner_registers_runner_tool(tmp_path):
    skill = ClaudeSkill(
        str(_write_skill(tmp_path)),
        runner=PyRunner(python=sys.executable),
    )
    manager = ToolManager([skill])

    names = _tool_names(manager)
    assert "calc" in names
    assert "calc__read_file" in names
    assert "calc__run_python" in names

    content = asyncio.run(skill.call({}))
    assert "Use calc__run_python" in content
    assert "cwd fixed to this skill directory" in content


def test_pyrunner_executes_skill_script_with_cwd_args_and_stdin(tmp_path):
    skill_dir = _write_skill(tmp_path)
    skill = ClaudeSkill(str(skill_dir), runner=PyRunner(python=sys.executable))
    runner = skill.get_companion_tools()[1]

    result = asyncio.run(
        runner.call(
            {
                "script_path": "scripts/example.py",
                "args": ["--kind", "dti"],
                "stdin": "payload",
            }
        )
    )

    assert "Timed out: false" in result
    assert "Exit code: 0" in result
    assert "cwd=calc" in result
    assert "args=--kind,dti" in result
    assert "stdin=payload" in result


def test_pyrunner_decodes_utf8_stdout_on_windows_locale(tmp_path):
    skill_dir = _write_skill(tmp_path)
    (skill_dir / "scripts" / "unicode_output.py").write_text(
        "import sys\n"
        "sys.stdout.buffer.write('status 🔧 ok … done'.encode('utf-8'))\n",
        encoding="utf-8",
    )
    skill = ClaudeSkill(str(skill_dir), runner=PyRunner(python=sys.executable))
    runner = skill.get_companion_tools()[1]

    result = asyncio.run(runner.call({"script_path": "scripts/unicode_output.py"}))

    assert "Timed out: false" in result
    assert "Exit code: 0" in result
    assert "status 🔧 ok … done" in result
    assert "STDOUT:\nNone" not in result


def test_pyrunner_rejects_path_traversal(tmp_path):
    skill = ClaudeSkill(
        str(_write_skill(tmp_path)),
        runner=PyRunner(python=sys.executable),
    )
    runner = skill.get_companion_tools()[1]

    result = asyncio.run(runner.call({"script_path": "../outside.py"}))

    assert "Error:" in result
    assert "Path traversal" in result


def test_pyrunner_rejects_scripts_outside_allowed_dirs(tmp_path):
    skill_dir = _write_skill(tmp_path)
    (skill_dir / "tool.py").write_text("print('no')\n", encoding="utf-8")
    skill = ClaudeSkill(str(skill_dir), runner=PyRunner(python=sys.executable))
    runner = skill.get_companion_tools()[1]

    result = asyncio.run(runner.call({"script_path": "tool.py"}))

    assert "Error:" in result
    assert "allowed dirs" in result


def test_pyrunner_rejects_non_python_files(tmp_path):
    skill_dir = _write_skill(tmp_path)
    (skill_dir / "scripts" / "notes.txt").write_text("hello\n", encoding="utf-8")
    skill = ClaudeSkill(str(skill_dir), runner=PyRunner(python=sys.executable))
    runner = skill.get_companion_tools()[1]

    result = asyncio.run(runner.call({"script_path": "scripts/notes.txt"}))

    assert "Error:" in result
    assert "Only .py scripts" in result


def test_pyrunner_auto_discovers_skill_venv(tmp_path):
    skill_dir = _write_skill(tmp_path)
    if os.name == "nt":
        python_path = skill_dir / ".venv" / "Scripts" / "python.exe"
    else:
        python_path = skill_dir / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("placeholder", encoding="utf-8")

    skill = ClaudeSkill(str(skill_dir), runner=PyRunner())
    runner = skill.get_companion_tools()[1]

    assert runner.name == "calc__run_python"


def test_tool_manager_rejects_companion_tool_name_conflict(tmp_path):
    skill = ClaudeSkill(
        str(_write_skill(tmp_path)),
        runner=PyRunner(python=sys.executable),
    )

    with pytest.raises(ValueError, match="Duplicate tool name 'calc__run_python'"):
        ToolManager([skill, DuckTool("calc__run_python")])


def test_claude_skill_scan_files_uses_conservative_default_allowlist(tmp_path):
    skill_dir = _write_skill(tmp_path)
    (skill_dir / "pyproject.toml").write_text("[project]\nname = 'calc'\n", encoding="utf-8")
    (skill_dir / "skill.toml").write_text("enabled = true\n", encoding="utf-8")
    (skill_dir / "src").mkdir()
    (skill_dir / "src" / "engine.py").write_text("VALUE = 1\n", encoding="utf-8")
    (skill_dir / "formulas").mkdir()
    (skill_dir / "formulas" / "income.md").write_text("formula docs\n", encoding="utf-8")
    (skill_dir / ".venv" / "Lib" / "site-packages").mkdir(parents=True)
    (skill_dir / ".venv" / "Lib" / "site-packages" / "huge.py").write_text("x\n", encoding="utf-8")
    (skill_dir / "__pycache__").mkdir()
    (skill_dir / "__pycache__" / "ignored.py").write_text("x\n", encoding="utf-8")
    (skill_dir / ".pytest_cache").mkdir()
    (skill_dir / ".pytest_cache" / "ignored.txt").write_text("x\n", encoding="utf-8")
    (skill_dir / ".git").mkdir()
    (skill_dir / ".git" / "config").write_text("secret\n", encoding="utf-8")
    (skill_dir / "notes.md").write_text("root notes should not be exposed\n", encoding="utf-8")
    (skill_dir / "scripts" / "image.png").write_bytes(b"binary")

    skill = ClaudeSkill(str(skill_dir))

    assert skill._files == ["scripts/example.py"]


def test_claude_skill_scan_files_supports_explicit_allowlist_extensions(tmp_path):
    skill_dir = _write_skill(tmp_path)
    (skill_dir / "pyproject.toml").write_text("[project]\nname = 'calc'\n", encoding="utf-8")
    (skill_dir / "src").mkdir()
    (skill_dir / "src" / "engine.py").write_text("VALUE = 1\n", encoding="utf-8")
    (skill_dir / "docs").mkdir()
    (skill_dir / "docs" / "guide.md").write_text("guide\n", encoding="utf-8")
    (skill_dir / "formulas").mkdir()
    (skill_dir / "formulas" / "income.md").write_text("formula docs\n", encoding="utf-8")

    skill = ClaudeSkill(
        str(skill_dir),
        allowed_file_dirs=("scripts", "src", "docs"),
        allowed_files=("pyproject.toml",),
    )

    assert skill._files == [
        "docs/guide.md",
        "pyproject.toml",
        "scripts/example.py",
        "src/engine.py",
    ]


def test_claude_skill_read_file_enforces_same_allowlist(tmp_path):
    skill_dir = _write_skill(tmp_path)
    (skill_dir / "src").mkdir()
    (skill_dir / "src" / "engine.py").write_text("VALUE = 1\n", encoding="utf-8")
    (skill_dir / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (skill_dir / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (skill_dir / ".venv" / "Lib" / "site-packages").mkdir(parents=True)
    (skill_dir / ".venv" / "Lib" / "site-packages" / "pkg.py").write_text("secret\n", encoding="utf-8")

    default_reader = ClaudeSkill(str(skill_dir)).get_file_reader()
    configured_reader = ClaudeSkill(
        str(skill_dir),
        allowed_file_dirs=("scripts", "src"),
        allowed_files=("pyproject.toml",),
    ).get_file_reader()

    assert asyncio.run(default_reader.call({"path": "scripts/example.py"})).startswith("import os")
    assert "Error: File is not exposed by this skill" in asyncio.run(
        default_reader.call({"path": "src/engine.py"})
    )
    assert "Error: File is not exposed by this skill" in asyncio.run(
        default_reader.call({"path": "pyproject.toml"})
    )

    assert asyncio.run(configured_reader.call({"path": "src/engine.py"})) == "VALUE = 1\n"
    assert asyncio.run(configured_reader.call({"path": "pyproject.toml"})) == "[project]\n"

    env_result = asyncio.run(configured_reader.call({"path": ".env"}))
    venv_result = asyncio.run(configured_reader.call({"path": ".venv/Lib/site-packages/pkg.py"}))
    root_result = asyncio.run(configured_reader.call({"path": "notes.md"}))

    assert "Error: File is not exposed by this skill" in env_result
    assert "Error: File is not exposed by this skill" in venv_result
    assert "Error: File is not exposed by this skill" in root_result


def test_claude_skill_read_file_rejects_absolute_paths(tmp_path):
    skill_dir = _write_skill(tmp_path)
    reader = ClaudeSkill(str(skill_dir)).get_file_reader()

    result = asyncio.run(reader.call({"path": str(skill_dir / "scripts" / "example.py")}))

    assert "Error:" in result
    assert "relative path" in result
