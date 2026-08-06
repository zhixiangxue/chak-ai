"""Verifies PyRunner auto-discovery against real, independently created venvs.

Unlike test_claude_skill_runner.py (which passes ``python=sys.executable`` for
speed), this test spins up two genuinely separate virtual environments with
the stdlib ``venv`` module (no pip installs, so it stays fast) to prove that
each ClaudeSkill routes to its *own* interpreter rather than falling back to
whatever process is running the test suite. This does not require the
developer's actual per-skill venvs to exist ahead of time.
"""

import asyncio
import sys
import venv
from pathlib import Path

import pytest

from chak.tools.skills import ClaudeSkill, PyRunner


pytestmark = [pytest.mark.unit, pytest.mark.slow]


def _create_light_venv(venv_dir: Path) -> Path:
    """Create a minimal venv (no pip) and return its interpreter path."""
    venv.create(str(venv_dir), with_pip=False, symlinks=False)
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _write_skill(skill_dir: Path, name: str) -> None:
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo skill for venv isolation\n---\n"
        "Report which interpreter executed this script.\n",
        encoding="utf-8",
    )
    (scripts_dir / "whoami.py").write_text(
        "import sys\nprint('interpreter=' + sys.executable)\n",
        encoding="utf-8",
    )


def test_two_skills_route_to_two_independent_real_venvs(tmp_path):
    """Two ClaudeSkill instances with auto-discovered .venv interpreters must
    each execute scripts using their own interpreter, not a shared/default one.
    """
    skill_a_dir = tmp_path / "skill-a"
    skill_b_dir = tmp_path / "skill-b"
    _write_skill(skill_a_dir, "skill-a")
    _write_skill(skill_b_dir, "skill-b")

    python_a = _create_light_venv(skill_a_dir / ".venv")
    python_b = _create_light_venv(skill_b_dir / ".venv")

    # Sanity: the two venvs really are different interpreter binaries, and
    # neither is the interpreter running this test process.
    assert python_a.resolve() != python_b.resolve()
    assert python_a.resolve() != Path(sys.executable).resolve()
    assert python_b.resolve() != Path(sys.executable).resolve()

    skill_a = ClaudeSkill(str(skill_a_dir), runner=PyRunner())
    skill_b = ClaudeSkill(str(skill_b_dir), runner=PyRunner())

    runner_a = skill_a.get_companion_tools()[1]
    runner_b = skill_b.get_companion_tools()[1]

    # Tool names carry the skill namespace, so no collision even though both
    # skills expose a "run_python" capability.
    assert runner_a.name == "skill-a__run_python"
    assert runner_b.name == "skill-b__run_python"

    result_a = asyncio.run(runner_a.call({"script_path": "scripts/whoami.py"}))
    result_b = asyncio.run(runner_b.call({"script_path": "scripts/whoami.py"}))

    assert f"interpreter={python_a.resolve()}" in result_a
    assert f"interpreter={python_b.resolve()}" in result_b
    # Cross-check: skill A's output must NOT report skill B's interpreter and
    # vice versa -- this is the actual routing guarantee developers need.
    assert str(python_b.resolve()) not in result_a
    assert str(python_a.resolve()) not in result_b
