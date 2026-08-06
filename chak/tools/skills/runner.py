"""Script runners for ClaudeSkill companion execution tools."""

import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class ScriptRunner:
    """Base class for skill-scoped script runners.

    Runner instances describe runtime configuration. ``bind`` returns a concrete
    duck-typed tool bound to one ClaudeSkill name and directory.
    """

    name_suffix: str = "run"
    description: str = "Run a script in this skill."

    def bind(self, skill_name: str, skill_dir: Path):
        """Bind this runner to a concrete skill directory."""
        raise NotImplementedError


class PyRunner(ScriptRunner):
    """Python script runner for ClaudeSkill.

    The runner executes existing ``.py`` files inside the skill directory using
    a fixed Python interpreter. It does not expose arbitrary shell commands or
    let the LLM override the interpreter.
    """

    name_suffix = "run_python"
    description = "Execute a Python script in this skill using its configured interpreter."

    def __init__(
        self,
        python: Optional[str] = None,
        timeout: int = 60,
        allowed_dirs: Sequence[str] = ("scripts",),
        allow_arbitrary_code: bool = False,
    ):
        """
        Args:
            python: Python interpreter path. If omitted, bind() looks for a
                ``.venv`` interpreter under the skill directory.
            timeout: Maximum seconds allowed per script execution.
            allowed_dirs: Relative directories whose scripts may be executed.
                Pass an empty sequence to allow any script under the skill dir.
            allow_arbitrary_code: Reserved for future explicit eval support.
                The first implementation only executes existing script files.
        """
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        self.python = python
        self.timeout = timeout
        self.allowed_dirs = tuple(allowed_dirs)
        self.allow_arbitrary_code = allow_arbitrary_code

    def bind(self, skill_name: str, skill_dir: Path) -> "BoundPyRunner":
        python_path = self._resolve_python(skill_dir)
        return BoundPyRunner(
            skill_name=skill_name,
            skill_dir=skill_dir,
            python=python_path,
            timeout=self.timeout,
            allowed_dirs=self.allowed_dirs,
            allow_arbitrary_code=self.allow_arbitrary_code,
        )

    def _resolve_python(self, skill_dir: Path) -> Path:
        if self.python:
            python_path = Path(self.python).expanduser().resolve()
        else:
            win_python = skill_dir / ".venv" / "Scripts" / "python.exe"
            posix_python = skill_dir / ".venv" / "bin" / "python"
            python_path = win_python if win_python.exists() else posix_python
            python_path = python_path.resolve()

        if not python_path.exists():
            raise FileNotFoundError(
                f"Python interpreter not found for skill runner: {python_path}. "
                "Pass PyRunner(python=...) or create a .venv in the skill directory."
            )
        if not python_path.is_file():
            raise FileNotFoundError(
                f"Python interpreter path is not a file: {python_path}"
            )
        return python_path


class BoundPyRunner:
    """Concrete Python runner tool bound to one skill directory."""

    def __init__(
        self,
        skill_name: str,
        skill_dir: Path,
        python: Path,
        timeout: int,
        allowed_dirs: Sequence[str],
        allow_arbitrary_code: bool = False,
    ):
        self._skill_name = skill_name
        self._skill_dir = skill_dir.resolve()
        self._python = python.resolve()
        self._timeout = timeout
        self._allowed_dirs = tuple(allowed_dirs)
        self._allow_arbitrary_code = allow_arbitrary_code

    @property
    def name(self) -> str:
        """Tool name: {skill_name}__run_python."""
        return f"{self._skill_name}__run_python"

    @property
    def description(self) -> str:
        dirs = ", ".join(self._allowed_dirs) if self._allowed_dirs else "the skill directory"
        return (
            f"Execute an existing Python script for the '{self._skill_name}' skill. "
            f"The working directory is fixed to the skill directory. Only .py "
            f"scripts under {dirs} may be executed."
        )

    def to_openai_tool(self) -> Dict[str, Any]:
        """Return the OpenAI function-calling schema for this runner."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script_path": {
                            "type": "string",
                            "description": (
                                "Relative path to a .py script in this skill, "
                                "for example 'scripts/dti_calc.py'."
                            ),
                        },
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional command-line arguments for the script.",
                        },
                        "stdin": {
                            "type": "string",
                            "description": "Optional standard input passed to the script.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": (
                                f"Optional timeout in seconds. Must be between 1 and {self._timeout}."
                            ),
                        },
                    },
                    "required": ["script_path"],
                },
            },
        }

    async def call(self, arguments: Dict[str, Any], **kwargs) -> str:
        """Execute a skill-local Python script asynchronously."""
        try:
            script = self._resolve_script(arguments.get("script_path", ""))
            args = self._validate_args(arguments.get("args", []))
            stdin = arguments.get("stdin", None)
            if stdin is not None and not isinstance(stdin, str):
                return "Error: 'stdin' must be a string when provided."
            timeout = self._validate_timeout(arguments.get("timeout", self._timeout))
        except ValueError as exc:
            return f"Error: {exc}"

        return await asyncio.to_thread(self._run, script, args, stdin, timeout)

    def _resolve_script(self, script_path: str) -> Path:
        if not isinstance(script_path, str) or not script_path.strip():
            raise ValueError("'script_path' is required.")

        raw = Path(script_path)
        if raw.is_absolute():
            raise ValueError("'script_path' must be a relative path inside the skill directory.")

        target = (self._skill_dir / raw).resolve()
        try:
            target.relative_to(self._skill_dir)
        except ValueError as exc:
            raise ValueError("Path traversal is not allowed for 'script_path'.") from exc

        if self._allowed_dirs and not any(
            self._is_under_allowed_dir(target, allowed)
            for allowed in self._allowed_dirs
        ):
            allowed = ", ".join(self._allowed_dirs)
            raise ValueError(f"Script must be under one of the allowed dirs: {allowed}.")

        if target.suffix.lower() != ".py":
            raise ValueError("Only .py scripts can be executed by PyRunner.")
        if not target.exists():
            raise ValueError(f"Script not found: {script_path}")
        if not target.is_file():
            raise ValueError(f"Script path is not a file: {script_path}")
        return target

    def _is_under_allowed_dir(self, target: Path, allowed_dir: str) -> bool:
        allowed = Path(allowed_dir)
        if allowed.is_absolute():
            return False
        allowed_root = (self._skill_dir / allowed).resolve()
        return target == allowed_root or allowed_root in target.parents

    @staticmethod
    def _validate_args(raw_args: Any) -> List[str]:
        if raw_args is None:
            return []
        if not isinstance(raw_args, list):
            raise ValueError("'args' must be an array of strings.")
        if not all(isinstance(arg, str) for arg in raw_args):
            raise ValueError("'args' must be an array of strings.")
        return raw_args

    def _validate_timeout(self, raw_timeout: Any) -> int:
        if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, int):
            raise ValueError("'timeout' must be an integer number of seconds.")
        if raw_timeout <= 0:
            raise ValueError("'timeout' must be greater than 0.")
        if raw_timeout > self._timeout:
            raise ValueError(f"'timeout' must not exceed {self._timeout} seconds.")
        return raw_timeout

    def _run(self, script: Path, args: List[str], stdin: Optional[str], timeout: int) -> str:
        command = [str(self._python), str(script), *args]
        try:
            completed = subprocess.run(
                command,
                input=stdin,
                text=True,
                capture_output=True,
                cwd=str(self._skill_dir),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return (
                f"Timed out: true\n"
                f"Timeout seconds: {timeout}\n"
                f"Exit code: timeout\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}"
            )
        except Exception as exc:
            return f"Error executing script: {exc}"

        return (
            f"Timed out: false\n"
            f"Exit code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    def __repr__(self) -> str:
        return (
            f"BoundPyRunner(skill='{self._skill_name}', "
            f"python='{self._python}', dir='{self._skill_dir}')"
        )
