"""
Bash: Built-in shell execution tool for chak

Executes shell commands and returns their combined stdout/stderr output.
Handles cross-platform compatibility automatically:
  - pip <cmd> → routed through the active venv interpreter
  - mkdir -p  → stripped to plain mkdir (Windows already handles parent dirs)
"""

import asyncio
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


class Bash:
    """Built-in shell execution tool.

    Executes arbitrary shell commands and returns their output.  Designed to
    give an LLM agent the ability to install packages, create directories, and
    perform other system-level operations needed to unblock task execution.

    Cross-platform notes (Windows):
      - ``pip install <pkg>`` is automatically routed through the active venv
        Python interpreter so the package lands in the correct environment.
      - ``mkdir -p`` is silently normalised to ``mkdir`` because the Windows
        built-in already creates intermediate directories.

    Example::

        bash = Bash()
        result = await bash.call({"command": "pip install pdf2image"})
    """

    def __init__(self, timeout: int = 60, venv_python: str = None):
        """
        Args:
            timeout: Maximum seconds to wait for a command before raising a
                     timeout error.  Defaults to 60.
            venv_python: Path to the Python executable to use when routing pip
                         commands.  Defaults to ``sys.executable`` (the
                         interpreter running the current process).
        """
        self._timeout = timeout
        self._python = venv_python or sys.executable

    # ------------------------------------------------------------------
    # Duck-typing interface required by ToolManager
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Tool name exposed to the LLM."""
        return "bash"

    @property
    def description(self) -> str:
        """Short description shown to the LLM."""
        return (
            "Execute a shell command and return its stdout and stderr output. "
            "Use this to install missing Python packages "
            "(e.g., 'pip install pdf2image'), create directories "
            "('mkdir /path/to/dir'), or perform other system-level operations "
            "needed to unblock task execution."
        )

    def to_openai_tool(self) -> Dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }

    async def call(self, arguments: Dict[str, Any], **kwargs) -> str:
        """Execute the shell command asynchronously.

        Args:
            arguments: Dict with a single key ``"command"`` whose value is the
                       shell command string to run.
            **kwargs: Ignored (present for interface compatibility).

        Returns:
            Combined stdout + stderr output, or an error message string.
        """
        command = arguments.get("command", "").strip()
        # Run blocking subprocess in a thread so we don't block the event loop
        return await asyncio.to_thread(self._run, command)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Regex that matches: mkdir [-p] [optional-quotes] <path> [optional-quotes]
    _MKDIR_RE = re.compile(
        r'^mkdir(?:\s+-p)?\s+["\']?(.+?)["\']?\s*$',
        re.IGNORECASE,
    )

    def _run(self, command: str) -> str:
        """Execute *command* synchronously and return its output.

        Called from a worker thread via :func:`asyncio.to_thread`.
        """
        # Route pip/python through the active venv interpreter so packages
        # install into—and scripts run inside—the correct environment.
        if command.startswith("pip "):
            command = f"{self._python} -m {command}"
        elif command.startswith("python ") or command == "python":
            command = f"{self._python} {command[len('python'):]}"

        # Handle mkdir cross-platform via pathlib so that forward-slash paths,
        # 'mkdir -p', and quoting all work correctly on Windows and Unix alike.
        m = self._MKDIR_RE.match(command)
        if m:
            path_str = m.group(1).strip('"').strip("'")
            try:
                Path(path_str).mkdir(parents=True, exist_ok=True)
                return f"Directory created: {path_str}"
            except Exception as exc:
                return f"Error creating directory: {exc}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {self._timeout} seconds"
        except Exception as exc:
            return f"Error running command: {exc}"
