"""
Python: Built-in Python code execution tool for chak

Executes arbitrary Python code strings in a temporary script file using the
active venv interpreter.  Complements Bash for cases where the LLM wants to
run inline code snippets (e.g. from skill documentation) without writing a
separate script file first.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict


class Python:
    """Built-in Python code execution tool.

    Executes a Python code string by writing it to a temporary file and
    running it with the active venv interpreter.  Returns combined
    stdout/stderr output.

    Use this when the LLM wants to run inline code snippets directly —
    for example, adapting a code example from skill documentation and
    executing it immediately without creating a persistent script file.

    For shell commands, package installation, or running pre-existing
    scripts, use Bash instead.

    .. warning::
        This tool executes **arbitrary Python code** with the same
        permissions as the host process.  The LLM can read files,
        write files, make network requests, and delete data.  Only use
        this tool in trusted environments.  For untrusted or multi-tenant
        deployments, supply a sandboxed interpreter via ``venv_python``
        or wrap the tool with chak's HITL approval mechanism.

    Example::

        py = Python()
        result = await py.call({"code": "print('hello')"})
    """

    def __init__(self, timeout: int = 60, venv_python: str = None):
        """
        Args:
            timeout: Maximum seconds to wait before raising a timeout error.
                     Defaults to 60.
            venv_python: Path to the Python interpreter.  Defaults to
                         ``sys.executable`` (the interpreter running the
                         current process).
        """
        self._timeout = timeout
        self._python = venv_python or sys.executable
        self._warn_unsafe()

    # ------------------------------------------------------------------
    # Duck-typing interface required by ToolManager
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Tool name exposed to the LLM."""
        return "python"

    @property
    def description(self) -> str:
        """Short description shown to the LLM."""
        return (
            "Execute a Python code snippet and return its output. "
            "Use this to run inline code examples directly without creating "
            "a separate script file — for example, adapting code from "
            "documentation and executing it immediately."
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
                        "code": {
                            "type": "string",
                            "description": "Python source code to execute.",
                        }
                    },
                    "required": ["code"],
                },
            },
        }

    async def call(self, arguments: Dict[str, Any], **kwargs) -> str:
        """Execute the Python code asynchronously.

        Args:
            arguments: Dict with a single key ``"code"`` containing the
                       Python source to run.
            **kwargs: Ignored (present for interface compatibility).

        Returns:
            Combined stdout + stderr output, or an error message string.
        """
        code = arguments.get("code", "")
        return await asyncio.to_thread(self._run, code)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _warn_unsafe() -> None:
        """Print a prominent security warning to stderr on instantiation."""
        msg = (
            "[CHAK WARNING] Python tool grants the LLM full host-process "
            "permissions (file I/O, network, subprocess). "
            "Use HITL or a sandboxed interpreter in production."
        )
        try:
            from rich.console import Console
            from rich.panel import Panel

            Console(stderr=True).print(
                Panel(msg, title="[bold red]Security Warning[/bold red]", border_style="red")
            )
        except ImportError:
            print(f"\n{'!' * 60}\n{msg}\n{'!' * 60}\n", file=sys.stderr)

    def _run(self, code: str) -> str:
        """Write *code* to a temp file and execute it synchronously.

        Called from a worker thread via :func:`asyncio.to_thread`.
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [self._python, tmp_path],
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
            return f"Error: code timed out after {self._timeout} seconds"
        except Exception as exc:
            return f"Error executing code: {exc}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
