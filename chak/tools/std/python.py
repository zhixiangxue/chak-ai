"""
Python: Built-in Python code execution tool for chak

Executes arbitrary Python code strings in a temporary script file using the
active venv interpreter.  Complements Bash for cases where the LLM wants to
run inline code snippets (e.g. from skill documentation) without writing a
separate script file first.

NOTE: This tool is a **subprocess script runner**, NOT a Jupyter / REPL.
The model frequently writes Jupyter-style code (``result = expr`` with no
``print``) expecting the trailing expression to be auto-echoed, gets back
``(no output)``, then loops trying to "fix" it — burning huge amounts of
tokens (a single report observed 1.5M tokens wasted before hitting
max_iterations).  We mitigate this on three layers:
  1. ``description`` tells the model upfront that output comes from stdout
     and bare expressions are NOT echoed.
  2. When stdout is empty, return an instructive message instead of the
     ambiguous ``(no output)``.
  3. ``_auto_print_last_expr`` rewrites the AST so that the last top-level
     expression is implicitly ``print(repr(...))``-ed, matching the
     Jupyter mental model and removing the foot-gun entirely.
"""

import asyncio
import ast
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

    def __init__(self, timeout: int = 60, venv_python: str = None, auto_print_last_expr: bool = True):
        """
        Args:
            timeout: Maximum seconds to wait before raising a timeout error.
                     Defaults to 60.
            venv_python: Path to the Python interpreter.  Defaults to
                         ``sys.executable`` (the interpreter running the
                         current process).
            auto_print_last_expr: If True (default), automatically wrap the
                         last top-level bare expression in ``print(repr(...))``
                         so Jupyter-style scripts produce visible output.
                         Disable for strict subprocess semantics.
        """
        self._timeout = timeout
        self._python = venv_python or sys.executable
        self._auto_print_last_expr = auto_print_last_expr
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
        """Short description shown to the LLM.
    
        Explicitly warns that this is a subprocess script runner, not a
        REPL/Jupyter cell: bare expressions are NOT echoed and stdout is
        the only output channel.
        """
        return (
            "Execute a Python code snippet as a one-shot subprocess script "
            "(NOT a REPL / Jupyter cell) and return its captured stdout+stderr. "
            "You MUST `print(...)` any value you want to inspect \u2014 bare "
            "expressions like `result` or `a + b` are NOT auto-echoed and will "
            "produce empty output. Example: write `print(my_var)`, never just "
            "`my_var`. Use this to run inline code examples directly without "
            "creating a separate script file."
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
        if self._auto_print_last_expr:
            code = self._auto_print_last_expr_transform(code)
        return await asyncio.to_thread(self._run, code)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_print_last_expr_transform(code: str) -> str:
        """Rewrite *code* so the last top-level bare expression is printed.

        Mirrors Jupyter / IPython behaviour: if the script ends in an
        expression statement (``ast.Expr``) whose value is not already a
        ``print(...)`` call, wrap it as ``print(repr(<expr>))`` and emit a
        one-line stderr hint so the model learns the convention.

        Falls back to the original code when:
          - the source has a SyntaxError (let the subprocess surface it),
          - the module body is empty,
          - the last statement is not a bare expression,
          - the expression is already a ``print(...)`` / ``pprint(...)`` /
            ``display(...)`` / docstring literal.
        """
        if not code or not code.strip():
            return code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code  # let the subprocess produce a normal traceback
        if not tree.body:
            return code
        last = tree.body[-1]
        if not isinstance(last, ast.Expr):
            return code
        value = last.value
        # Skip docstring-only modules (`"""..."""`).
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return code
        # Skip if already a print-ish call.
        if isinstance(value, ast.Call):
            func = value.func
            fname = None
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            if fname in {"print", "pprint", "display", "pp"}:
                return code
        # Rewrite: `<expr>`  =>  `print(repr(<expr>))` with a stderr hint.
        wrapped = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Name(id="repr", ctx=ast.Load()),
                        args=[value],
                        keywords=[],
                    )
                ],
                keywords=[],
            )
        )
        ast.copy_location(wrapped, last)
        ast.copy_location(wrapped.value, last)
        tree.body[-1] = wrapped
        try:
            new_source = ast.unparse(ast.fix_missing_locations(tree))
        except Exception:
            return code  # never break the user; fall back to original
        # Prefix a stderr hint so the LLM learns it shouldn't rely on this.
        hint = (
            "import sys as _chak_sys; "
            "print('[chak] auto-printed trailing expression via repr(); "
            "use print(...) explicitly next time', file=_chak_sys.stderr)\n"
        )
        return new_source + "\n" + hint

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
            stripped = output.strip()
            if stripped:
                return stripped
            # Empty stdout is the #1 source of the "50-iteration retry loop"
            # failure mode — give the model an actionable hint instead of an
            # ambiguous "(no output)".
            return (
                "(no stdout captured) The code executed successfully and the "
                "process exited cleanly, but nothing was written to stdout. "
                "This tool is a one-shot subprocess, NOT a REPL: bare "
                "expressions are not echoed. If you wanted to see a value, "
                "wrap it in `print(...)` (e.g. `print(result)`) and call the "
                "tool again. Do NOT retry with the same code expecting "
                "different output."
            )
        except subprocess.TimeoutExpired:
            return f"Error: code timed out after {self._timeout} seconds"
        except Exception as exc:
            return f"Error executing code: {exc}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
