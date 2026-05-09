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

    Safety:
      - A built-in deny list blocks the most destructive shell patterns
        (recursive deletes, disk formatting, fork bombs, system shutdown).
        Pass ``deny_patterns=[]`` to disable, or supply your own list.

    Example::

        bash = Bash()
        result = await bash.call({"command": "pip install pdf2image"})
    """

    # Default patterns for commands that should never be executed by an agent.
    _DEFAULT_DENY: list[str] = [
        r"\brm\s+-[rf]{1,2}\b",          # rm -r / rm -rf / rm -fr
        r"\bdel\s+/[fq]\b",              # del /f  del /q  (Windows)
        r"\brmdir\s+/s\b",               # rmdir /s  (Windows recursive delete)
        r"(?:^|[;&|]\s*)format\b",       # format C:  (standalone only)
        r"\b(mkfs|diskpart)\b",          # mkfs.*  diskpart  (disk operations)
        r"\bdd\s+if=",                   # dd if=… (raw disk write)
        r">\s*/dev/sd",                  # redirect to disk device
        r"\b(shutdown|reboot|poweroff)\b",  # system power commands
        r":\(\)\s*\{.*\};\s*:",          # fork bomb  :(){ :|:& };:
    ]

    # Sensitive filenames/patterns — blocked regardless of working_dir.
    _DEFAULT_SENSITIVE: list[str] = [
        r"\.env\b",              # .env  .env.local  .env.production
        r"\bid_rsa\b",           # SSH private key (RSA)
        r"\bid_ed25519\b",       # SSH private key (ed25519)
        r"\bauthorized_keys\b",  # SSH authorized_keys
        r"\.pem\b",              # certificates / private keys
        r"\.key\b",              # generic key files
        r"\.p12\b",              # PKCS#12
        r"\.pfx\b",              # PFX certificate
        r"\bcredentials\b",      # AWS credentials file
    ]

    def __init__(
        self,
        timeout: int = 60,
        venv_python: str = None,
        deny_patterns: list[str] | None = None,
        sensitive_files: list[str] | None = None,
        working_dir: str | None = None,
    ):
        r"""
        Args:
            timeout: Maximum seconds to wait for a command before raising a
                     timeout error.  Defaults to 60.
            venv_python: Path to the Python executable to use when routing pip
                         commands.  Defaults to ``sys.executable`` (the
                         interpreter running the current process).
            deny_patterns: Regex patterns for destructive commands (rm -rf etc.).
                           Defaults to ``Bash._DEFAULT_DENY``.  Pass ``[]`` to disable.
            sensitive_files: Regex patterns matched against the full command string
                             to block access to sensitive files (.env, id_rsa, etc.).
                             Always active.  Defaults to ``Bash._DEFAULT_SENSITIVE``.
                             Pass ``[]`` to disable.
            working_dir: If set, commands may not reference absolute paths outside
                         this directory, and path-traversal sequences (../ or ..\)
                         are blocked.  Defaults to ``None`` (no path restriction).
        """
        self._timeout = timeout
        self._python = venv_python or sys.executable
        self._deny_patterns = (
            deny_patterns if deny_patterns is not None else self._DEFAULT_DENY
        )
        self._sensitive_files = (
            sensitive_files if sensitive_files is not None else self._DEFAULT_SENSITIVE
        )
        self._working_dir = Path(working_dir).resolve() if working_dir else None

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

    def _guard(self, command: str) -> str | None:
        """Return an error string if *command* should be blocked, else None.

        Three independent checks (each can be disabled independently):

        1. deny_patterns  — destructive system commands (rm -rf, format, etc.)
        2. sensitive_files — commands that reference sensitive file names (.env, id_rsa, etc.)
        3. working_dir    — absolute paths outside the working directory / path traversal
        """
        lower = command.lower()

        # 1. Destructive command patterns
        for pattern in self._deny_patterns:
            if re.search(pattern, lower):
                return f"Error: command blocked — dangerous pattern: {pattern}"

        # 2. Sensitive file access
        for pattern in self._sensitive_files:
            if re.search(pattern, lower):
                return f"Error: command blocked — references a sensitive file: {pattern}"

        # 3. Path restriction (only when working_dir is configured)
        if self._working_dir:
            if "..\\" in command or "../" in command:
                return "Error: command blocked — path traversal detected"

            # Extract Windows-style absolute paths (e.g. C:\Users\...)
            win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]*", command)
            # Extract POSIX absolute paths (e.g. /home/...)
            posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command)

            for raw in win_paths + posix_paths:
                try:
                    target = Path(raw.strip()).resolve()
                except Exception:
                    continue
                if (
                    target != self._working_dir
                    and self._working_dir not in target.parents
                ):
                    return (
                        f"Error: command blocked — path '{raw.strip()}' is outside "
                        f"working directory '{self._working_dir}'"
                    )

        return None

    def _run(self, command: str) -> str:
        """Execute *command* synchronously and return its output.

        Called from a worker thread via :func:`asyncio.to_thread`.
        """
        # Route pip/python through the active venv interpreter so packages
        # install into—and scripts run inside—the correct environment.
        if guard_error := self._guard(command):
            return guard_error

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
