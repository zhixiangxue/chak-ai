"""
Sandbox: Built-in e2b cloud sandbox execution tool for chak

Executes code in an isolated e2b cloud sandbox (pure shell mode via e2b.Sandbox).
One-shot design: spin up → write files → run shell script → return output → destroy.

Usage:
    from chak.tools.std import Sandbox

    sb = Sandbox()
    conv = Conversation(model, tools=[sb])

Requires:
    e2b package (pip install e2b)
    E2B_API_KEY environment variable (or api_key passed to constructor)
"""

import asyncio
import os
from typing import Any, Dict, List, Optional


class Sandbox:
    """Built-in e2b cloud sandbox execution tool.

    Writes source files into an isolated cloud sandbox, then runs a bash
    shell script that controls the full execution sequence (install packages,
    run code, etc.).  The sandbox is created fresh for each call and
    destroyed afterwards.

    Example::

        sb = Sandbox()
        result = await sb.call({
            "shell_script": "pip install httpx -q\\npython /project/main.py",
            "files": [{"path": "/project/main.py", "content": "import httpx; print(httpx.get('https://example.com').status_code)"}]
        })
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Args:
            api_key: E2B API key.  Falls back to the ``E2B_API_KEY``
                     environment variable when not provided.
            timeout: Maximum seconds to wait for the shell script to finish.
                     Defaults to 120.
        """
        self._api_key = api_key or os.getenv("E2B_API_KEY")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Duck-typing interface required by ToolManager
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Tool name exposed to the LLM."""
        return "sandbox"

    @property
    def description(self) -> str:
        """Short description shown to the LLM."""
        return (
            "Execute code in an isolated cloud sandbox and return its output. "
            "IMPORTANT: Call this tool EXACTLY ONCE per task — never in parallel or sequentially. "
            "Each call creates a completely fresh environment; files do NOT persist between calls. "
            "The entire pipeline (install packages, write files, run code) must be done in a single call. "
            "Write all source files via 'files', then run a bash shell script via 'shell_script'. "
            "Supports any language reachable from bash (Python, Node, Ruby, etc.)."
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
                        "shell_script": {
                            "type": "string",
                            "description": (
                                "Bash shell script to execute inside the sandbox. "
                                "Each line is a shell command — install packages, "
                                "create directories, run code, etc. "
                                "All lines share the same bash session. "
                                "Example: \"pip install httpx -q\\npython /project/main.py\""
                            ),
                        },
                        "files": {
                            "type": "array",
                            "description": (
                                "Source files to write into the sandbox before "
                                "the shell script runs. "
                                "Provide the full content of each file here — "
                                "do NOT use heredocs in shell_script for large files."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {
                                        "type": "string",
                                        "description": "Absolute path inside the sandbox, e.g. /project/main.py",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Full source code content of the file.",
                                    },
                                },
                                "required": ["path", "content"],
                            },
                        },
                    },
                    "required": ["shell_script"],
                },
            },
        }

    async def call(self, arguments: Dict[str, Any], **kwargs) -> str:
        """Execute the sandbox task asynchronously.

        Args:
            arguments: Dict with ``shell_script`` (required) and optionally
                       ``files`` (list of ``{path, content}`` dicts).
            **kwargs: Ignored (present for interface compatibility).

        Returns:
            Combined stdout + stderr output, or an error string on failure.
        """
        shell_script: str = arguments.get("shell_script", "").strip()
        files: List[Dict[str, str]] = arguments.get("files") or []

        return await asyncio.to_thread(self._run, shell_script, files)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, shell_script: str, files: List[Dict[str, str]]) -> str:
        """Spin up sandbox, write files, run shell script.

        Called from a worker thread via :func:`asyncio.to_thread`.
        """
        try:
            from e2b import Sandbox as E2BSandbox  # pyright: ignore[reportMissingTypeStubs]
        except ImportError:
            return "Error: 'e2b' package is not installed. Run: pip install e2b"

        if not self._api_key:
            return (
                "Error: E2B API key not found. "
                "Set E2B_API_KEY environment variable or pass api_key to Sandbox()."
            )

        try:
            with E2BSandbox.create(api_key=self._api_key) as sbx:
                # 1. Write source files
                for f in files:
                    path = f.get("path", "")
                    content = f.get("content", "")
                    if path:
                        sbx.files.write(path, content)  # pyright: ignore[reportUnknownMemberType]

                # 2. Write the shell script to a file and execute via bash
                sbx.files.write(  # pyright: ignore[reportUnknownMemberType]
                    "/tmp/_chak_script.sh",
                    "#!/bin/bash\nset -e\n" + shell_script,
                )
                result = sbx.commands.run(
                    "bash /tmp/_chak_script.sh",
                    timeout=self._timeout,
                )

                output = result.stdout or ""
                if result.stderr:
                    output += "\n[stderr]\n" + result.stderr
                return output.strip() or "(no output)"

        except Exception as exc:
            return f"Error running sandbox: {exc}"

    def __repr__(self) -> str:
        key = "set" if self._api_key else "missing"
        return f"<Sandbox api_key={key} timeout={self._timeout}>"
