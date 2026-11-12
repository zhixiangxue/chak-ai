"""
Sandbox: Built-in cloud sandbox execution tool for chak

Executes code in an isolated cloud sandbox (pure shell mode).
One-shot design: spin up → write files → run shell script → return output → destroy.

Providers:
    e2b (default): an e2b cloud sandbox via e2b.Sandbox
    boxd:          an isolated boxd machine (https://boxd.sh/?ref=chak-ai)

Usage:
    from chak.tools.std import Sandbox

    sb = Sandbox()                    # e2b
    sb = Sandbox(provider="boxd")     # boxd
    conv = Conversation(model, tools=[sb])

Requires:
    e2b:  e2b package (pip install e2b) and E2B_API_KEY
    boxd: boxd package (pip install boxd) and BOXD_API_KEY
    (or api_key passed to constructor)
"""

import asyncio
import os
import posixpath
import shlex
import uuid
from typing import Any, Dict, List, Optional

_PROVIDERS = ("e2b", "boxd")
_SCRIPT_PATH = "/tmp/_chak_script.sh"
# Extra seconds a boxd machine may live beyond `timeout` (startup and file
# writes) before boxd destroys it on its own, in case cleanup never runs.
_BOXD_TTL_MARGIN = 600

# Prepended to the shell script on boxd. Machines run Ubuntu as a non-root
# user with sudo: let `pip install` fall back to a user install (PEP 668)
# and put the user's script directory on PATH.
_BOXD_PRELUDE = (
    "export PIP_BREAK_SYSTEM_PACKAGES=1\n"
    'export PATH="$HOME/.local/bin:$PATH"\n'
)


class Sandbox:
    """Built-in cloud sandbox execution tool (e2b or boxd).

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

    The same call runs on an isolated boxd machine with
    ``Sandbox(provider="boxd")``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 120,
        provider: str = "e2b",
        org: Optional[str] = None,
        region: Optional[str] = None,
        name_prefix: str = "chak-sandbox-",
    ):
        """
        Args:
            api_key: API key for the provider.  Falls back to the
                     ``E2B_API_KEY`` (e2b) or ``BOXD_API_KEY`` (boxd)
                     environment variable when not provided.
            timeout: Maximum seconds to wait for the shell script to finish.
                     Defaults to 120.
            provider: ``"e2b"`` (default) or ``"boxd"``.
            org:     boxd only.  Org to create the machine in.  Falls back
                     to ``BOXD_ORG``; when unset, the API key's org is used.
            region:  boxd only.  ``"fra"`` or ``"nyc"``.  Falls back to
                     ``BOXD_REGION``; when unset, boxd picks the region.
            name_prefix: boxd only.  Prefix for the machine name; a random
                     suffix is added.
        """
        provider = (provider or "").lower()
        if provider not in _PROVIDERS:
            raise ValueError(
                f"Unknown sandbox provider {provider!r}. "
                f"Expected one of: {', '.join(_PROVIDERS)}"
            )
        self._provider = provider
        if provider == "boxd":
            self._api_key = api_key or os.getenv("BOXD_API_KEY")
        else:
            self._api_key = api_key or os.getenv("E2B_API_KEY")
        self._timeout = timeout
        self._org = org or os.getenv("BOXD_ORG") or None
        self._region = region or os.getenv("BOXD_REGION") or None
        self._name_prefix = name_prefix

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
        if self._provider == "boxd":
            return self._run_boxd(shell_script, files)
        return self._run_e2b(shell_script, files)

    def _run_e2b(self, shell_script: str, files: List[Dict[str, str]]) -> str:
        """Run the task in an e2b sandbox."""
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

    def _run_boxd(self, shell_script: str, files: List[Dict[str, str]]) -> str:
        """Run the task on a fresh isolated boxd machine, then delete it."""
        try:
            from boxd import Boxd  # pyright: ignore[reportMissingTypeStubs]
        except ImportError:
            return "Error: 'boxd' package is not installed. Run: pip install boxd"

        if not self._api_key:
            return (
                "Error: boxd API key not found. "
                "Set BOXD_API_KEY environment variable or pass api_key to Sandbox()."
            )

        machine_id: Optional[str] = None
        try:
            with Boxd(api_key=self._api_key, region=self._region) as client:
                try:
                    # Isolated: LLM-written code runs with no access to other
                    # machines on the account.  The auto-destroy timer removes
                    # the machine even if this process dies before cleanup.
                    machine = client.machines.create(
                        f"{self._name_prefix}{uuid.uuid4().hex[:12]}",
                        org=self._org,
                        isolated=True,
                        auto_destroy_timeout=int(self._timeout) + _BOXD_TTL_MARGIN,
                    )
                    machine_id = machine.id
                    client.machines.wait_until_ready(machine_id)

                    # 1. Create parent directories the machine user can write
                    #    to (e.g. /project), then write source files
                    paths = [f.get("path", "") for f in files if f.get("path")]
                    dirs = sorted({posixpath.dirname(p) for p in paths} - {"", "/"})
                    if dirs:
                        mkdir = client.machines.exec(
                            machine_id,
                            f"for d in {' '.join(shlex.quote(d) for d in dirs)}; do "
                            'mkdir -p -- "$d" 2>/dev/null || '
                            '{ sudo mkdir -p -- "$d" && sudo chown "$(id -u):$(id -g)" -- "$d"; } || exit 1; '
                            "done",
                            timeout=self._timeout,
                        )
                        if mkdir.exit_code != 0:
                            detail = (mkdir.stderr or mkdir.stdout or "").strip()
                            return (
                                "Error running sandbox: could not create directories "
                                f"(exit code {mkdir.exit_code}): {detail}"
                            )
                    for f in files:
                        path = f.get("path", "")
                        if path:
                            client.machines.fs.write(machine_id, path, f.get("content", ""))

                    # 2. Write the shell script to a file and execute via bash
                    client.machines.fs.write(
                        machine_id,
                        _SCRIPT_PATH,
                        "#!/bin/bash\nset -e\n" + _BOXD_PRELUDE + shell_script,
                    )
                    result = client.machines.exec(
                        machine_id,
                        f"bash {_SCRIPT_PATH}",
                        timeout=self._timeout,
                    )

                    output = result.stdout or ""
                    if result.stderr:
                        output += "\n[stderr]\n" + result.stderr
                    if result.exit_code != 0:
                        output += f"\n[exit code {result.exit_code}]"
                    return output.strip() or "(no output)"
                finally:
                    if machine_id is not None:
                        try:
                            client.machines.delete(machine_id)
                        except Exception as exc:
                            from ...utils.logger import logger  # noqa: PLC0415

                            logger.warning(
                                f"[Sandbox] failed to delete boxd machine {machine_id}: {exc}"
                            )

        except Exception as exc:
            return f"Error running sandbox: {exc}"

    def __repr__(self) -> str:
        key = "set" if self._api_key else "missing"
        return f"<Sandbox provider={self._provider} api_key={key} timeout={self._timeout}>"
