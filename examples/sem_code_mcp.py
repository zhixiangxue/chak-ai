"""Find code entities using a local Sem MCP server, without a model API key.

Install Sem separately (https://github.com/Ataraxy-Labs/sem; tested with 0.27.0).
From this repository root:
    python -m pip install -e . "mcp>=1.9,<2"
    python examples/sem_code_mcp.py /absolute/path/to/repo parse_config
    python examples/sem_code_mcp.py /absolute/path/to/repo parse_config --context

Only sem_find is selected. Sem can create index caches; repository selection is
not a filesystem sandbox. Use a trusted checkout. No source edits or validation
commands are requested by this example.
"""

import argparse
import asyncio
from pathlib import Path
from shutil import which

from chak.tools.mcp import Server


async def run(args: argparse.Namespace) -> None:
    repo = Path(args.repo).expanduser().resolve(strict=True)
    if not repo.is_dir():
        raise ValueError("Repository must be a directory")
    executable = which("sem")
    if executable is None:
        raise RuntimeError("Install Sem and put the sem executable on PATH first")
    server = Server(
        command=executable,
        args=["mcp"],
        env={
            "SEM_REPO": str(repo),
            "SEM_NO_NETWORK": "1",
            "SEM_NO_TELEMETRY": "1",
        },
    )
    tools = await server.tools(["sem_find"])
    if not tools:
        raise RuntimeError("Sem MCP did not advertise sem_find")
    arguments = {"query": args.query}
    if args.context:
        arguments.update(mode="context", token_budget=1200)
    result = await tools[0].call(arguments, retry=False)
    for block in result.content:
        if block.type == "text":
            print(block.text)
    if result.isError:
        raise RuntimeError("Sem entity lookup failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Trusted local source repository")
    parser.add_argument("query", help="Symbol name to look up")
    parser.add_argument("--context", action="store_true", help="Read bounded source context")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
