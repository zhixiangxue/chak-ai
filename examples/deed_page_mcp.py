"""Publish an HTML page to a public URL with the deed.page MCP server.

No account, API key, or LLM is required. From the repository root:
    python -m pip install -e . "mcp>=1.9,<2"
    python examples/deed_page_mcp.py publish                  # sample page, expires in 1 hour
    python examples/deed_page_mcp.py publish report.html --ttl 0
    python examples/deed_page_mcp.py delete <slug>

The first publish returns an owner token. It is saved to
~/.config/deed-page/token (mode 600) and reused for later publishes and
deletes, or read from DEEDPAGE_TOKEN. Published pages are public.
Docs: https://deed.page/docs
"""

import argparse
import asyncio
import json
import os
from importlib.metadata import version
from pathlib import Path

from chak.tools.mcp import Server

TOKEN_FILE = Path.home() / ".config" / "deed-page" / "token"
SAMPLE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Published with chak</title></head>
<body><h1>Hello from chak</h1><p>This page was published through MCP.</p></body></html>
"""


def load_token() -> str:
    token = os.environ.get("DEEDPAGE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    return token


def save_token(token: str) -> None:
    TOKEN_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    TOKEN_FILE.touch(mode=0o600, exist_ok=True)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)


async def run(args: argparse.Namespace) -> None:
    server = Server(
        url="https://deed.page/mcp",
        type="streamable-http",
        headers={
            "User-Agent": f"chak-ai/{version('chakpy')} (+https://github.com/zhixiangxue/chak-ai)"
        },
    )
    tool_name = "publish_site" if args.command == "publish" else "delete_site"
    tools = await server.tools([tool_name])
    if not tools:
        raise RuntimeError(f"deed.page MCP did not provide {tool_name}")

    token = load_token()
    if args.command == "publish":
        html = Path(args.file).read_text(encoding="utf-8") if args.file else SAMPLE_PAGE
        arguments = {"files": {"index.html": html}}
        if args.slug:
            arguments["slug"] = args.slug
        if args.ttl > 0:
            arguments["ttl_seconds"] = args.ttl
    else:
        if not token:
            raise SystemExit("delete needs the owner token (DEEDPAGE_TOKEN or ~/.config/deed-page/token)")
        arguments = {"slug": args.slug}
    if token:
        arguments["token"] = token

    result = await tools[0].call(arguments, retry=False)
    text = "".join(block.text for block in result.content if block.type == "text")
    if result.isError:
        raise RuntimeError(f"deed.page {tool_name} failed: {text}")
    try:
        data = json.loads(text)
    except ValueError:
        print(text)
        return
    if not data.get("ok", False):
        raise RuntimeError(f"deed.page {tool_name} failed: {text}")
    if data.get("token"):
        save_token(data.pop("token"))
        print(f"Owner token saved to {TOKEN_FILE}")
    keys = ("url", "slug", "version", "expires_at") if args.command == "publish" else ("ok", "slug")
    print(json.dumps({key: data[key] for key in keys if key in data}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish or delete a deed.page site over MCP")
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish", help="Publish an HTML file (or a sample page)")
    publish.add_argument("file", nargs="?", help="HTML file to publish as index.html")
    publish.add_argument("--slug", help="Subdomain to claim; random when omitted")
    publish.add_argument("--ttl", type=int, default=3600, help="Expiry in seconds; 0 keeps the page")
    delete = commands.add_parser("delete", help="Delete a site you own")
    delete.add_argument("slug")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
