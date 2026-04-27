"""
Built-in Standard Tools (std) — Demo

Demonstrates all 8 chak built-in tools in one file.
Each tool has a dedicated demo function; pick one via the command-line argument.

Tools:
    bash        - Execute shell commands on the local machine
    python      - Execute Python code snippets on the local machine
    filesystem  - Read/write/edit files and directories
    http        - HTTP client (GET/POST/PUT/PATCH/DELETE)
    search      - Web search (Tavily → Brave → DuckDuckGo fallback)
    web         - Fetch and extract readable content from web pages
    pdf         - Extract text/tables from PDF files (local path or URL)
    sandbox     - Run multi-file code projects in an isolated e2b cloud sandbox

Usage:
    python examples/tool_calling_std.py bash
    python examples/tool_calling_std.py python
    python examples/tool_calling_std.py filesystem
    python examples/tool_calling_std.py http
    python examples/tool_calling_std.py search
    python examples/tool_calling_std.py web
    python examples/tool_calling_std.py pdf
    python examples/tool_calling_std.py sandbox
    python examples/tool_calling_std.py all

Prerequisites:
    ANTHROPIC_API_KEY — or swap model to any supported provider
    E2B_API_KEY       — required for the sandbox demo only (https://e2b.dev)

Optional (for richer search/web results):
    TAVILY_API_KEY   — Tavily AI search (search demo)
    BRAVE_API_KEY    — Brave Search (search demo)
"""

import asyncio
import os
import tempfile

import dotenv

dotenv.load_dotenv()

import chak
from chak.tools.std import Bash, Python, FileSystem, Http, Search, Web, Pdf, Sandbox

# Default model — swap to any chak-supported provider/model string.
_MODEL = "anthropic/claude-sonnet-4-6"
_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------

async def demo_bash():
    """Execute shell commands on the local machine.

    The LLM can use bash to install packages, inspect the environment,
    or run any system-level command it needs.
    """
    conv = chak.Conversation(_MODEL, api_key=_API_KEY, tools=[Bash()])
    response = await conv.asend(
        "Check the current Python version and list the top-level files "
        "in the current directory. Show both results."
    )
    print(response.content)


# ---------------------------------------------------------------------------
# python
# ---------------------------------------------------------------------------

async def demo_python():
    """Execute Python code snippets directly on the local machine.

    Useful when the LLM wants to run a short computation or prototype
    without creating a persistent script file.
    """
    conv = chak.Conversation(_MODEL, api_key=_API_KEY, tools=[Python()])
    response = await conv.asend(
        "Write and run Python code that computes the first 15 Fibonacci numbers "
        "and prints them as a comma-separated list."
    )
    print(response.content)


# ---------------------------------------------------------------------------
# filesystem
# ---------------------------------------------------------------------------

async def demo_filesystem():
    """Read, write, and manage files in a sandboxed tmp directory.

    The workdir is printed so you can open the generated file and inspect it.
    """
    workdir = os.path.join(tempfile.gettempdir(), "chak_fs_demo")
    os.makedirs(workdir, exist_ok=True)
    print(f"Working directory: {workdir}\n")

    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        tools=[FileSystem(workdir=workdir)],
    )
    response = await conv.asend(
        f"Do the following steps in order:\n"
        f"1. Write a file called 'notes.txt' with three interesting facts about Python.\n"
        f"2. Read the file back and confirm its contents.\n"
        f"3. Append a fourth fact by editing the file.\n"
        f"4. Show the final file content.\n"
        f"All paths are relative to the working directory."
    )
    print(response.content)
    print(f"\nGenerated file is at: {os.path.join(workdir, 'notes.txt')}")


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------

async def demo_http():
    """Make HTTP requests to a public API.

    Uses httpbin.org which echoes back request details — great for demos.
    """
    conv = chak.Conversation(_MODEL, api_key=_API_KEY, tools=[Http()])
    response = await conv.asend(
        "Do two HTTP calls:\n"
        "1. GET https://httpbin.org/json and show the JSON it returns.\n"
        "2. POST to https://httpbin.org/post with JSON body "
        '{"tool": "chak", "version": "demo"} and show the response.'
    )
    print(response.content)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

async def demo_search():
    """Search the web using DuckDuckGo (no API key needed).

    Pass TAVILY_API_KEY or BRAVE_API_KEY env vars to upgrade to a
    higher-quality search backend.
    """
    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        tools=[Search(
            tavily_key=os.getenv("TAVILY_API_KEY"),
            brave_key=os.getenv("BRAVE_API_KEY"),
            max_results=5,
        )],
    )
    response = await conv.asend(
        "Search for 'chak python llm agent framework' and summarise "
        "the top 3 results in a few bullet points."
    )
    print(response.content)


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------

async def demo_web():
    """Fetch and extract readable content from a public web page."""
    conv = chak.Conversation(_MODEL, api_key=_API_KEY, tools=[Web()])
    response = await conv.asend(
        "Fetch the page https://quotes.toscrape.com and list the first 5 quotes "
        "with their authors."
    )
    print(response.content)


# ---------------------------------------------------------------------------
# pdf
# ---------------------------------------------------------------------------

async def demo_pdf():
    """Extract text and structure from a remote PDF.

    The PDF is fetched directly from a public URL — no local file needed.
    Requires: pip install pymupdf4llm
    """
    pdf_url = (
        "https://wcbpub.oss-cn-hangzhou.aliyuncs.com/xue/xxxx/"
        "Effective-harnesses-for-long-running-agents.pdf"
    )
    conv = chak.Conversation(_MODEL, api_key=_API_KEY, tools=[Pdf()])
    response = await conv.asend(
        f"Read the PDF at {pdf_url} and give me:\n"
        "1. The title and author(s) if mentioned.\n"
        "2. A 3-sentence summary of the main argument.\n"
        "3. The key techniques or concepts introduced."
    )
    print(response.content)


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------

async def demo_sandbox():
    """Run a multi-file Python project in an isolated e2b cloud sandbox.

    Requires: pip install e2b  and  E2B_API_KEY environment variable.
    The sandbox is a fresh Linux VM — nothing persists between calls.
    """
    e2b_key = os.getenv("E2B_API_KEY")
    if not e2b_key:
        print("E2B_API_KEY not set. Get a free key at https://e2b.dev")
        return

    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        tools=[Sandbox(api_key=e2b_key, timeout=300)],
    )
    response = await conv.asend(
        "Scrape https://quotes.toscrape.com and print all quotes with their "
        "authors and tags. Use code to do it."
    )
    print(response.content)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEMOS = {
    "bash": demo_bash,
    "python": demo_python,
    "filesystem": demo_filesystem,
    "http": demo_http,
    "search": demo_search,
    "web": demo_web,
    "pdf": demo_pdf,
    "sandbox": demo_sandbox,
}


async def demo_all():
    """Run every demo in sequence and report pass/fail for each."""
    results: list[tuple[str, bool, str]] = []
    for name, fn in _DEMOS.items():
        print(f"\n{'=' * 60}")
        print(f"  DEMO: {name}")
        print(f"{'=' * 60}")
        try:
            await fn()
            results.append((name, True, ""))
        except Exception as exc:
            print(f"[ERROR] {exc}")
            results.append((name, False, str(exc)))

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        suffix = f"  — {err}" if err else ""
        print(f"  [{status}] {name}{suffix}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or (sys.argv[1] not in _DEMOS and sys.argv[1] != "all"):
        print("Usage: python examples/tool_calling_std.py <tool>")
        print(f"Available: {', '.join(_DEMOS)}, all")
        sys.exit(1)

    if sys.argv[1] == "all":
        asyncio.run(demo_all())
    else:
        asyncio.run(_DEMOS[sys.argv[1]]())
