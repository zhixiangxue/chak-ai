"""
GitHub Project Runner Demo

Give the LLM a GitHub URL. It will:
  1. Clone the repo to a local directory
  2. Read the README and source files to understand the project
  3. Create a fresh virtual environment
  4. Install all dependencies
  5. Write a runnable example that exercises the project's main feature
  6. Execute it and print the output

Tools used: Bash + FileSystem
No human guidance after the initial prompt — the LLM figures everything out autonomously.
"""

import asyncio
import os
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import chak
from chak.tools.std import Bash, FileSystem
from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent

_MODEL = "anthropic/claude-sonnet-4-6"
_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

GITHUB_URL = "https://github.com/zhixiangxue/seeka-ai"
# Cloned repo lands in playground/output/seeka-ai by default.
# Override via env var:  DEMO_TARGET_DIR=/your/path
TARGET_DIR = os.getenv(
    "DEMO_TARGET_DIR",
    str(Path(__file__).parent / "output" / "seeka-ai"),
)
# .env file with API keys — look next to this file, then at the project root
_env_candidates = [
    str(Path(__file__).parent / ".env"),
    str(Path(__file__).parent.parent / ".env"),
]
_ENV_FILE = next((p for p in _env_candidates if Path(p).exists()), _env_candidates[-1])


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

async def main():
    # Extend timeout to 300s so pip install doesn't get cut off.
    # Remove rmdir /s from deny list so the LLM can delete and re-clone the repo.
    _deny = [p for p in Bash._DEFAULT_DENY if r"rmdir" not in p]
    bash = Bash(timeout=300, deny_patterns=_deny)
    fs = FileSystem()

    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        tools=[bash, fs],
    )
    # This task needs many steps; raise the cap well above the default 50.
    conv._tool_manager.max_iterations = 120

    prompt = f"""
You are an autonomous developer agent. Your mission is to take the following GitHub project,
understand it, and get it running — end to end, without any human help.

**GitHub URL**: {GITHUB_URL}
**Target directory**: {TARGET_DIR}

Follow these steps in order:

### Step 1 — Clone or update
If {TARGET_DIR} does NOT exist, clone the repository into it.
If {TARGET_DIR} already exists, just run `git pull` inside it — do NOT delete and re-clone.
Either way, confirm which commit/branch is checked out.

### Step 2 — Explore
Read the README (and any docs/ folder if present).
Then browse the source tree to understand:
- What does this project do?
- What is its main entry point / public API?
- What dependencies does it need?
- Are there any existing examples in the repo?

### Step 3 — Set up the environment
If {TARGET_DIR}\\.venv already exists, skip creation and reuse it.
Otherwise create a fresh Python virtual environment at {TARGET_DIR}\\.venv using the system Python.
Either way, verify the venv is functional and all dependencies are installed.
Prefer `pip install -e .` if pyproject.toml/setup.py is present; otherwise use requirements.txt.
If neither exists, inspect imports in the source and install what’s needed manually.

### Step 4 — Write an example
Write a Python script to {TARGET_DIR}\\run_demo.py that demonstrates the project's most
interesting capability. The example should:
- Be self-contained (all imports, config, and logic in one file)
- Show a realistic, non-trivial use case — not just "hello world"
- Include inline comments explaining what each section does
- Handle any required API keys via os.getenv() with a clear error message if missing

### Step 5 — Run it
Execute {TARGET_DIR}\\run_demo.py using the venv interpreter.
Show the full output.

### Step 6 — Report
Summarise what you found and what the demo does. If anything failed, explain why and
what a developer would need to do to fix it.

Important notes:
- Use the venv Python for all installs and execution:
  {TARGET_DIR}\\.venv\\Scripts\\python.exe
- If you encounter import errors, fix them before reporting failure.
- If the project needs API keys (LLM provider keys, embedding keys, etc.),
  read the file {_ENV_FILE} to find available keys.
  Load them via os.environ or dotenv before running the demo script.
- If the project needs API keys you cannot find in that .env file, write the
  example so it gracefully prints what keys are needed and exits cleanly.
- Do NOT investigate or debug sys.path, site-packages layout, or why a package
  resolves to an unexpected location. If `import X` succeeds, move on immediately.
  Only investigate if the actual import raises an error.
"""

    print("=" * 70)
    print(f"  GitHub Runner — {GITHUB_URL}")
    print("=" * 70)
    print()

    async for event in await conv.asend(prompt, event=True):
        match event:
            case MessageChunk(content=text) if text:
                print(text, end="", flush=True)

            case ToolCallStartEvent(tool_name=name, arguments=args):
                # Show a short hint of what the tool is about to do
                hint = ""
                if "command" in args:
                    hint = args["command"][:150]
                elif "path" in args:
                    hint = args["path"]
                elif "uri" in args:
                    hint = args["uri"]
                elif "sql" in args:
                    hint = args["sql"][:100]
                print(f"\n\n>>> [{name}] {hint}")

            case ToolCallSuccessEvent(tool_name=name, result=res):
                preview = res[:200].replace("\n", " ") if res else ""
                print(f"<<< {preview}")

            case ToolCallErrorEvent(tool_name=name, error=err):
                print(f"<<< ERROR: {err}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
