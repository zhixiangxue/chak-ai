"""Claude Agent Skill demo: multiple skills, each with its own isolated venv.

Replaces the old pattern of registering a global Bash/Python instance per
skill (which collides on tool name once 2+ skills are active at once). Each
ClaudeSkill now carries its own PyRunner, bound to that skill's interpreter,
and is exposed to the LLM as a uniquely-named companion tool:

    dti-calculator, dti-calculator__read_file, dti-calculator__run_python
    income-calc,    income-calc__read_file,    income-calc__run_python

No global Python/Bash tool is registered, so there is nothing for the two
skills' tool names to collide on.

This demo is self-contained: it creates two lightweight venvs on the fly
(stdlib ``venv`` module, no pip installs) so it can run without the reader
needing pre-existing per-skill virtual environments. In a real deployment,
each skill directory would already have its own ``.venv`` with the
dependencies that skill's scripts need, and ``PyRunner()`` (no arguments)
auto-discovers it.

Usage:
    python examples/tool_calling_claude_skill_multi_venv.py
    python examples/tool_calling_claude_skill_multi_venv.py openai/gpt-4o
"""

import argparse
import asyncio
import os
import venv
from pathlib import Path

import dotenv

import chak
from chak.tools.skills import ClaudeSkill, PyRunner

dotenv.load_dotenv()

_ROOT = Path(__file__).parent.parent
_DEMO_DIR = _ROOT / "tmp" / "multi_venv_skills_demo"


def _create_light_venv(venv_dir: Path) -> None:
    """Create a minimal venv (no pip) purely to demonstrate per-skill isolation."""
    if venv_dir.exists():
        return
    venv.create(str(venv_dir), with_pip=False, symlinks=False)


def _write_skill(skill_dir: Path, name: str, description: str, script_body: str) -> None:
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n"
        f"## Usage\nCall {name}__run_python with "
        f"{{\"script_path\": \"scripts/calc.py\", \"stdin\": \"<json>\"}}.\n",
        encoding="utf-8",
    )
    (scripts_dir / "calc.py").write_text(script_body, encoding="utf-8")


def _setup_demo_skills() -> tuple[Path, Path]:
    """Create two demo skill directories, each with its own venv."""
    dti_dir = _DEMO_DIR / "dti-calculator"
    income_dir = _DEMO_DIR / "income-calc"

    _write_skill(
        dti_dir,
        name="dti-calculator",
        description="Calculate debt-to-income ratio from monthly income and debts.",
        script_body=(
            "import json, sys\n"
            "data = json.loads(sys.stdin.read() or '{}')\n"
            "income = data.get('monthly_income', 0)\n"
            "debts = data.get('monthly_debts', 0)\n"
            "dti = (debts / income) if income else 0\n"
            "print(json.dumps({'dti_ratio': round(dti, 4), 'interpreter': sys.executable}))\n"
        ),
    )
    _write_skill(
        income_dir,
        name="income-calc",
        description="Normalize a list of income entries into a monthly average.",
        script_body=(
            "import json, sys\n"
            "data = json.loads(sys.stdin.read() or '{}')\n"
            "entries = data.get('entries', [])\n"
            "avg = (sum(entries) / len(entries)) if entries else 0\n"
            "print(json.dumps({'monthly_average': round(avg, 2), 'interpreter': sys.executable}))\n"
        ),
    )

    _create_light_venv(dti_dir / ".venv")
    _create_light_venv(income_dir / ".venv")
    return dti_dir, income_dir


async def main(model_uri: str):
    provider = model_uri.split("/")[0].upper()
    api_key = os.getenv(f"{provider}_API_KEY", "")
    if not api_key:
        raise ValueError(f"Please set {provider}_API_KEY in .env")

    dti_dir, income_dir = _setup_demo_skills()

    # Each skill gets its own PyRunner. PyRunner() with no arguments
    # auto-discovers <skill_dir>/.venv/{Scripts,bin}/python — this is the
    # per-skill interpreter isolation the old global Python/Bash pattern
    # could not express without name collisions.
    tools = [
        ClaudeSkill(str(dti_dir), runner=PyRunner()),
        ClaudeSkill(str(income_dir), runner=PyRunner()),
    ]

    conv = chak.Conversation(model_uri, api_key=api_key, tools=tools)

    user_request = (
        "Use the dti-calculator skill to compute DTI for monthly_income=8000, "
        "monthly_debts=2400. Then use the income-calc skill to average these "
        "monthly income entries: [7000, 7500, 8200]. Report both results, "
        "including which interpreter executed each script."
    )

    print(f"User: {user_request}\n")
    print("-" * 60)

    async for event in await conv.asend(user_request, event=True, timeout=120):
        if isinstance(event, chak.MessageChunk):
            if event.content:
                print(event.content, end="", flush=True)
            if event.is_final:
                print()
        elif isinstance(event, chak.ToolCallStartEvent):
            print(f"\n[Tool call] {event.tool_name}")
            if event.arguments:
                print(f"  args: {event.arguments}")
        elif isinstance(event, chak.ToolCallSuccessEvent):
            result_preview = event.result[:200] + "..." if len(event.result) > 200 else event.result
            print(f"[Tool result] {event.tool_name} -> {result_preview}\n")
        elif isinstance(event, chak.ToolCallErrorEvent):
            print(f"[Tool error] {event.tool_name}: {event.error}\n")

    print("-" * 60)


def main_entry():
    parser = argparse.ArgumentParser(description="ClaudeSkill multi-venv PyRunner demo")
    parser.add_argument(
        "model_uri",
        nargs="?",
        default="anthropic/claude-sonnet-4-6",
        help="Model URI (e.g., openai/gpt-4o, bailian/qwen-plus)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model_uri))


if __name__ == "__main__":
    main_entry()
