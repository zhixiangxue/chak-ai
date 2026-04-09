"""Claude Agent Skill e2e demo: pdf skill

Demonstrates how ClaudeSkill integrates Anthropic Agent Skills into chak's
function calling loop with three-layer progressive disclosure:

  Layer 1 - LLM sees 'pdf' as a callable tool (name + description only)
  Layer 2 - LLM calls 'pdf' -> receives full SKILL.md body + file manifest
  Layer 3 - LLM calls 'pdf__read_file' -> reads specific scripts/reference docs

Usage:
    python examples/tool_calling_claude_skill_pdf.py
    python examples/tool_calling_claude_skill_pdf.py openai/gpt-4o
"""

import argparse
import asyncio
import os
from pathlib import Path

import dotenv

import chak
from chak.tools.exec import Bash, Python
from chak.tools.skills import ClaudeSkill

dotenv.load_dotenv()

# Project root (examples/ -> ..)
_ROOT = Path(__file__).parent.parent

# Skill directory (Anthropic official skill repo, already cloned to tmp/)
SKILL_DIR = _ROOT / "tmp" / "skills" / "skills" / "pdf"


async def main(model_uri: str, pdf_path: str):
    provider = model_uri.split("/")[0].upper()
    api_key = os.getenv(f"{provider}_API_KEY", "")
    if not api_key:
        raise ValueError(f"Please set {provider}_API_KEY in .env")

    # Load the Claude pdf skill
    skill = ClaudeSkill(SKILL_DIR)
    bash = Bash()
    python = Python()

    conv = chak.Conversation(
        model_uri,
        api_key=api_key,
        tools=[skill, bash, python],
    )

    output_dir = str(Path(pdf_path).parent / "images")
    user_request = (
        f"Convert this PDF to images and save to {output_dir}: {pdf_path}. "
        f"Write and execute the Python code directly."
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
    parser = argparse.ArgumentParser(description="ClaudeSkill pdf demo")
    parser.add_argument(
        "model_uri",
        nargs="?",
        default="anthropic/claude-sonnet-4-6",
        help="Model URI (e.g., openai/gpt-4o, bailian/qwen-plus)",
    )
    parser.add_argument(
        "--pdf",
        default=str(_ROOT / "tmp" / "ddd" / "04.pdf"),
        help="Path to the PDF file",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model_uri, args.pdf))


if __name__ == "__main__":
    main_entry()
