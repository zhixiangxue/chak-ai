"""
Chat With PDF Agent Demo
========================
Give the LLM a large remote PDF and a question. It autonomously:
  1. Reads PDF metadata to understand document size and structure
  2. Uses outline/search to locate relevant sections
  3. Reads only the necessary PDF physical pages
  4. Answers with citations

Tools: Pdf
No human guidance after the initial prompt.
"""

import asyncio
import os

import dotenv
dotenv.load_dotenv()

import chak
from chak.tools.std import Pdf
from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent

_MODEL = "anthropic/claude-sonnet-4-6"
_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# Remote PDF
# ---------------------------------------------------------------------------
_PDF_URL = "https://wcbpub.oss-cn-hangzhou.aliyuncs.com/xue/xxxx/SellingGuide.pdf"


def _tool_hint(args: dict) -> str:
    if "source" in args:
        parts = [args["source"]]
        if "query" in args:
            parts.append(f"query={args['query']!r}")
        if "start_page" in args or "end_page" in args:
            parts.append(f"pages={args.get('start_page')}–{args.get('end_page')}")
        return " | ".join(parts)
    return str(args)[:160]


async def main():
    pdf = Pdf()
    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        system_prompt=(
            "You are a careful document analyst. Use the Pdf tool to get metadata "
            "for large documents before reading specific parts. Cite PDF physical "
            "pages first; printed page labels may be mentioned separately."
        ),
        tools=[pdf],
    )
    conv._tool_manager.max_iterations = 30

    prompt = f"""You are given a large PDF:
{_PDF_URL}

Question:
Where are the HomeReady Mortgage loan eligibility and borrower eligibility
requirements in the Selling Guide, and what are the key requirements?

Instructions:
- Do not assume the whole PDF fits in context.
- Let the Pdf tool guide your workflow: metadata, outline/search, then read_pages.
- Answer in Chinese.
- Cite PDF physical pages first. If printed page labels appear in the document,
  mention them separately and do not mix them into the same page range.
"""

    print("=" * 70)
    print("  Chat With PDF Agent — Selling Guide")
    print("=" * 70)
    print()

    async for event in await conv.asend(prompt, event=True):
        match event:
            case MessageChunk(content=text) if text:
                print(text, end="", flush=True)

            case ToolCallStartEvent(tool_name=name, arguments=args):
                print(f"\n\n>>> [{name}] {_tool_hint(args)}")

            case ToolCallSuccessEvent(tool_name=name, result=res):
                preview = (res or "")[:220].replace("\n", " ")
                print(f"<<< {preview}")

            case ToolCallErrorEvent(tool_name=name, error=err):
                print(f"<<< ERROR: {err}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
