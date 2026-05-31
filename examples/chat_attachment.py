"""
Example: chat with a PDF attachment.

Attaches Anthropic's "Effective Harnesses for Long-Running Agents" PDF and
asks a question about it.  The PDF content is extracted locally via
pymupdf4llm before being sent to the model — no upload required.

Usage:
    python examples/chat_attachment.py
"""

import asyncio
import os

import dotenv

import chak
from chak.attachment import PDF

dotenv.load_dotenv()

PDF_URL = (
    "https://wcbpub.oss-cn-hangzhou.aliyuncs.com/xue/xxxx/"
    "Effective-harnesses-for-long-running-agents.pdf"
)


def _make_conv() -> chak.Conversation:
    if key := os.getenv("ANTHROPIC_API_KEY"):
        return chak.Conversation("anthropic/claude-haiku-4-5", api_key=key)
    if key := os.getenv("OPENAI_API_KEY"):
        return chak.Conversation("openai/gpt-4o", api_key=key)
    raise EnvironmentError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY")


async def main() -> None:
    conv = _make_conv()

    print(f"PDF: {PDF_URL}")
    print("-" * 60)

    response = await conv.asend(
        "Summarize the core ideas of this paper in 3-5 sentences.",
        attachments=[PDF(PDF_URL)],
        timeout=120,
    )
    print(response.content)
    print()
    print(f"tokens: {conv.stats()}")


if __name__ == "__main__":
    asyncio.run(main())
