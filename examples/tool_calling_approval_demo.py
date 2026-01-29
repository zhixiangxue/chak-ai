import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

from chak.conversation import Conversation
from chak.tools.manager import ToolCallApproval


async def approval_handler(approval: ToolCallApproval) -> bool:
    """Simple approval handler that always allows tool calls.

    This demo prints tool name and arguments before the call.
    In real applications, this function can send the approval
    request to a frontend and wait for user decision.
    """
    print("[approval] tool =", approval.tool_name)
    print("[approval] args  =", approval.arguments)

    answer = input("Allow this tool call? (y/n): ").strip().lower()
    return answer == "y"


def load_openai_api_key() -> str:
    """Load OPENAI_API_KEY from .env using python-dotenv."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found after load_dotenv")
    return api_key


def get_current_time() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main() -> None:
    conv = Conversation(
        model_uri="openai/gpt-4o-mini",
        api_key=load_openai_api_key(),
        tools=[get_current_time],  # Simple demo tool
        tool_approval_handler=approval_handler,
    )

    try:
        response = await conv.asend("请用 get_current_time 工具告诉我当前时间")
        print("Assistant:", getattr(response, "content", response))
    finally:
        conv.close()


if __name__ == "__main__":
    asyncio.run(main())
