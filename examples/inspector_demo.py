"""
Inspector Demo — 多 turn 的 agent 任务实时观察

演示 chak.inspector.watch(): 打开浏览器 http://127.0.0.1:7878 就能实时看到
messages 一条条进来（HumanMessage / AIMessage / tool_calls / ToolMessage），
不用等 agent 跑完。

任务设计:
  用 3 个连贯的对话轮次，逐步深入 Python asyncio 主题。每一轮都会触发
  工具调用，让浏览器里能明显看到 Turn 1 / Turn 2 / Turn 3 的分组结构。

Prerequisites:
    export DASHSCOPE_API_KEY=sk-xxx      # 通义千问（bailian）
    pip install 'chakpy[server]'         # inspector 需要 fastapi + uvicorn

Usage:
    python examples/inspector_demo.py
"""

import asyncio
import os
import re
from datetime import datetime

import dotenv
import httpx

dotenv.load_dotenv()

import chak
from chak.inspector import watch


# ---------------------------------------------------------------------------
# 用作 LLM 工具的原生 Python 函数
# ---------------------------------------------------------------------------
# 选用 docs.python.org 中文版是因为它:
#   - 国内可稳定直连
#   - 静态 HTML，无反爬
#   - 页面主题相关，方便串成一个"调研 asyncio"的连贯任务

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


async def fetch_url(url: str, max_chars: int = 2500) -> str:
    """
    抓取一个网页并返回其正文（去 HTML 标签、压缩空白）。

    Args:
        url: 要抓取的 URL（http/https）
        max_chars: 最多返回多少字符（避免撑爆 context，默认 2500）

    Returns:
        提取出的纯文本。超过 max_chars 会自动截断。
    """
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    # Strip HTML tags + collapse whitespace. Good enough for docs pages.
    text = _HTML_TAG_RE.sub(" ", r.text)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " …[truncated]"
    return text


def count_chars(text: str) -> int:
    """
    统计一段文本的字符数（用于给 LLM 做量化对比）。

    Args:
        text: 任意文本

    Returns:
        字符总数
    """
    return len(text)


def get_current_time() -> str:
    """
    获取当前日期时间，用于报告落款。

    Returns:
        形如 '2026-07-14 16:30:00' 的字符串
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Every turn drills into one facet, so tool calls line up cleanly with turns.
TURNS = [
    {
        "label": "Turn 1 · asyncio 模块概览",
        "prompt": (
            "第一轮调研：请先调用 `get_current_time` 记录开始时间；"
            "然后调用 `fetch_url` 抓取 "
            "https://docs.python.org/zh-cn/3/library/asyncio.html ，"
            "用 `count_chars` 统计正文字符数；"
            "最后用一段中文向我概括 **asyncio 模块本身是做什么的**（3-5 句）。"
        ),
    },
    {
        "label": "Turn 2 · 深入 Task / 协程",
        "prompt": (
            "第二轮：基于上一轮，我想深入了解 Task 与协程。"
            "请调用 `fetch_url` 抓取 "
            "https://docs.python.org/zh-cn/3/library/asyncio-task.html ，"
            "并用 `count_chars` 统计字符数。"
            "然后用 markdown 输出：\n"
            "- **协程 vs Task** 的核心区别（用表格）\n"
            "- 一段最小可运行代码（含 async/await）\n"
            "- 3 个初学者常见坑"
        ),
    },
    {
        "label": "Turn 3 · 事件循环与总结",
        "prompt": (
            "最后一轮：请调用 `fetch_url` 抓取 "
            "https://docs.python.org/zh-cn/3/library/asyncio-eventloop.html ，"
            "用 `count_chars` 统计字符数，再调用 `get_current_time` 记录结束时间。"
            "然后综合前 3 轮，用 markdown 输出一份**完整调研总结**，包含：\n"
            "- 三个页面的字符数对比（表格）\n"
            "- **asyncio / Task / 事件循环** 三者关系（一段话讲透）\n"
            "- 给初学者的**推荐阅读顺序**及理由\n"
            "- 报告落款（开始时间 → 结束时间）"
        ),
    },
]


async def main():
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("❌ 未检测到 DASHSCOPE_API_KEY / BAILIAN_API_KEY")
        print("   在 https://dashscope.aliyuncs.com/ 申请后设置到 .env")
        return

    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_prompt=(
            "你是一名严谨的技术调研助手，负责多轮深入调研 Python asyncio。"
            "每一轮我都会给出明确步骤 —— 遇到需要访问外部资源时主动调用工具。"
            "调用完所有必要工具后再输出中文回答。"
        ),
        tools=[fetch_url, count_chars, get_current_time],
    )

    # 副 conv：同一进程里另一个会话，用不同模型验证 multi-conv。
    # 顶部 tab bar 会当作另一个页签出现，stats 表格会看到两行
    # 不同的 model_uri（qwen-plus vs qwen-turbo）。
    conv2 = chak.Conversation(
        "bailian/qwen-turbo",
        api_key=api_key,
        system_prompt="你是一个轻量级 QA 助手，简短回答。",
        tools=[get_current_time],
    )

    # 两个 conv 都挂到同一个 inspector（同一个端口）。第一次 watch()
    # 启服务器 + 开浏览器；后续 watch() 只把新 conv 注册到运行中的
    # 服务器，页面顶部 tab bar 会自动长出新 tab。
    watch(conv)
    watch(conv2)

    print()
    print("=" * 70)
    print("  Inspector Demo — 多轮 agent 任务实时观察")
    print("=" * 70)
    print()
    print("💡 浏览器已打开 http://127.0.0.1:7878")
    print("   agent 每追加一条 message，页面就会自动刷新。")
    print(f"   主 conv：{len(TURNS)} 轮 asyncio 调研（qwen-plus）")
    print("   副 conv：小问题（qwen-turbo）—— 顶部 tab bar 可切换")
    print("   3 秒后开始，你可以先切到浏览器盯着看 👀")
    print()
    await asyncio.sleep(3)

    # 先给副 conv 发一条，让 tab bar 里两个 conv 都有内容可看
    print("👥 [副 conv] 发一条预热消息…")
    await conv2.asend("用一句话告诉我现在几点了（调用工具）。")
    print("   → 副 conv 已有 1 轮。现在开始主 conv。")
    await asyncio.sleep(1)

    for idx, turn in enumerate(TURNS, 1):
        print()
        print("─" * 70)
        print(f"👤 [{turn['label']}] User prompt 已发出…")
        print("─" * 70)
        response = await conv.asend(turn["prompt"])
        print(f"🤖 Turn {idx} 完成 —— assistant 输出 {len(response.content)} 字")
        # Small pause between turns so the browser can visibly separate them.
        await asyncio.sleep(1)

    print()
    print("=" * 70)
    stats = conv.stats()
    print(f"📊 总消息数: {stats['total_messages']}   "
          f"总 tokens: {stats['total_tokens']}   "
          f"轮次: {len(TURNS)}")
    print("=" * 70)
    print()
    print("💡 浏览器页面保持打开中，可以回看整个消息流。")
    print("   按 Ctrl+C 结束进程后 inspector 也会一起退出。")
    print()

    # Hold the process open so the user can inspect the final state in browser.
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("\n👋 Bye")


if __name__ == "__main__":
    asyncio.run(main())
