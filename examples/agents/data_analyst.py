"""
Data Analyst Agent Demo
=======================
Give the LLM a sales CSV and a business question.
It uses Python (pandas) to load, analyse, and summarise the data,
then writes a clean Markdown report.

Tools: Python + FileSystem
No human guidance after the initial prompt.
"""

import asyncio
import os
import urllib.request
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import chak
from chak.tools.std import Python, FileSystem
from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent

_MODEL   = "anthropic/claude-sonnet-4-6"
_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# Paths  (all relative to this file so the demo runs anywhere)
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).parent
_DATA_DIR   = _HERE / "data"
_OUTPUT_DIR = _HERE / "output"
_CSV_URL    = "https://wcbpub.oss-cn-hangzhou.aliyuncs.com/xue/xxxx/Superstore.csv"

DATA_FILE   = str(_DATA_DIR   / "Superstore.csv")
REPORT_FILE = str(_OUTPUT_DIR / "superstore_report.md")


def _ensure_data() -> None:
    """Download the sample CSV from OSS if it is not already present."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = Path(DATA_FILE)
    if csv.exists():
        return
    print(f"Downloading sample data → {csv}")
    urllib.request.urlretrieve(_CSV_URL, csv)
    print(f"Downloaded {csv.stat().st_size:,} bytes")


async def main():
    _ensure_data()
    py = Python()
    fs = FileSystem()

    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        tools=[py, fs],
    )
    conv._tool_manager.max_iterations = 80

    prompt = f"""You are a senior data analyst. You have been given a retail sales dataset.

**Data file**: {DATA_FILE}
**Output report**: {REPORT_FILE}

Your task: produce a comprehensive sales analysis report.
Use Python with pandas to analyse the data.
If pandas is not installed, install it first with pip.

The report must cover the following sections:

1. **Executive Summary**
   Total revenue, total profit, overall profit margin, number of orders,
   number of unique customers, and the date range of the dataset.

2. **Top 10 Products by Revenue**
   Product name, revenue, profit, and profit margin — ranked by revenue.

3. **Category & Sub-Category Performance**
   Revenue and profit for each Category and Sub-Category, ranked by profit.

4. **Regional Analysis**
   Revenue, profit, and profit margin by Region.
   Also show the top 5 states by revenue and the bottom 5 states by profit margin.

5. **Monthly Revenue Trend**
   Total monthly revenue for the most recent full year in the dataset,
   formatted as a simple ASCII bar chart or table.

6. **Discount Impact on Profitability**
   Group orders into discount bands (0 %, 1-20 %, 21-40 %, >40 %) and
   show how average profit margin changes across bands.

7. **Key Findings & Recommendations**
   3-5 bullet points — the most important insights from the data and
   what a business should do about them.

Write the full report in Markdown and save it to {REPORT_FILE}.
Also print a short console summary (executive summary only) to stdout.
"""

    print("=" * 70)
    print("  Data Analyst Agent — Superstore Sales Analysis")
    print("=" * 70)
    print()

    async for event in await conv.asend(prompt, event=True):
        match event:
            case MessageChunk(content=text) if text:
                print(text, end="", flush=True)

            case ToolCallStartEvent(tool_name=name, arguments=args):
                hint = ""
                if "code" in args:
                    hint = args["code"][:150].replace("\n", " ")
                elif "path" in args:
                    hint = args["path"]
                print(f"\n\n>>> [{name}] {hint}")

            case ToolCallSuccessEvent(tool_name=name, result=res):
                preview = (res or "")[:200].replace("\n", " ")
                print(f"<<< {preview}")

            case ToolCallErrorEvent(tool_name=name, error=err):
                print(f"<<< ERROR: {err}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
