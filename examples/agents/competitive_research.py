"""
Competitive Research Agent Demo
=================================
Give the LLM a list of competitor URLs.  It autonomously:
  1. Fetches each competitor's pricing / product page
  2. Falls back to Search when a page is blocked or malformed
  3. Extracts structured data: models, pricing, context windows, key features
  4. Builds a side-by-side comparison table
  5. Writes a competitive intelligence report

Topic: LLM API provider pricing comparison
Competitors: OpenAI, Anthropic, Google Gemini, Groq, Together AI

Tools: Web + Search + FileSystem
No human guidance after the initial prompt.
"""

import asyncio
import os
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import chak
from chak.tools.std import Web, Search, FileSystem
from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent

_MODEL   = "anthropic/claude-sonnet-4-6"
_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# Paths  (all relative to this file so the demo runs anywhere)
# ---------------------------------------------------------------------------
_OUTPUT_DIR = Path(__file__).parent / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE   = str(_OUTPUT_DIR / "llm_pricing_competitive_report.md")
RAW_DATA_FILE = str(_OUTPUT_DIR / "llm_pricing_raw.json")

COMPETITORS = {
    "OpenAI":       "https://openai.com/api/pricing/",
    "Anthropic":    "https://www.anthropic.com/pricing",
    "Google Gemini":"https://ai.google.dev/gemini-api/docs/pricing",
    "Groq":         "https://groq.com/pricing/",
    "Together AI":  "https://www.together.ai/pricing",
}


async def main():
    # Jina Reader improves fetch quality; Search falls back to DuckDuckGo if no key
    web    = Web(jina_key=os.getenv("JINA_READER_API_KEY"))
    search = Search(
        tavily_key=os.getenv("TAVILY_API_KEY"),
        brave_key=os.getenv("BRAVE_API_KEY"),
    )
    fs     = FileSystem()

    def make_conv(tools):
        conv = chak.Conversation(_MODEL, api_key=_API_KEY, tools=tools)
        conv._tool_manager.max_iterations = 40
        return conv

    async def run(conv, prompt, label):
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        async for event in await conv.asend(prompt, event=True):
            match event:
                case MessageChunk(content=text) if text:
                    print(text, end="", flush=True)
                case ToolCallStartEvent(tool_name=name, arguments=args):
                    hint = args.get("url") or args.get("query", "")[:80] or args.get("path", "")
                    print(f"\n\n>>> [{name}] {hint}")
                case ToolCallSuccessEvent(tool_name=name, result=res):
                    print(f"<<< {(res or '')[:200].replace(chr(10),' ')}")
                case ToolCallErrorEvent(tool_name=name, error=err):
                    print(f"<<< ERROR: {err}")
        print()

    competitors_block = "\n".join(
        f"- **{name}**: {url}" for name, url in COMPETITORS.items()
    )

    print("=" * 70)
    print("  Competitive Research Agent — LLM API Pricing")
    print("=" * 70)
    print()

    # ── Phase 1: Research ────────────────────────────────────────────────
    # Small-context conv: fetch + save structured data per provider
    research_prompt = f"""You are a market intelligence researcher.

For each of the following LLM API providers, fetch its pricing page and
extract structured data.  Save ALL findings to a single JSON file.

## Providers
{competitors_block}

## For each provider extract
- provider name
- models (list of: name, input_price_per_1m, output_price_per_1m, context_window, highlights)
- free_tier (description or null)
- notable_features (list of strings)
- data_source ("official_page" or "web_search")

## Instructions
1. Fetch each pricing page. If a page fails or has no pricing data, use
   search to find recent pricing info instead.
2. After collecting data for ALL 5 providers, save the result as a
   JSON array to: {RAW_DATA_FILE}
3. Print a one-line confirmation: "Data saved to {RAW_DATA_FILE}"

Keep your written response minimal.
"""

    await run(make_conv([web, search, fs]), research_prompt, "Phase 1 — Research")

    # ── Phase 2: Report ──────────────────────────────────────────────────
    # Fresh conv: read JSON, write full Markdown report
    report_prompt = f"""You are a competitive intelligence analyst.

You have raw pricing data in: {RAW_DATA_FILE}

Read that file, then write a comprehensive Markdown competitive intelligence
report to: {REPORT_FILE}

The report must contain:
1. **Executive Summary** (3-5 sentences)
2. **Flagship Model Comparison Table** — one row per provider
   Columns: Provider | Model | Input $/1M | Output $/1M | Context Window | Highlights
3. **Full Model Catalogue** — one table per provider with all models
4. **Feature Matrix** — rows=providers, cols: Vision | Tool-calling |
   JSON mode | Batch API | Streaming | Fine-tuning | Free tier
5. **Pricing Analysis** — cheapest for: high-volume summarisation,
   conversational assistant, long-document analysis
6. **Strategic Recommendations** — 3-5 bullet points

Save the report to {REPORT_FILE}. After saving, print a brief executive summary to stdout.
"""

    await run(make_conv([fs]), report_prompt, "Phase 2 — Report")


if __name__ == "__main__":
    asyncio.run(main())
