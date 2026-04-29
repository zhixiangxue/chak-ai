"""
BI Monthly Report Agent Demo
=============================
Two-phase autonomous pipeline:

  Phase 1 — ETL
    The LLM reads the Superstore CSV, creates a SQLite database, and loads
    all rows into a normalised `orders` table.

  Phase 2 — BI Analysis & Report
    The LLM queries the database with SQL to produce a December 2017 monthly
    report covering revenue, profit, MoM/YoY comparison, category breakdown,
    regional ranking, and discount analysis.  The final report is written to
    a Markdown file.

Tools: Python (ETL) + SQL (analysis) + FileSystem (report output)
No human guidance after the initial prompt.
"""

import asyncio
import os
import urllib.request
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import chak
from chak.tools.std import Python, SQL, FileSystem
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

CSV_FILE  = str(_DATA_DIR   / "Superstore.csv")
DB_FILE   = str(_OUTPUT_DIR / "superstore.db")
DB_URI    = f"sqlite:///{DB_FILE}"
REPORT    = str(_OUTPUT_DIR / "superstore_monthly_report.md")


def _ensure_data() -> None:
    """Download the sample CSV from OSS if it is not already present."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = Path(CSV_FILE)
    if csv.exists():
        return
    print(f"Downloading sample data \u2192 {csv}")
    urllib.request.urlretrieve(_CSV_URL, csv)
    print(f"Downloaded {csv.stat().st_size:,} bytes")


async def main():
    _ensure_data()
    py  = Python()
    sql = SQL()
    fs  = FileSystem()

    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        tools=[py, sql, fs],
    )
    conv._tool_manager.max_iterations = 100

    prompt = f"""You are a BI engineer and data analyst. Complete the following two-phase task
end-to-end without any human assistance.

---

## Phase 1 — ETL: Load CSV into SQLite

Source file : {CSV_FILE}
Target DB   : {DB_FILE}
Target URI  : {DB_URI}

Use Python + pandas (install with pip if missing) to:
1. Read the CSV (try utf-8 first, fall back to latin-1 if it fails).
2. Create / replace a table called `orders` in the SQLite database with
   these columns (use the exact names as SQL identifiers):
     row_id, order_id, order_date, ship_date, ship_mode,
     customer_id, customer_name, segment, country, city, state,
     postal_code, region, product_id, category, sub_category,
     product_name, sales, quantity, discount, profit
3. Load all rows into the table.
4. Print a confirmation: "ETL complete — N rows loaded".

Important: the Python environment is stateless (each tool call is a fresh
interpreter). Put the entire ETL logic in a SINGLE Python call.
Use sys.stdout.reconfigure(encoding='utf-8', errors='replace') at the top
to avoid Windows GBK encoding errors.

---

## Phase 2 — BI Analysis: December 2017 Monthly Report

Report month : December 2017
Database URI : {DB_URI}
Output file  : {REPORT}

Use ONLY the SQL tool (no Python) for all analysis from this point on.

Run the following SQL queries and collect the results:

### 2-A  Month Overview
Total revenue, total profit, profit margin %, number of orders, and number
of distinct customers for December 2017.

### 2-B  Month-over-Month Comparison
Compare December 2017 vs November 2017:
revenue, profit, profit margin, order count — with absolute delta and % change.

### 2-C  Year-over-Year Comparison
Compare December 2017 vs December 2016:
same four metrics, absolute delta and % change.

### 2-D  Category Performance (Dec 2017)
Revenue, profit, and profit margin % by Category, sorted by revenue desc.

### 2-E  Sub-Category Performance (Dec 2017)
Revenue, profit, and profit margin % by Sub-Category, sorted by profit desc.
Show top 5 and bottom 3.

### 2-F  Regional Performance (Dec 2017)
Revenue, profit, and profit margin % by Region, sorted by revenue desc.

### 2-G  Top 5 States by Revenue (Dec 2017)

### 2-H  Discount Impact (Dec 2017)
Group orders into discount bands:
  0 %  |  1–20 %  |  21–40 %  |  > 40 %
Show order count, total revenue, and avg profit margin % per band.

### 2-I  Top 10 Products by Revenue (Dec 2017)

### 2-J  Full-Year 2017 Monthly Trend
Monthly revenue and profit for all 12 months of 2017, sorted by month.
Format as a table and also render a simple ASCII bar chart of monthly revenue.

---

## Phase 3 — Write Report

Compile ALL the query results from Phase 2 into a single Markdown report
and save it to: {REPORT}

The report must include:
- A title and generation timestamp
- Each section headed with ## and the section name
- All data in Markdown tables (with aligned columns)
- A final **Executive Insights** section with 5 key takeaways from the data
  (focus on month-over-month and year-over-year trends, and actionable findings)

After saving, print the executive summary to stdout.
"""

    print("=" * 70)
    print("  BI Monthly Report Agent — Superstore December 2017")
    print("=" * 70)
    print()

    async for event in await conv.asend(prompt, event=True):
        match event:
            case MessageChunk(content=text) if text:
                print(text, end="", flush=True)

            case ToolCallStartEvent(tool_name=name, arguments=args):
                hint = ""
                if "code" in args:
                    hint = args["code"][:120].replace("\n", " ")
                elif "query" in args:
                    hint = args["query"][:120].replace("\n", " ")
                elif "sql" in args:
                    hint = args["sql"][:120].replace("\n", " ")
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
