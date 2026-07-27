"""
Form Filler Agent — Real-world Government & Financial Forms
===========================================================
Give the LLM a person's profile in free prose and a blank fillable PDF.
It autonomously walks the Pdf tool's three-step form workflow:

1. ``metadata`` — discover that the PDF is a fillable AcroForm and its size
2. ``schema``   — fetch the field dictionary (name → type/label/options)
3. ``fill``     — submit {field_name: value} mappings, incrementally

Four built-in presets, each a real mainstream form fetched by URL:

=============  ==================================================  ========
preset         form                                                fields
=============  ==================================================  ========
1003           Fannie Mae URLA 1003 (mortgage application)         423
w9             IRS W-9 (taxpayer identification)                   23
1040           IRS 1040 (individual income tax return)             199
china-visa     China Visa Application Form V.2013 (CJK filling)    157
=============  ==================================================  ========

The presets stress different tool capabilities: the 1003 has broken
tooltips (nearby_text rescue), IRS forms are AcroForm+XFA hybrids, and the
China visa form has zero labels plus Chinese text filling.

Tools: Pdf
No human guidance after the initial prompt.

Usage:
    python examples/agents/form_filler.py                # default: 1003
    python examples/agents/form_filler.py --form w9
    python examples/agents/form_filler.py --form china-visa
    python examples/agents/form_filler.py --form 1040 --source /path/or/url.pdf
"""

import argparse
import asyncio
import os
from pathlib import Path

import dotenv
from rich import box
from rich.console import Console
from rich.table import Table

dotenv.load_dotenv()

import chak
from chak.metadata import Usage
from chak.tools.std import Pdf
from chak.message import (
    MessageChunk,
    ToolCallStartEvent,
    ToolCallSuccessEvent,
    ToolCallErrorEvent,
)

# Silence chak's INFO-level tool-call logs; the >>>/<<< lines below already
# visualize every tool call.
chak.set_log_level("WARNING")

_MODEL = "deepseek/deepseek-v4-pro"
_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Per-MTok unit prices for the cost column, keyed by resolved model_uri —
# same convention as examples/usage_token_tracking.py. Numbers are
# illustrative; swap in the real ones for your account.
_PRICE_TABLE: dict[str, dict] = {
    "deepseek/deepseek-v4-pro": {
        "input_price": 0.56, "output_price": 1.68,
        "cache_read_price": 0.056, "currency": "USD",
    },
}

_console = Console()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _REPO_ROOT / "tmp" / "filled"

# A fictional profile shared by the US forms — deliberately in free prose,
# not field names.  Mapping the narrative onto each form is the agent's job.
_JANE_PROFILE = """\
Borrower: Jane Quinn Doe
- SSN: 500-22-6789, born 03/14/1985, U.S. citizen
- Unmarried, no dependents
- Cell phone: (415) 555-0134, email: jane.doe@example.com
- Current address: 2847 Fillmore Street, Unit 3B, San Francisco, CA 94123, USA
  (renting for 3 years and 2 months, $3,400/month)
- Employer: Acme Analytics Inc., (415) 555-0100,
  500 Howard Street, San Francisco, CA 94105, USA
- Position: Senior Data Scientist, started 06/01/2018, 9 years in this line of work
- Not self-employed, no ownership share, not employed by a party to the transaction
- Gross monthly income: base $12,500, bonus $1,000
"""

_FORMS: dict[str, dict[str, str]] = {
    "1003": {
        "url": "https://singlefamily.fanniemae.com/media/7896/display",
        "output": "urla-1003.filled.pdf",
        "profile": _JANE_PROFILE + """\
- Applying for INDIVIDUAL credit
- Loan purpose: purchase a primary residence; she will occupy it herself
- Loan amount: $850,000; property value: $1,100,000
- Property: 129 Chestnut Avenue, San Jose, CA 95110, Santa Clara county, 1 unit
- First-time homebuyer: has never owned any real estate
- No family or business relationship with the property seller
- Not borrowing any undisclosed money for this transaction; no other new
  mortgage or credit applications before closing; no lien could take
  priority over the first mortgage
- Clean financial history: not a co-signer on any loan, no outstanding
  judgments, no delinquent federal debt, no lawsuits, and no deed-in-lieu,
  short sale, foreclosure, or bankruptcy — ever
- Never served in the military
- Declines to provide demographic information (ethnicity / sex / race)
""",
    },
    "w9": {
        "url": "https://www.irs.gov/pub/irs-pdf/fw9.pdf",
        "output": "fw9.filled.pdf",
        "profile": _JANE_PROFILE + """\
- Filing the W-9 as an individual / sole proprietor (side consulting work)
- No business name separate from her legal name
- Uses her SSN as the taxpayer identification number
- Not subject to backup withholding; no FATCA exemption codes apply
""",
    },
    "1040": {
        "url": "https://www.irs.gov/pub/irs-pdf/f1040.pdf",
        "output": "f1040.filled.pdf",
        "profile": _JANE_PROFILE + """\
- Filing status: Single, calendar tax year
- Did NOT receive, sell, or dispose of any digital assets this year
- Nobody can claim her as a dependent
- W-2 wages for the year: $162,000; federal income tax withheld: $31,500
- Interest income from savings: $850 (no tax-exempt interest)
- No other income; takes the STANDARD deduction ($14,600 for single)
- No credits, no estimated payments; wants any refund by check
- Occupation: Data Scientist; no third-party designee; no presidential
  election campaign contribution
""",
    },
    "china-visa": {
        "url": "https://www.nyu.edu/content/dam/nyu/globalServices/documents/forms/facultyandscholars/faculty-visa/Shanghai-Visa-Application-Form.pdf",
        "output": "china-visa.filled.pdf",
        # CJK stress test: the applicant profile mixes English and Chinese,
        # and the form itself is Chinese/Spanish with unnamed fields.
        "profile": """\
Visa applicant: Dali Wang (Chinese name: 王大力)
- U.S. citizen, male, born 07/22/1988 in San Francisco, USA
- No former nationality, no other names ever used
- National ID: none (write 无 where a Chinese form expects "none")
- Ordinary passport No. 588012345, issued 05/10/2019 in San Francisco,
  expires 05/09/2029
- Occupation: company employee; education: university graduate
- Employer: Acme Analytics Inc., +1 (415) 555-0100,
  500 Howard Street, San Francisco, CA 94105, USA, postal code 94105
- Home address: 2847 Fillmore Street, Unit 3B, San Francisco, CA 94123, USA
- Phone: (415) 555-0134, email: dali.wang@example.com
- Purpose of visit: tourism (L visa), single entry, 30 days, one trip
- Planned arrival 10/01/2026, departure 10/15/2026
- Itinerary: Beijing and Shanghai, staying at Beijing Grand Hotel
  (28 Chang'an Avenue, Beijing 北京长安街28号)
- Pays for the trip himself; never visited China; no criminal record;
  no serious diseases; never overstayed or been refused a visa anywhere
""",
    },
}

_SYSTEM_PROMPT = """\
You are a form-filling assistant.  You operate a Pdf tool with a
three-step form workflow, and you MUST follow it in order:

1. Call `pdf-metadata` on the source PDF to confirm it is a fillable form
   and see how many fields there are and on which pages.
2. Call `pdf-schema` ONCE to get the full field dictionary.  Each entry has
   `name` (the exact key to use when filling), `type`, `page`, `label`
   (the field's meaning), and for radio/dropdown an `options` index map.
   Real-world forms ship broken or missing tooltips: when many fields share
   one label, the label is a copy-paste error — trust the entry's
   `nearby_text` (the printed text next to the field) over the label in
   that case.  If field meanings are still unclear, call `pdf-read_pages`
   on the relevant pages to read the printed form text before filling.
3. Call `pdf-fill` with `{field_name: value}` mappings.  Fill INCREMENTALLY,
   in multiple rounds grouped by form section — do not attempt everything in
   one giant call.  For the FIRST round pass the blank form as `source` and
   the designated output path as `output_path`; for every LATER round pass
   the output path as BOTH `source` and `output_path` so values accumulate.

Filling rules:
- Use field names EXACTLY as returned by `schema`.
- Radio/dropdown: pass the zero-based option index (or the exact option
  label).  Checkboxes: pass true/false.
- Skip fields marked `"fillable": false` — the filler cannot address them;
  list them in your final report instead.
- Only fill what the profile actually answers.  NEVER invent data:
  leave unknown fields blank and report them.
- Total/sum fields do NOT auto-calculate (that requires Acrobat JavaScript,
  which the filler does not run).  Compute totals yourself from the values
  you filled.
- If a `fill` response contains `errors`, read the reasons, correct the
  values if possible, and retry only those fields.

When done, produce a final report with: (a) sections/fields you filled,
(b) fields the profile could not answer, grouped by form section, and
(c) any fields rejected by the tool and why.
"""


def _tool_hint(args: dict) -> str:
    """One-line hint for tool-call logging."""
    parts = []
    if "source" in args:
        parts.append(Path(str(args["source"])).name)
    if "data" in args:
        data = args["data"]
        n = len(data) if isinstance(data, dict) else "?"
        parts.append(f"{n} fields")
    if "output_path" in args:
        parts.append(f"→ {Path(str(args['output_path'])).name}")
    return " | ".join(parts) if parts else str(args)[:120]


def _render_usage_report(messages: list, title: str) -> None:
    """Per-model token/cost table — the pattern from usage_token_tracking.py.

    One asend() with a tool loop fans out into many LLM calls; only summing
    every assistant message shows what filling the whole form really cost.
    """
    # Aggregate the four disjoint usage buckets per resolved model_uri.
    buckets: dict[str, dict] = {}
    for m in messages:
        if getattr(m, "role", None) != "assistant":
            continue
        usage = m.metadata.usage
        if not usage:
            continue
        key = f"{m.metadata.provider}/{m.metadata.model}" if m.metadata.model else "unknown"
        b = buckets.setdefault(key, {"calls": 0, "usage": Usage()})
        b["calls"] += 1
        u = b["usage"]
        u.prompt_tokens += usage.prompt_tokens
        u.completion_tokens += usage.completion_tokens
        u.cache_creation_input_tokens += usage.cache_creation_input_tokens
        u.cache_read_input_tokens += usage.cache_read_input_tokens
        u.total_tokens += usage.total_tokens

    if not buckets:
        _console.print("[dim]no llm calls recorded[/dim]")
        return

    table = Table(title=title, box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("model_uri", style="cyan", overflow="fold")
    table.add_column("calls", justify="right", style="dim")
    table.add_column("in", justify="right")        # prompt_tokens (fresh input)
    table.add_column("out", justify="right")       # completion_tokens
    table.add_column("cache_w", justify="right")   # cache_creation_input_tokens
    table.add_column("cache_r", justify="right")   # cache_read_input_tokens
    table.add_column("total", justify="right", style="bold")
    table.add_column("cost", justify="right", style="green")

    def nz(n: int) -> str:
        return f"{n:,}" if n > 0 else "[dim]0[/dim]"

    for uri, b in sorted(buckets.items(), key=lambda kv: kv[1]["usage"].total_tokens, reverse=True):
        u = b["usage"]
        prices = _PRICE_TABLE.get(uri)
        cost = u.estimate_cost(**prices) if prices else None
        table.add_row(
            uri, f"{b['calls']}",
            nz(u.prompt_tokens), nz(u.completion_tokens),
            nz(u.cache_creation_input_tokens), nz(u.cache_read_input_tokens),
            f"{u.total_tokens:,}",
            f"{cost.amount:.4f} {cost.currency}" if cost else "[dim]n/a[/dim]",
        )

    _console.print(table)


async def main(form: str, source_override: str | None):
    if not _API_KEY:
        print("DEEPSEEK_API_KEY not set.  Check .env")
        return

    preset = _FORMS[form]
    source = source_override or preset["url"]
    # Local paths are validated up front; URLs are downloaded by the Pdf
    # tool itself on first use.
    if not source.startswith(("http://", "https://")):
        source = str(Path(source).expanduser().resolve())
        if not Path(source).exists():
            print(f"Blank form not found: {source}")
            return
    output_pdf = _OUTPUT_DIR / preset["output"]
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        system_prompt=_SYSTEM_PROMPT,
        tools=[Pdf()],
    )
    # The largest preset has 400+ fields; leave room for several fill rounds.
    conv.tool.loop.max(30)

    prompt = (
        f"Blank form to fill: {source}\n"
        f"Write the filled form to: {output_pdf}\n\n"
        f"Profile:\n{preset['profile']}\n"
        "Fill in everything this profile answers, then give me your report."
    )

    print("=" * 70)
    print(f"  Form Filler Agent — {form}")
    print(f"  form:   {source}")
    print(f"  output: {output_pdf}")
    print("=" * 70)
    print()

    async for event in await conv.asend(prompt, event=True):
        match event:
            case MessageChunk(content=text) if text:
                print(text, end="", flush=True)

            case ToolCallStartEvent(tool_name=name, arguments=args):
                print(f"\n\n>>> [{name}] {_tool_hint(args)}")

            case ToolCallSuccessEvent(tool_name=name, result=res):
                preview = (res or "")[:200].replace("\n", " ")
                print(f"<<< {preview}")

            case ToolCallErrorEvent(tool_name=name, error=err):
                print(f"<<< ERROR: {err}")

    print()
    if output_pdf.exists():
        print(f"Filled form saved: {output_pdf}")

    # What did filling this form cost? Sum ALL assistant messages — the tool
    # loop makes many LLM calls behind the single asend() above.
    print()
    _render_usage_report(conv.messages, f"Token usage — {form}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Let an agent fill a real-world form from a person's profile."
    )
    parser.add_argument(
        "--form",
        choices=sorted(_FORMS),
        default="1003",
        help="Which built-in form preset to fill (default: 1003).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Override the form source with a local path or URL "
        "(the preset's profile is still used).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.form, args.source))
