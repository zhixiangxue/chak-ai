"""
Per-Model Token Usage & Cost Tracking — the reference example.

This is the canonical pattern for usage accounting in chak. It mirrors the
stats table of the built-in Inspector (chak/inspector/viewer.html): usage is
aggregated **per model_uri** (``provider/model``), NOT per conversation and
NOT per provider, because cost is model-specific — a per-model row is what
actually matches a billing line item on your invoice.

Why per-model instead of ``conv.stats()``?
    ``conv.stats()`` sums the whole conversation into one bucket. That is
    fine for a single-model chat, but the moment fallbacks or routers are
    involved, one conversation can be served by several different models,
    each with its own unit prices. Aggregating per resolved model_uri keeps
    every token attributed to the model that actually processed it.

Canonical Usage semantics (see chak/metadata.py::Usage):
    chak normalizes every provider's usage into four DISJOINT buckets:

        prompt_tokens                : fresh input (billed at Input rate)
        completion_tokens            : output tokens
        cache_creation_input_tokens  : cache WRITE  (Anthropic ~1.25x input)
        cache_read_input_tokens      : cache READ   (~0.1x input)

    Invariant: total_tokens == sum of the four buckets, on every provider.
    This is why the cost formula below is uniform — no per-provider math.

What this demo does:
    1. Runs a tool-calling agent on a resilient (fallback) chain whose
       primary endpoint is deliberately unreachable, so every call
       demonstrably fails over — and the stats attribute tokens to the
       model that RESOLVED the call, not the one you asked for.
    2. Runs a second, direct conversation on a different model so the
       final report contains multiple per-model rows.
    3. Aggregates all messages into a rich table: calls / in / out /
       cache_w / cache_r / total / estimated cost, plus a grand-total row
       and per-currency cost totals (Money refuses to mix currencies).

Prerequisites:
    export DEEPSEEK_API_KEY=...   # resolves the fallback chain
    export BAILIAN_API_KEY=...    # second model in the report

Usage:
    python examples/usage_token_tracking.py
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, List

import dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

dotenv.load_dotenv()

import chak
from chak.metadata import Money, Usage

console = Console()


# ============================================================================
# Price table — per-MTok unit prices, keyed by model_uri
# ============================================================================
# Same convention as OpenAI's / Anthropic's pricing pages: currency units per
# million tokens. The four disjoint buckets each get their own rate; cache
# read is typically ~10x cheaper than fresh input, which is exactly why the
# report keeps cache_r in its own column instead of folding it into "in".
# Numbers below are illustrative — swap in the real ones for your account.
PRICE_TABLE: Dict[str, Dict] = {
    "openai/gpt-4o-mini": {
        "input_price": 0.15, "output_price": 0.60,
        "cache_read_price": 0.075, "currency": "USD",
    },
    "deepseek/deepseek-v4-flash": {
        "input_price": 0.27, "output_price": 1.10,
        "cache_read_price": 0.027, "currency": "USD",
    },
    "bailian/qwen-plus": {
        "input_price": 0.8, "output_price": 2.0,
        "cache_read_price": 0.16, "currency": "CNY",
    },
}


# ============================================================================
# Tools — small, deterministic, offline (the point here is accounting,
# so the tools themselves stay boring on purpose)
# ============================================================================

def get_city_temperature(city: str) -> str:
    """
    Get the current temperature of a city in Celsius.

    Args:
        city: City name in English, e.g. "Tokyo"

    Returns:
        Temperature description in Celsius
    """
    fake_weather = {"tokyo": 18.5, "paris": 12.0, "beijing": 6.5, "sydney": 24.0}
    temp = fake_weather.get(city.strip().lower())
    if temp is None:
        return f"No weather data for {city}"
    return f"The temperature in {city} is {temp} Celsius"


def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert a Celsius temperature to Fahrenheit.

    Args:
        celsius: Temperature in Celsius

    Returns:
        Temperature in Fahrenheit
    """
    return celsius * 9 / 5 + 32


# ============================================================================
# Aggregation — the reusable core of this example
# ============================================================================

@dataclass
class ModelBucket:
    """Accumulated usage for one resolved model_uri."""
    model_uri: str
    calls: int = 0
    usage: Usage = field(default_factory=Usage)


def aggregate_usage(messages: List) -> List[ModelBucket]:
    """Bucket token usage by resolved ``provider/model``.

    This is a 1:1 Python port of ``aggregateStats()`` in the Inspector
    (chak/inspector/viewer.html): walk assistant messages, key each one by
    the model_uri stamped in its metadata, and sum the four disjoint
    buckets. Only assistant messages carry usage — user/tool/system
    messages cost nothing by themselves (their tokens are counted as the
    NEXT assistant call's input).

    Works on any message list: one conversation, several conversations
    concatenated, or a filtered slice (e.g. a single turn_id).
    """
    by_uri: Dict[str, ModelBucket] = {}
    for m in messages:
        if getattr(m, "role", None) != "assistant":
            continue
        usage = m.metadata.usage
        if not usage:
            continue
        # metadata.provider/model reflect the model that RESOLVED the call.
        # Under failover this differs from the primary you configured —
        # which is exactly what billing attribution needs.
        key = f"{m.metadata.provider}/{m.metadata.model}" if m.metadata.model else "unknown"
        bucket = by_uri.setdefault(key, ModelBucket(model_uri=key))
        bucket.calls += 1
        bucket.usage.prompt_tokens += usage.prompt_tokens
        bucket.usage.completion_tokens += usage.completion_tokens
        bucket.usage.cache_creation_input_tokens += usage.cache_creation_input_tokens
        bucket.usage.cache_read_input_tokens += usage.cache_read_input_tokens
        bucket.usage.total_tokens += usage.total_tokens
    return sorted(by_uri.values(), key=lambda b: b.usage.total_tokens, reverse=True)


def bucket_cost(bucket: ModelBucket) -> Money | None:
    """Price one bucket with the uniform disjoint-bucket formula.

    ``Usage.estimate_cost`` multiplies each bucket by its own unit price, so
    the same call works for OpenAI, Anthropic, DeepSeek, Bailian, ... —
    provider differences were already normalized away at ingestion time.
    """
    prices = PRICE_TABLE.get(bucket.model_uri)
    if prices is None:
        return None
    return bucket.usage.estimate_cost(**prices)


# ============================================================================
# Rendering — rich tables mirroring the Inspector's stats panel
# ============================================================================

def render_usage_report(messages: List, title: str) -> None:
    """Print the per-model usage/cost table for a message list."""
    buckets = aggregate_usage(messages)
    if not buckets:
        console.print("[dim]no llm calls yet[/dim]")
        return

    table = Table(title=title, box=box.SIMPLE_HEAVY, title_justify="left")
    # model_uri folds onto a second line on narrow terminals instead of
    # squeezing the numeric columns into unreadable "1,4…" fragments.
    table.add_column("model_uri", style="cyan", overflow="fold")
    table.add_column("calls", justify="right", style="dim")
    table.add_column("in", justify="right")        # prompt_tokens (fresh input)
    table.add_column("out", justify="right")       # completion_tokens
    table.add_column("cache_w", justify="right")   # cache_creation_input_tokens
    table.add_column("cache_r", justify="right")   # cache_read_input_tokens
    table.add_column("total", justify="right", style="bold")
    table.add_column("cost", justify="right", style="green")

    # Zeros render dim so real numbers stand out — same trick as viewer.html.
    def nz(n: int) -> str:
        return f"{n:,}" if n > 0 else "[dim]0[/dim]"

    # Compact money formatting for table cells (str(Money) uses 6 decimals,
    # which is more precision than a glanceable report needs).
    def fmt_money(m: Money) -> str:
        return f"{m.amount:.4f} {m.currency}"

    costs: List[Money] = []
    for b in buckets:
        cost = bucket_cost(b)
        if cost is not None:
            costs.append(cost)
        u = b.usage
        table.add_row(
            b.model_uri, f"{b.calls}",
            nz(u.prompt_tokens), nz(u.completion_tokens),
            nz(u.cache_creation_input_tokens), nz(u.cache_read_input_tokens),
            f"{u.total_tokens:,}",
            fmt_money(cost) if cost is not None else "[dim]n/a[/dim]",
        )

    # Grand-total row — only meaningful when several models are in play.
    # Money enforces currency safety: we may only sum within one currency,
    # so mixed-currency totals are stacked side by side, never added.
    if len(buckets) > 1:
        by_currency: Dict[str, Money] = {}
        for c in costs:
            by_currency[c.currency] = by_currency.get(c.currency, Money(amount=0, currency=c.currency)) + c
        total_cost = "\n".join(fmt_money(c) for c in by_currency.values()) or "[dim]n/a[/dim]"
        table.add_section()
        table.add_row(
            "[bold]total[/bold]",
            f"{sum(b.calls for b in buckets)}",
            f"{sum(b.usage.prompt_tokens for b in buckets):,}",
            f"{sum(b.usage.completion_tokens for b in buckets):,}",
            f"{sum(b.usage.cache_creation_input_tokens for b in buckets):,}",
            f"{sum(b.usage.cache_read_input_tokens for b in buckets):,}",
            f"[bold]{sum(b.usage.total_tokens for b in buckets):,}[/bold]",
            total_cost,
        )

    console.print(table)


def render_failover_traces(conv: chak.Conversation) -> None:
    """Show which model actually resolved each LLM call.

    Every assistant message carries a ProviderTrace — even without
    failover — so this table doubles as an audit log tying the usage
    report back to concrete routing decisions.
    """
    traces = conv.get_provider_traces()
    if not traces:
        return
    table = Table(title="Provider routing trace", box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("call", justify="right", style="dim")
    table.add_column("primary", style="red", overflow="fold")
    table.add_column("resolved", style="green", overflow="fold")
    table.add_column("fallback", justify="center")
    table.add_column("failed attempts", overflow="fold")
    for i, t in enumerate(traces, 1):
        failures = "; ".join(
            f"{f.provider}/{f.model} ({f.error_type})" for f in t.failed_providers
        )
        table.add_row(
            str(i),
            f"{t.primary_provider}/{t.primary_model}",
            f"{t.resolved_provider}/{t.resolved_model}",
            "[yellow]yes[/yellow]" if t.fallback_used else "[dim]no[/dim]",
            failures or "[dim]—[/dim]",
        )
    console.print(table)


# ============================================================================
# Main demo
# ============================================================================

def main() -> None:
    # Tool calling in chak runs on the async path, so the demo is async too.
    asyncio.run(run_demo())


async def run_demo() -> None:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    bailian_key = os.getenv("BAILIAN_API_KEY")
    missing = [k for k, v in {
        "DEEPSEEK_API_KEY": deepseek_key,
        "BAILIAN_API_KEY": bailian_key,
    }.items() if not v]
    if missing:
        console.print(f"[red]Error:[/red] please set {', '.join(missing)}")
        return

    # ------------------------------------------------------------------
    # Part 1 — tool-calling agent on a fallback chain.
    #
    # The primary points at an unreachable local endpoint, so EVERY call in
    # the tool loop fails over to DeepSeek. The usage report will attribute
    # all tokens to deepseek/deepseek-v4-flash — the resolved model — not
    # to the gpt-4o-mini we nominally asked for. The API key of the broken
    # primary is never sent anywhere, so a placeholder is fine.
    # ------------------------------------------------------------------
    console.print(Panel.fit(
        "[bold]Part 1[/bold] — tool-calling agent with failover\n"
        "primary [red]openai@127.0.0.1:9[/red] (unreachable) → "
        "fallback [green]deepseek/deepseek-v4-flash[/green]",
        border_style="blue",
    ))

    agent = chak.Conversation(
        "openai@http://127.0.0.1:9/v1:gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", "sk-unused"),
        timeout=2,
        fallbacks=[
            {"model_uri": "deepseek/deepseek-v4-flash", "api_key": deepseek_key, "timeout": 60},
        ],
        tools=[get_city_temperature, celsius_to_fahrenheit],
    )

    # This question chains two tools (lookup Celsius → convert), so one
    # asend() fans out into several LLM calls — each with its own usage.
    question = "What is the current temperature in Tokyo, in Fahrenheit? Use the tools."
    console.print(f"\n[bold]User:[/bold] {question}")
    response = await agent.asend(question)
    console.print(f"[bold]Agent:[/bold] {response.content}\n")

    # Single-call inspection: the final assistant message carries the usage
    # of the LAST llm call only — a common trap. Per-turn totals require
    # aggregating over all assistant messages, which is what
    # aggregate_usage() does below.
    u = response.metadata.usage
    console.print(
        f"[dim]Last call only ({response.metadata.provider}/{response.metadata.model}): "
        f"in={u.prompt_tokens} out={u.completion_tokens} "
        f"cache_w={u.cache_creation_input_tokens} cache_r={u.cache_read_input_tokens} "
        f"total={u.total_tokens}[/dim]\n"
    )

    render_failover_traces(agent)
    console.print()
    render_usage_report(agent.messages, "Part 1 — usage by resolved model")

    # ------------------------------------------------------------------
    # Part 2 — a second conversation on a different model.
    #
    # Real applications run many conversations across many models. Because
    # aggregate_usage() takes a plain message list, app-wide accounting is
    # just list concatenation — no framework machinery required.
    # ------------------------------------------------------------------
    console.print(Panel.fit(
        "[bold]Part 2[/bold] — same tools, direct [green]bailian/qwen-plus[/green] "
        "(CNY pricing)",
        border_style="blue",
    ))

    chat = chak.Conversation(
        "bailian/qwen-plus",
        api_key=bailian_key,
        tools=[get_city_temperature, celsius_to_fahrenheit],
    )
    question = "What is the current temperature in Paris, in Fahrenheit? Use the tools."
    console.print(f"\n[bold]User:[/bold] {question}")
    response = await chat.asend(question)
    console.print(f"[bold]Agent:[/bold] {response.content}\n")

    # ------------------------------------------------------------------
    # Final report — one table across BOTH conversations, one row per
    # model_uri, one cost per row, currency-safe grand totals.
    # ------------------------------------------------------------------
    console.print(Panel.fit(
        "[bold]Final report[/bold] — app-wide usage across all conversations",
        border_style="magenta",
    ))
    render_usage_report(agent.messages + chat.messages, "All conversations — usage by model")

    console.print(
        "\n[bold]Takeaways[/bold]\n"
        "  • Aggregate per [cyan]model_uri[/cyan], not per conversation — that matches billing.\n"
        "  • Under failover, tokens belong to the [green]resolved[/green] model, not the primary.\n"
        "  • One tool-calling asend() = several LLM calls; sum assistant messages, "
        "not just the final response.\n"
        "  • The four usage buckets are disjoint on every provider, so "
        "[cyan]Usage.estimate_cost()[/cyan] is one uniform formula.\n"
        "  • [cyan]Money[/cyan] keeps USD and CNY from being silently added together."
    )


if __name__ == "__main__":
    main()
