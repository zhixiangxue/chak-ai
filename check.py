"""
Chak provider capability check — one-shot health check for any ``model_uri``.

Given a chak model URI (e.g. ``moonshot/kimi-k3``), run every core capability
chak exposes and print a pass/fail/skip report. Use this whenever you plug in
a new provider or a new flagship model and want to know — in one command —
whether it can carry real production traffic.

Capabilities exercised (in order):

    1.  sync-nonstream           conv.send(...)
    2.  async-nonstream          await conv.asend(...)
    3.  sync-stream              conv.send(..., stream=True)
    4.  async-stream             async for c in await conv.asend(..., stream=True)
    5.  event-stream             async for e in await conv.asend(..., event=True)
    6.  multi-turn               two turns, context retained
    7.  usage-tokens             metadata.usage.total_tokens > 0
    8.  tool-call (function)     wrap_tools([function]) + async event stream
    9.  tool-call (object)       wrap_tools([instance]) + async event stream
    10. structured-single        returns=CityInfo
    11. structured-list          returns=list[CityInfo]
    12. structured-with-tools    returns=CityInfo + wrap_tools([...])
    13. reasoning (optional)     reasoning=chak.Reasoning(effort="high")
    14. cache (optional)         Cache(system_prompt=True) with long prefix,
                                 measured across two turns

Cache / reasoning are optional because they are provider-specific: cache
today is Anthropic + OpenAI, reasoning is Bailian / OpenAI o-series /
DeepSeek-R1 / etc. They are skipped unless ``--with-cache`` or
``--with-reasoning`` is passed.

Live observation
----------------
By default this script also starts the chak inspector and auto-attaches
every Conversation it creates. Open the printed URL in a browser to watch
messages / tool calls / streaming chunks arrive in real time — each check
appears as its own tab labeled with the check name. Disable with
``--no-inspect`` for CI use.

Usage:

    # simplest — auto-detects API key env from provider name
    python check.py moonshot/kimi-k3

    # explicit env var (e.g. Kimi key stored under KIMI_API_KEY)
    python check.py moonshot/kimi-k3 --api-key-env KIMI_API_KEY

    # pass the key inline
    python check.py moonshot/kimi-k3 --api-key sk-...

    # opt-in extras
    python check.py anthropic/claude-sonnet-4-6 --with-cache
    python check.py openai/o4-mini --with-reasoning

    # run only some checks
    python check.py deepseek/deepseek-chat --only tool-call,structured

    # skip broken/expensive ones
    python check.py bailian/qwen-plus --skip structured-with-tools

    # CI-friendly: no inspector, no browser
    python check.py openai/gpt-4o --no-inspect

Exit code is 0 iff every non-skipped check passed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

import dotenv
from pydantic import BaseModel

import chak
from chak.tools import wrap_tools
from chak.utils.uri import parse as parse_uri

# Load .env once at import so ``os.getenv`` sees developer-local keys.
dotenv.load_dotenv()

# ── constants ──────────────────────────────────────────────────────────────

# Padded system prompt (~1300 tokens) used ONLY by the cache check. Both
# Anthropic and OpenAI require >= 1024 tokens in the cached prefix, so a short
# system prompt would silently produce zero cache activity and mask real bugs.
_CACHE_SYSTEM_PROMPT = (
    "You are ChakBot, an expert Python engineering assistant. Follow these "
    "coding standards strictly on every response.\n\n"
) + ("\n".join(
    f"Rule {i}: Prefer explicit type hints, Google-style docstrings, specific "
    f"exception handling, snake_case names, pytest with parametrize, asyncio "
    f"for IO-bound work, pydantic BaseModel for validation, cProfile before "
    f"optimizing, secrets over random, and never eval() untrusted input."
    for i in range(1, 30)
))


class CityInfo(BaseModel):
    """Trivial structured-output target used by the structured checks."""

    city: str
    country: str


# ── result plumbing ────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Outcome of a single capability probe."""

    name: str
    status: str  # PASS / FAIL / SKIP
    elapsed: float = 0.0
    detail: str = ""
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


@dataclass
class Registry:
    """Collects the ordered list of checks and their filters."""

    order: List[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)  # name -> async callable

    def add(self, name: str, fn: Callable[..., Awaitable["CheckResult"]]) -> None:
        self.order.append(name)
        self.checks[name] = fn


# ── shared conversation factory ────────────────────────────────────────────


def _make_conv(
    model_uri: str,
    api_key: str,
    *,
    title: Optional[str] = None,
    tools: Optional[list] = None,
    system_prompt: Optional[str] = None,
    cache: Optional[chak.Cache] = None,
    timeout: int = 60,
) -> chak.Conversation:
    """Build a fresh Conversation. Each check gets its own to avoid state bleed.

    ``title`` is applied post-construction so it shows up in the inspector
    tab bar. We set it via ``conv.title = ...`` (chak's public API) rather
    than a constructor kwarg to stay compatible with older chak versions
    where the argument didn't exist.
    """
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if tools is not None:
        kwargs["tools"] = tools
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    if cache is not None:
        kwargs["cache"] = cache
    conv = chak.Conversation(model_uri, **kwargs)
    if title:
        conv.title = title
    return conv


# ── individual checks ──────────────────────────────────────────────────────
# Each check returns a CheckResult. They ALL raise-safe: the driver wraps
# every call in try/except so one broken provider quirk cannot short-circuit
# the whole report.


async def check_sync_nonstream(uri: str, key: str) -> CheckResult:
    """Baseline: blocking send returns a non-empty AIMessage."""
    conv = _make_conv(uri, key, title="sync-nonstream")
    # Run the sync path off the event loop so we don't block async siblings.
    # send() positional args are (message, attachments) — always pass timeout by name.
    response = await asyncio.to_thread(
        lambda: conv.send(
            "Reply with one short sentence containing: chak sync ok.",
            timeout=60,
        )
    )
    assert isinstance(response, chak.AIMessage), f"unexpected type: {type(response)}"
    assert response.content, "empty content"
    return CheckResult(
        "sync-nonstream", "PASS",
        detail=f"{len(response.content)} chars",
    )


async def check_async_nonstream(uri: str, key: str) -> CheckResult:
    """Async blocking send — main API surface for backend workloads."""
    conv = _make_conv(uri, key, title="async-nonstream")
    response = await conv.asend(
        "Reply with one short sentence containing: chak async ok.", timeout=60
    )
    assert isinstance(response, chak.AIMessage), f"unexpected type: {type(response)}"
    assert response.content, "empty content"
    return CheckResult(
        "async-nonstream", "PASS",
        detail=f"{len(response.content)} chars",
    )


async def check_sync_stream(uri: str, key: str) -> CheckResult:
    """Sync streaming: iter yields at least one MessageChunk with content."""
    conv = _make_conv(uri, key, title="sync-stream")

    def _run() -> tuple[int, str]:
        chunks = list(conv.send(
            "Reply with one short sentence containing: chak stream ok.",
            stream=True, timeout=60,
        ))
        text = "".join(c.content for c in chunks if isinstance(c, chak.MessageChunk))
        return len(chunks), text

    n_chunks, text = await asyncio.to_thread(_run)
    assert text.strip(), "empty stream content"
    assert n_chunks >= 1, "no chunks emitted"
    return CheckResult(
        "sync-stream", "PASS",
        detail=f"{n_chunks} chunks, {len(text)} chars",
    )


async def check_async_stream(uri: str, key: str) -> CheckResult:
    """Async streaming: matches sync-stream semantics via asend(stream=True)."""
    conv = _make_conv(uri, key, title="async-stream")
    chunks: list[Any] = []
    text_parts: list[str] = []
    async for chunk in await conv.asend(
        "Reply with one short sentence containing: chak astream ok.",
        stream=True, timeout=60,
    ):
        chunks.append(chunk)
        if isinstance(chunk, chak.MessageChunk) and chunk.content:
            text_parts.append(chunk.content)
    text = "".join(text_parts)
    assert text.strip(), "empty stream content"
    assert chunks, "no chunks emitted"
    return CheckResult(
        "async-stream", "PASS",
        detail=f"{len(chunks)} chunks, {len(text)} chars",
    )


async def check_event_stream(uri: str, key: str) -> CheckResult:
    """Event mode: same wire flow as tool calling, but without tools bound.

    A well-behaved provider still emits terminal message events even when no
    tools are configured; this pins that contract.
    """
    conv = _make_conv(uri, key, title="event-stream")
    events: list[Any] = []
    async for event in await conv.asend(
        "Reply with one short sentence containing: chak event ok.",
        event=True, timeout=60,
    ):
        events.append(event)
    assert events, "no events emitted"
    assert conv.messages[-1].role == "assistant"
    assert conv.messages[-1].content, "empty final assistant message"
    return CheckResult(
        "event-stream", "PASS",
        detail=f"{len(events)} events",
    )


async def check_multi_turn(uri: str, key: str) -> CheckResult:
    """Two turns on the same Conversation — context must be retained.

    Turn 1 seeds a fact; turn 2 asks for it back. If the provider drops
    history (bad session handling in chak's converter) this will regress
    with a wrong or refusing answer.
    """
    conv = _make_conv(uri, key, title="multi-turn")
    await conv.asend(
        "My favorite color is periwinkle. Acknowledge briefly.", timeout=60,
    )
    resp = await conv.asend(
        "What color did I just say was my favorite? "
        "Answer with only the single word.", timeout=60,
    )
    body = (resp.content or "").lower()
    assert "periwinkle" in body, f"context lost, model said: {resp.content!r}"
    return CheckResult(
        "multi-turn", "PASS",
        detail=f"{len(conv.messages)} messages retained",
    )


async def check_usage_tokens(uri: str, key: str) -> CheckResult:
    """Normalized usage bucket must be populated.

    If usage is missing, cost/budget dashboards downstream silently break.
    """
    conv = _make_conv(uri, key, title="usage-tokens")
    resp = await conv.asend("Reply with one short sentence.", timeout=60)
    usage = resp.metadata.usage
    assert usage is not None, "metadata.usage is None"
    assert usage.total_tokens > 0, f"total_tokens = {usage.total_tokens}"
    assert usage.prompt_tokens >= 0 and usage.completion_tokens > 0
    return CheckResult(
        "usage-tokens", "PASS",
        detail=(
            f"prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} "
            f"total={usage.total_tokens}"
        ),
    )


async def check_tool_call_function(uri: str, key: str) -> CheckResult:
    """Plain-function tool call via wrap_tools + event stream."""

    def add_numbers(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        return a + b

    conv = _make_conv(
        uri, key,
        title="tool-call-function",
        tools=wrap_tools([add_numbers]),
        timeout=120,
    )
    events: list[Any] = []
    async for ev in await conv.asend(
        "You must call the add_numbers tool with a=7 and b=5. "
        "Do not answer directly before calling it.",
        event=True, timeout=120,
    ):
        events.append(ev)

    starts = [e for e in events if isinstance(e, chak.ToolCallStartEvent)]
    ok = [e for e in events if isinstance(e, chak.ToolCallSuccessEvent)]
    assert any(e.tool_name == "add_numbers" for e in starts), \
        f"add_numbers was never invoked (starts: {[e.tool_name for e in starts]})"
    assert any(e.tool_name == "add_numbers" for e in ok), \
        "add_numbers invoked but never succeeded"
    return CheckResult(
        "tool-call-function", "PASS",
        detail=f"{len(starts)} calls, {len(ok)} success",
    )


async def check_tool_call_object(uri: str, key: str) -> CheckResult:
    """Object-method tool call — verifies wrap_tools works on instances."""

    class Calculator:
        def multiply(self, a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

    conv = _make_conv(
        uri, key,
        title="tool-call-object",
        tools=wrap_tools([Calculator()]),
        timeout=120,
    )
    events: list[Any] = []
    async for ev in await conv.asend(
        "You must call the calculator-multiply tool with a=6 and b=7. "
        "Do not answer directly before calling it.",
        event=True, timeout=120,
    ):
        events.append(ev)

    ok = [e for e in events if isinstance(e, chak.ToolCallSuccessEvent)]
    assert any(e.tool_name == "calculator-multiply" for e in ok), \
        "calculator-multiply never succeeded"
    return CheckResult("tool-call-object", "PASS", detail=f"{len(ok)} success")


async def check_structured_single(uri: str, key: str) -> CheckResult:
    """returns=CityInfo → single pydantic instance."""
    conv = _make_conv(uri, key, title="structured-single", timeout=90)
    result = await conv.asend(
        "Return structured data for Paris, France.",
        returns=CityInfo, timeout=90,
    )
    assert isinstance(result, CityInfo), f"got {type(result).__name__}"
    assert result.city and result.country
    return CheckResult(
        "structured-single", "PASS",
        detail=f"city={result.city!r} country={result.country!r}",
    )


async def check_structured_list(uri: str, key: str) -> CheckResult:
    """returns=list[CityInfo] → list of pydantic instances."""
    conv = _make_conv(uri, key, title="structured-list", timeout=90)
    result = await conv.asend(
        "Return structured data for exactly two cities: "
        "Paris, France and Tokyo, Japan.",
        returns=list[CityInfo], timeout=90,
    )
    assert isinstance(result, list), f"got {type(result).__name__}"
    assert len(result) == 2, f"expected 2 items, got {len(result)}"
    assert all(isinstance(x, CityInfo) for x in result)
    return CheckResult(
        "structured-list", "PASS",
        detail=", ".join(f"{x.city}/{x.country}" for x in result),
    )


async def check_structured_with_tools(uri: str, key: str) -> CheckResult:
    """Two-phase flow: tool call, then structured extraction.

    This is the most demanding shape — many providers pass the individual
    tests but stumble on the combined flow (wrong finish_reason handling,
    stale tool_choice, missing response_format on turn 2, etc.).
    """

    def lookup_city(name: str) -> str:
        """Return a fixed lookup so the assertion is deterministic."""
        return f"City: {name}, Country: France, Population: 2M"

    conv = _make_conv(
        uri, key,
        title="structured-with-tools",
        tools=wrap_tools([lookup_city]),
        timeout=120,
    )
    result = await conv.asend(
        "Look up information about Paris using the lookup_city tool, "
        "then return structured data.",
        returns=CityInfo, timeout=120,
    )
    assert isinstance(result, CityInfo), f"got {type(result).__name__}"
    assert result.city, "empty city in extracted result"
    return CheckResult(
        "structured-with-tools", "PASS",
        detail=f"city={result.city!r} country={result.country!r}",
    )


async def check_reasoning(uri: str, key: str) -> CheckResult:
    """Reasoning mode — provider decides whether to expose thinking traces."""
    conv = _make_conv(uri, key, title="reasoning", timeout=120)
    resp = await conv.asend(
        "How many prime numbers are there between 1 and 10? "
        "Think step by step, then answer with a single integer.",
        reasoning=chak.Reasoning(effort="medium", summary="auto"),
        timeout=120,
    )
    assert resp.content, "empty content"
    # reasoning_content is provider-optional; presence is a soft signal, not a
    # hard requirement (some providers gate this behind a flag or model tier).
    has_reasoning = bool(getattr(resp, "reasoning_content", None))
    return CheckResult(
        "reasoning", "PASS",
        detail=(
            f"reasoning_content={'yes' if has_reasoning else 'no (soft)'}, "
            f"answer_len={len(resp.content)}"
        ),
    )


async def check_cache(uri: str, key: str) -> CheckResult:
    """Two-turn cache probe on a >=1024-token system prompt.

    Turn 1 should write the cache; turn 2 should read it back. We accept EITHER
    signal (cache_creation on turn 1, or cache_read on turn 2) as PASS, since
    OpenAI's automatic caching doesn't expose cache_creation counters.
    """
    conv = _make_conv(
        uri, key,
        title="cache",
        system_prompt=_CACHE_SYSTEM_PROMPT,
        cache=chak.Cache(system_prompt=True, key="chak-check:cache-probe-v1"),
        timeout=90,
    )
    turn1 = await conv.asend("Write a one-line Python fibonacci function.", timeout=90)
    turn2 = await conv.asend("Now add memoization.", timeout=90)

    u1 = turn1.metadata.usage
    u2 = turn2.metadata.usage
    assert u1 and u2, "usage missing on cache probe"

    write = u1.cache_creation_input_tokens + u2.cache_creation_input_tokens
    read = u1.cache_read_input_tokens + u2.cache_read_input_tokens
    if write == 0 and read == 0:
        # Not a hard fail — some providers just don't implement caching. But
        # since the user opted in, surface it clearly.
        return CheckResult(
            "cache", "FAIL",
            detail="no cache activity (write=0, read=0); "
                   "provider may not support prompt caching",
        )
    return CheckResult(
        "cache", "PASS",
        detail=f"cache_write={write}, cache_read={read}",
    )


# ── inspector integration ──────────────────────────────────────────────────


def _maybe_start_inspector(
    enabled: bool,
    port: int,
    open_browser: bool,
) -> tuple[bool, str]:
    """Turn on inspector global auto-attach if requested.

    Uses ``watch()`` (no args) so every Conversation created by the checks
    below is picked up automatically — no per-check ``watch(conv)`` calls
    scattered through the file.

    Returns:
        (started, url_or_reason). ``started`` is True on success. The second
        element is the inspector URL when started, or a human-readable reason
        when skipped/failed. We never abort the run over inspector issues:
        even without live observation, the pass/fail report is what matters.
    """
    if not enabled:
        return False, "disabled via --no-inspect"
    try:
        from chak.inspector import watch
    except ImportError as e:
        # fastapi / uvicorn aren't installed — inspector is an optional
        # extra (`pip install 'chakpy[server]'`). Degrade to a plain run.
        return False, f"inspector deps missing ({e}); install 'chakpy[server]'"
    try:
        watch(port=port, open_browser=open_browser)
    except OSError as e:
        # Port already bound (usually another chak process). Surface but
        # don't abort — the user can retry with a different port later.
        return False, f"could not bind port {port} ({e})"
    return True, f"http://127.0.0.1:{port}"


# ── driver ─────────────────────────────────────────────────────────────────


def _build_registry(with_reasoning: bool, with_cache: bool) -> Registry:
    """Assemble the ordered check list. Opt-in checks are appended last."""
    reg = Registry()
    reg.add("sync-nonstream", check_sync_nonstream)
    reg.add("async-nonstream", check_async_nonstream)
    reg.add("sync-stream", check_sync_stream)
    reg.add("async-stream", check_async_stream)
    reg.add("event-stream", check_event_stream)
    reg.add("multi-turn", check_multi_turn)
    reg.add("usage-tokens", check_usage_tokens)
    reg.add("tool-call-function", check_tool_call_function)
    reg.add("tool-call-object", check_tool_call_object)
    reg.add("structured-single", check_structured_single)
    reg.add("structured-list", check_structured_list)
    reg.add("structured-with-tools", check_structured_with_tools)
    if with_reasoning:
        reg.add("reasoning", check_reasoning)
    if with_cache:
        reg.add("cache", check_cache)
    return reg


def _resolve_api_key(uri: str, explicit_key: Optional[str],
                     explicit_env: Optional[str]) -> tuple[str, str]:
    """Return (api_key, source_description).

    Precedence: --api-key > --api-key-env > <PROVIDER>_API_KEY.
    """
    if explicit_key:
        return explicit_key, "flag --api-key"

    if explicit_env:
        val = os.getenv(explicit_env, "")
        if not val:
            raise SystemExit(f"error: env var {explicit_env} is empty or unset")
        return val, f"env {explicit_env}"

    parsed = parse_uri(uri)
    default_env = f"{parsed['provider'].upper()}_API_KEY"
    val = os.getenv(default_env, "")
    if not val:
        raise SystemExit(
            f"error: no API key found. Tried env {default_env}. "
            f"Pass --api-key-env <NAME> or --api-key <VALUE>."
        )
    return val, f"env {default_env}"


def _parse_filter(raw: Optional[str]) -> Optional[set[str]]:
    """Split a comma/whitespace list. Also accepts short prefixes ('tool')."""
    if not raw:
        return None
    return {tok.strip() for tok in raw.replace(",", " ").split() if tok.strip()}


def _matches(name: str, filters: set[str]) -> bool:
    """A check matches if its name equals or starts with any filter token."""
    return any(name == f or name.startswith(f) for f in filters)


async def _run_check(name: str, fn, uri: str, key: str,
                     timeout_s: float) -> CheckResult:
    """Run a single check with per-check timeout + exception firewall."""
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(fn(uri, key), timeout=timeout_s)
    except asyncio.TimeoutError:
        return CheckResult(
            name, "FAIL",
            elapsed=time.monotonic() - start,
            detail=f"per-check timeout after {timeout_s:.0f}s",
        )
    except AssertionError as e:
        return CheckResult(
            name, "FAIL",
            elapsed=time.monotonic() - start,
            detail=f"assertion: {e}",
            error=e,
        )
    except BaseException as e:  # noqa: BLE001 — we want to swallow *everything*
        return CheckResult(
            name, "FAIL",
            elapsed=time.monotonic() - start,
            detail=f"{type(e).__name__}: {e}",
            error=e,
        )
    result.elapsed = time.monotonic() - start
    return result


def _print_header(
    uri: str,
    key_src: str,
    checks: list[str],
    inspector_status: str,
) -> None:
    print("=" * 72)
    print(f"  Chak provider check")
    print("=" * 72)
    parsed = parse_uri(uri)
    print(f"  URI        : {uri}")
    print(f"  Provider   : {parsed['provider']}")
    print(f"  Model      : {parsed['model']}")
    if parsed.get("base_url"):
        print(f"  Base URL   : {parsed['base_url']}")
    print(f"  API key    : {key_src}")
    print(f"  Inspector  : {inspector_status}")
    print(f"  Checks     : {len(checks)} → {', '.join(checks)}")
    print("=" * 72)


def _print_row(idx: int, total: int, r: CheckResult) -> None:
    tag = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "SKIP": "⏭  SKIP"}[r.status]
    print(f"  [{idx:2d}/{total}] {r.name:<24} {tag}  "
          f"({r.elapsed:5.2f}s)  {r.detail}")


def _print_summary(results: list[CheckResult], verbose: bool) -> None:
    passed = [r for r in results if r.status == "PASS"]
    failed = [r for r in results if r.status == "FAIL"]
    skipped = [r for r in results if r.status == "SKIP"]

    print()
    print("=" * 72)
    print(
        f"  Summary: {len(passed)} passed, {len(failed)} failed, "
        f"{len(skipped)} skipped   (total {sum(r.elapsed for r in results):.1f}s)"
    )
    print("=" * 72)

    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  ❌ {r.name}: {r.detail}")
            if verbose and r.error is not None:
                # Only print the traceback in verbose mode — the one-line
                # detail is usually enough and keeps the report readable.
                tb = "".join(traceback.format_exception(
                    type(r.error), r.error, r.error.__traceback__
                ))
                print("     " + "\n     ".join(tb.rstrip().splitlines()))


async def _main_async(args: argparse.Namespace) -> int:
    api_key, key_src = _resolve_api_key(args.model_uri, args.api_key, args.api_key_env)

    # Start the inspector BEFORE building any Conversations so global
    # auto-attach picks them all up. Failure here is soft — the report
    # is still the primary output and we don't want a missing optional
    # dep to break users' automation.
    inspector_started, inspector_status = _maybe_start_inspector(
        enabled=args.inspect,
        port=args.inspector_port,
        open_browser=not args.no_open_browser,
    )

    reg = _build_registry(
        with_reasoning=args.with_reasoning,
        with_cache=args.with_cache,
    )

    only = _parse_filter(args.only)
    skip = _parse_filter(args.skip)

    # Compute the actual run list: apply --only first (whitelist), then --skip.
    active: list[str] = []
    for name in reg.order:
        if only is not None and not _matches(name, only):
            continue
        if skip is not None and _matches(name, skip):
            continue
        active.append(name)

    _print_header(args.model_uri, key_src, active, inspector_status)

    results: list[CheckResult] = []
    # Report SKIPs for the filtered-out checks first so the run list feels honest.
    for name in reg.order:
        if name not in active:
            reason = "filtered by --only" if (only and not _matches(name, only)) \
                else "filtered by --skip"
            results.append(CheckResult(name, "SKIP", detail=reason))

    # Sequential run — parallel would blow through rate limits and make the
    # elapsed column meaningless.
    total = len(active)
    for i, name in enumerate(active, 1):
        result = await _run_check(
            name, reg.checks[name], args.model_uri, api_key, args.timeout,
        )
        results.append(result)
        _print_row(i, total, result)
        if args.stop_on_fail and result.status == "FAIL":
            print("\n  --stop-on-fail set, aborting remaining checks.")
            break

    # Restore original registry order for the summary
    order_index = {n: i for i, n in enumerate(reg.order)}
    results.sort(key=lambda r: order_index.get(r.name, 1_000))

    _print_summary(results, verbose=args.verbose)

    # When the inspector is up, let the user browse before the process exits.
    # Without this, the SSE stream tears down the instant checks finish and
    # partial state (final replies, ephemeral chunks) vanishes from the page.
    if inspector_started and args.hold_inspector:
        print()
        print(f"  Inspector still running at {inspector_status}")
        print("  Press Ctrl+C to exit (or pass --no-hold-inspector to exit immediately).")
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n  Bye.")

    return 0 if not any(r.status == "FAIL" for r in results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run chak's full capability suite against a single model_uri.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_uri", help="chak model URI, e.g. 'moonshot/kimi-k3'")
    parser.add_argument(
        "--api-key", default=None,
        help="Inline API key (overrides env). Prefer --api-key-env for secrets.",
    )
    parser.add_argument(
        "--api-key-env", default=None,
        help="Env var name holding the API key "
             "(default: <PROVIDER>_API_KEY derived from URI).",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated list of check name prefixes to run (whitelist). "
             "Example: --only tool-call,structured",
    )
    parser.add_argument(
        "--skip", default=None,
        help="Comma-separated list of check name prefixes to skip (blacklist).",
    )
    parser.add_argument(
        "--with-cache", action="store_true",
        help="Include the prompt-caching probe "
             "(Anthropic / OpenAI-capable models only).",
    )
    parser.add_argument(
        "--with-reasoning", action="store_true",
        help="Include the reasoning probe "
             "(o-series / QwQ / R1 / Sonnet-thinking, etc.).",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0,
        help="Per-check hard timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--stop-on-fail", action="store_true",
        help="Abort remaining checks after the first failure.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print full tracebacks for failed checks.",
    )
    # ── inspector flags ────────────────────────────────────────────────
    parser.add_argument(
        "--inspect", action=argparse.BooleanOptionalAction, default=True,
        help="Start chak.inspector and auto-attach every conversation "
             "(default: on). Use --no-inspect for CI / headless runs.",
    )
    parser.add_argument(
        "--inspector-port", type=int, default=7878,
        help="HTTP port for the inspector page (default: 7878).",
    )
    parser.add_argument(
        "--no-open-browser", action="store_true",
        help="Don't auto-open the inspector URL in the default browser. "
             "The server still starts; just print the URL.",
    )
    parser.add_argument(
        "--hold-inspector", action=argparse.BooleanOptionalAction, default=True,
        help="After checks finish, keep the process alive so you can browse "
             "final state in the inspector (default: on). Off-switch for CI.",
    )
    args = parser.parse_args()

    try:
        exit_code = asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        exit_code = 130

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
