"""
Chat With PDF Agent — Multi-turn with Intelligent Context Management
====================================================================

A practical PDF Q&A agent that sustains multi-turn conversations about a
large PDF without blowing up the context window.

Key innovations
---------------
1. **In-memory scratchpad** — the LLM saves distilled findings (with page
   ranges) as it reads.  This is its external memory layer.
2. **Contract-driven context handler** — mechanically offloads consumed
   PDF page content once the LLM has committed a note.  No LLM calls, no
   summarization ambiguity.
3. **Token / context observability** — every turn prints context size,
   token breakdown, and resolved model.

How context is managed
----------------------
* **Intra-turn** (``handle_round``): inside a single ``asend()`` tool loop,
  once the LLM calls ``scratchpad_save_note(...)``, every *older* tool
  cycle's bulky ToolMessages are replaced with compact stubs.  Cycles
  after the last save are kept verbatim (freshly read, not yet digested).
* **Inter-turn** (``handle_turn``): completed turns are offloaded — only
  the HumanMessage + final AIMessage survive, plus a placeholder pointing
  the LLM back to its scratchpad.

Tools: Pdf, Scratchpad (save_section / read_section / list_sections /
               remove_section / search_sections)
"""

import argparse
import asyncio
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import tiktoken

import dotenv
dotenv.load_dotenv()

# Silence chak's INFO-level tool-call logs so the console stays clean.
# Our rich panels handle all tool-call visualization.
os.environ.setdefault("CHAK_LOG_LEVEL", "WARNING")

import chak
from chak.tools.std import Pdf
from chak.context.handlers.base import BaseContextHandler
from chak.message import (
    AIMessage,
    HumanMessage,
    Message,
    SystemMessage,
    ToolMessage,
    MessageChunk,
    ToolCallStartEvent,
    ToolCallSuccessEvent,
    ToolCallErrorEvent,
)

# ── Config ──────────────────────────────────────────────────────────────────

_MODEL = "deepseek/deepseek-v4-pro"
_API_KEY = os.getenv("DEEPSEEK_API_KEY")
_CONTEXT_WINDOW = 128_000  # deepseek-v4-pro context limit (tokens)

# Default PDF source.  Override via CLI arg — accepts both remote URLs and
# local file paths, e.g.:
#   python examples/agents/chat_with_pdf.py
#   python examples/agents/chat_with_pdf.py /path/to/local.pdf
#   python examples/agents/chat_with_pdf.py https://example.com/doc.pdf
_DEFAULT_PDF = (
    "https://wcbpub.oss-cn-hangzhou.aliyuncs.com/xue/xxxx/SellingGuide.pdf"
)

# ── Token estimation helpers ────────────────────────────────────────────────

_enc: Optional[tiktoken.Encoding] = None


def _get_enc() -> tiktoken.Encoding:
    """Lazy-init the tiktoken encoder (cl100k_base ≈ safe for DeepSeek)."""
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _estimate_tokens(messages: List[Message]) -> int:
    """Rough total token estimate for a message list."""
    enc = _get_enc()
    total = 0
    for msg in messages:
        content = (
            msg.content if isinstance(msg.content, str)
            else str(msg.content or "")
        )
        total += len(enc.encode(content)) + 4  # per-message overhead
    return total


# ============================================================================
# 1.  In-memory Scratchpad  (the LLM's external memory layer)
# ============================================================================

class Scratchpad:
    """In-memory notebook with page-range indexing.

    Each section is a ``(heading, page_range, content)`` entry capturing
    distilled findings from PDF pages.  The LLM owns the save cadence —
    the context handler uses ``scratchpad-save_section`` calls as the
    signal that bulky tool results have been consumed.

    Methods are auto-discovered by ``NativeObjectTool`` and namespaced as
    ``scratchpad-{method_name}`` (e.g. ``scratchpad-save_section``).
    """

    _LINE_SOFT_LIMIT = 30

    def __init__(self):
        # heading -> {"page_range": str, "content": str}
        self._sections: dict[str, dict] = {}

    def __available__(self) -> frozenset:
        """Expose all CRUD methods as LLM tools."""
        return frozenset({
            "list_sections", "read_section", "save_section",
            "remove_section", "search_sections",
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_toc(self) -> str:
        """Format a table-of-contents string."""
        if not self._sections:
            return "(empty — no sections saved yet)"
        lines: list[str] = []
        for idx, (h, s) in enumerate(self._sections.items(), 1):
            n_lines = len(s["content"].splitlines()) if s["content"] else 0
            lines.append(f'{idx}. "{h}" (pages {s["page_range"]}, {n_lines} lines)')
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API — each method is exposed as an LLM tool
    # ------------------------------------------------------------------

    def list_sections(self) -> str:
        """List all section headings with page ranges and line counts.

        Returns a lightweight table of contents of everything stored in
        the scratchpad.  Use this to orient yourself before deciding
        which section to read.  Extremely cheap — call it any time you
        need a reminder of what you have already noted down.
        """
        return self._format_toc()

    def read_section(self, heading: str) -> str:
        """Read the content of ONE specific section by its heading.

        Use this to recall previously saved findings on a specific topic.
        Only read sections you actually need right now — do NOT read all
        sections sequentially; that defeats the purpose of section-based
        memory.

        Args:
            heading: The section title, exactly as shown in
                     ``list_sections`` output.
        """
        heading = heading.strip()
        if not heading:
            return "Error: heading must not be empty."
        s = self._sections.get(heading)
        if not s:
            avail = list(self._sections.keys())
            if avail:
                return f"Error: section '{heading}' not found.  Available: {avail}"
            return f"Error: section '{heading}' not found (scratchpad is empty)."
        return f"[pages {s['page_range']}]\n{s['content']}"

    def save_section(self, heading: str, page_range: str, content: str) -> str:
        """Create or overwrite a section with distilled findings.

        Call this IMMEDIATELY after reading PDF pages and extracting key
        information.  This is your external memory — once saved, the
        original page content will be removed from context to free space.
        "Use it or lose it."

        **Best practices:**
        - Store conclusions and key facts, NOT raw document text.
        - Keep it short: ideally 5–30 lines per section.
        - Include short verbatim quotes (1–2 sentences) for citation.
        - No need to call ``list_sections`` after saving — the response
          already includes the updated table of contents.

        Args:
            heading: Section title.  Use descriptive snake_case names
                     (e.g. ``homeready_dti``, ``borrower_eligibility``).
            page_range: PDF physical page range you read, e.g. "800-805".
            content: Distilled findings — specific requirements, numbers,
                     and key phrases you will need for your final answer.
        """
        heading = heading.strip()
        if not heading:
            return "Error: heading must not be empty."
        content = content.strip()
        if not content:
            return "Error: content must not be empty."

        replaced = heading in self._sections
        self._sections[heading] = {
            "page_range": page_range.strip() or "?",
            "content": content,
        }

        n_lines = len(content.splitlines())
        action = "Updated" if replaced else "Saved new"
        msg = f"{action} section '{heading}' (pages {page_range}, {n_lines} lines)."
        if n_lines > self._LINE_SOFT_LIMIT:
            msg += (
                f" ⚠️ Warning: section is {n_lines} lines "
                f"(recommended ≤{self._LINE_SOFT_LIMIT}). "
                "Consider splitting or summarizing further."
            )
        msg += f"\n\nCurrent sections:\n{self._format_toc()}"
        return msg

    def remove_section(self, heading: str) -> str:
        """Delete a section that is no longer needed.

        Use this to clean up stale or superseded notes.  Keeping the
        scratchpad lean makes ``list_sections`` more useful.

        Args:
            heading: Section title.
        """
        heading = heading.strip()
        if not heading:
            return "Error: heading must not be empty."
        if heading not in self._sections:
            avail = list(self._sections.keys())
            if avail:
                return f"Error: section '{heading}' not found.  Available: {avail}"
            return f"Error: section '{heading}' not found (scratchpad is empty)."
        del self._sections[heading]
        return f"Removed section '{heading}'.\n\nCurrent sections:\n{self._format_toc()}"

    def search_sections(self, keyword: str) -> str:
        """Search saved sections by keyword (matches headings and content).

        Useful when you are not sure which section heading to read.

        Args:
            keyword: Search term to look for across all sections.
        """
        kw = keyword.strip().lower()
        if not kw:
            return "Error: keyword must not be empty."
        matches = [
            (h, s) for h, s in self._sections.items()
            if kw in h.lower() or kw in s["content"].lower()
        ]
        if not matches:
            return f"No sections matching '{keyword}'."
        lines = []
        for h, s in matches:
            snippet = s["content"][:300]
            if len(s["content"]) > 300:
                snippet += "..."
            lines.append(f'  - "{h}" (pages {s["page_range"]}): {snippet}')
        return f"Found {len(matches)} section(s):\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Debug-only — NOT exposed to the LLM (absent from __available__)
    # ------------------------------------------------------------------

    def dump(self, path: str | Path) -> str:
        """Dump all sections to a human-readable Markdown file.

        Intended for post-run inspection — lets you see exactly what the
        LLM chose to persist throughout the conversation.  Not exposed as
        a tool (not listed in ``__available__``).

        Args:
            path: Output file path (``.md`` recommended).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# Scratchpad Dump", f"\n{len(self._sections)} section(s).\n"]
        for heading, s in self._sections.items():
            lines.append(f"## {heading}")
            lines.append(f"*Pages: {s['page_range']}*\n")
            lines.append(s["content"])
            lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")
        return f"Dumped {len(self._sections)} section(s) to {p}"


# ============================================================================
# 2.  PdfContextHandler  (contract-driven mechanical offloader)
# ============================================================================

# NativeObjectTool namespaces methods as "{class_name}-{method}".
# Scratchpad.save_section → "scratchpad-save_section".
#
# Any tool call starting with this prefix marks a cleanup boundary —
# the LLM has committed findings, so older bulky tool results are fair
# game for stubbing.
_SAVE_PREFIX = "scratchpad-save"

# Tools whose results represent the LLM's own memory access.  These are
# never stubbed: telling the model its own notes are "consumed" triggers
# destructive re-save loops.
_MEMORY_PREFIX = "scratchpad-"

# ToolMessage content above this token count is eligible for stubbing.
_STUB_THRESHOLD = 2000

# Placeholder inserted after each offloaded inter-turn round.
_OFFLOAD_PLACEHOLDER = SystemMessage(
    content=(
        "[Context compression] Tool results from this turn were consumed. "
        "Key findings should be in your scratchpad. Use "
        "scratchpad-list_sections to see what is available, then "
        "scratchpad-read_section(heading) for specifics."
    )
)

_STUB_TPL = (
    "[offloaded] {tool}({args}) returned ~{tokens} tokens. "
    "Content consumed — check scratchpad notes. "
    "Do not re-call unless you need different info."
)

# Framework-level error results (invalid JSON args, tool-not-found, etc.)
# can be huge because chak echoes full arguments back.  Stub them
# regardless of the save contract.
_ERROR_PREFIX = "Error: "


class PdfContextHandler(BaseContextHandler):
    """Mechanical offloader tailored for PDF research-loop agents.

    Two compression modes (both zero-LLM):

    * **handle_round** (intra-turn): once the LLM saves a scratchpad note,
      every older cycle's bulky ToolMessages (> ``stub_threshold_tokens``)
      are replaced with compact stubs.  AIMessage(tool_calls) are always
      preserved so the LLM sees its own call history.  Cycles after the
      last save are kept verbatim.
    * **handle_turn** (inter-turn): completed turns are offloaded —
      only the HumanMessage + final AIMessage survive per turn, plus a
      placeholder.  The most recent turn is always kept intact.
    """

    def __init__(self, stub_threshold_tokens: int = _STUB_THRESHOLD):
        super().__init__()
        self.stub_threshold_tokens = stub_threshold_tokens

    # ── Intra-turn: stub stale tool results ─────────────────────────────

    def handle_round(
        self,
        messages: List[Message],
        *,
        conversation_id: str = "",
        round_index: int = 0,
    ) -> List[Message]:
        """Compress intra-turn tool-loop history.

        Contract: the LLM promises to save findings via
        ``scratchpad_save_note(...)`` as soon as it digests a read.  Once
        we see such a save, every earlier ToolMessage above the size
        threshold is stubbed.  Cycles after the last save are kept
        verbatim (freshly read, not yet digested).
        """
        cycles = self._split_cycles(messages)
        if len(cycles) <= 2:
            return messages

        prefix = cycles[0]
        tool_cycles = cycles[1:]

        # Find the LAST cycle containing a scratchpad_save call.
        last_save = -1
        for i, cycle in enumerate(tool_cycles):
            if self._cycle_has_save(cycle):
                last_save = i

        result: List[Message] = list(prefix)
        if last_save < 0:
            # No save yet — only collapse bulky error results.
            for cycle in tool_cycles:
                result.extend(self._stub_errors_only(cycle))
            return result

        for i, cycle in enumerate(tool_cycles):
            if i <= last_save:
                result.extend(self._stub_cycle(cycle))
            else:
                result.extend(self._stub_errors_only(cycle))
        return result

    @staticmethod
    def _cycle_has_save(cycle: List[Message]) -> bool:
        """True if the cycle's AIMessage calls scratchpad_save_*."""
        for msg in cycle:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name.startswith(_SAVE_PREFIX):
                        return True
        return False

    def _stub_errors_only(self, cycle: List[Message]) -> List[Message]:
        """Stub only bulky error results; keep everything else."""
        result: List[Message] = []
        for msg in cycle:
            if isinstance(msg, ToolMessage):
                content = str(msg.content or "")
                if (
                    content.startswith(_ERROR_PREFIX)
                    and self._token_count(content) > self.stub_threshold_tokens
                ):
                    result.append(self._make_stub(msg, cycle))
                    continue
            result.append(msg)
        return result

    def _stub_cycle(self, cycle: List[Message]) -> List[Message]:
        """Stub bulky ToolMessages in a digested cycle; keep AIMessages."""
        result: List[Message] = []
        for msg in cycle:
            if isinstance(msg, ToolMessage):
                tool_name, _ = self._lookup_tool_info(cycle, msg)
                # Memory-access tools are exempt from stubing.
                if tool_name.startswith(_MEMORY_PREFIX):
                    result.append(msg)
                    continue
                content = str(msg.content or "")
                if self._token_count(content) > self.stub_threshold_tokens:
                    result.append(self._make_stub(msg, cycle))
                else:
                    result.append(msg)
            else:
                result.append(msg)
        return result

    # ── Inter-turn: offload completed rounds ────────────────────────────

    def handle_turn(
        self,
        messages: List[Message],
        *,
        conversation_id: str = "",
    ) -> List[Message]:
        """Offload tool results from completed turns between asend() calls.

        Keeps the HumanMessage + final AIMessage of each completed turn
        (the LLM's own answer is valuable context), drops all intermediate
        ToolMessages and tool-calling AIMessages, and inserts a
        placeholder pointing to the scratchpad.  The most recent turn is
        always preserved intact.
        """
        if not messages:
            return []

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        conv_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        # Find round boundaries: each boundary = (HumanMessage, final AIMessage)
        boundaries: list[Tuple[int, int]] = []
        i = len(conv_msgs) - 1
        while i >= 0:
            msg = conv_msgs[i]
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                round_end = i
                round_start = round_end
                for j in range(round_end - 1, -1, -1):
                    if isinstance(conv_msgs[j], HumanMessage):
                        round_start = j
                        break
                boundaries.append((round_start, round_end))
                i = round_start - 1
            else:
                i -= 1
        boundaries.reverse()

        if not boundaries:
            return messages

        # Always offload turns BEFORE the last boundary; keep the last
        # turn intact as "recent context" for the upcoming round.
        result: List[Message] = list(system_msgs)

        for rs, re_ in boundaries[:-1]:
            for idx in range(rs, re_ + 1):
                msg = conv_msgs[idx]
                # Drop consumed tool results and their tool-calling AIMessages.
                if isinstance(msg, ToolMessage):
                    continue
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    continue
                result.append(msg)
            result.append(_OFFLOAD_PLACEHOLDER)

        # Keep the last turn (and anything trailing it) intact.
        last_start = boundaries[-1][0]
        for idx in range(last_start, len(conv_msgs)):
            result.append(conv_msgs[idx])

        return result

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _lookup_tool_info(
        cycle: List[Message], tool_msg: ToolMessage
    ) -> Tuple[str, str]:
        """Find tool name + arguments for a ToolMessage by tool_call_id."""
        for msg in cycle:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id == tool_msg.tool_call_id:
                        return tc.function.name, tc.function.arguments
        return "", ""

    @staticmethod
    def _make_stub(
        tool_msg: ToolMessage, cycle: List[Message]
    ) -> ToolMessage:
        """Replace bulky ToolMessage content with a compact stub."""
        tool_name, args_hint = PdfContextHandler._lookup_tool_info(
            cycle, tool_msg
        )
        content = str(tool_msg.content or "")
        tokens = len(_get_enc().encode(content))
        stub = _STUB_TPL.format(
            tool=tool_name or "unknown",
            args=(args_hint or "")[:200],
            tokens=tokens,
        )
        return tool_msg.model_copy(update={"content": stub})

    @staticmethod
    def _token_count(text: str) -> int:
        return len(_get_enc().encode(text))

    @staticmethod
    def _split_cycles(messages: List[Message]) -> List[List[Message]]:
        """Split into [prefix, cycle1, cycle2, ...].

        prefix = system + human + leading AI text (no tool_calls).
        Each cycle = AIMessage(tool_calls) + its ToolMessages.
        """
        groups: List[List[Message]] = [[]]
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                groups.append([msg])
            elif isinstance(msg, ToolMessage) and len(groups) > 1:
                groups[-1].append(msg)
            else:
                groups[-1].append(msg)
        return groups


# ============================================================================
# 3.  Rich-powered UI helpers
# ============================================================================

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

_console = Console(theme=Theme({
    "user": "bold cyan",
    "assistant": "bold green",
    "stats": "bold yellow",
    "tool": "dim white",
    "error": "bold red",
}))


def _print_user(msg: str):
    """Render user message as a bordered panel."""
    _console.print(Panel(msg, title="You", title_align="left",
                         border_style="user", expand=False))


def _print_assistant(text: str):
    """Render assistant's final answer as a bordered panel."""
    _console.print(Panel(
        text.rstrip() or "(empty)",
        title="Assistant",
        title_align="left",
        border_style="assistant",
    ))


def _print_tool_call(name: str, hint: str):
    """Compact one-line tool call indicator."""
    _console.print(f"  [{name}] {hint}", style="tool")


def _print_tool_result(name: str, result: str, ok: bool = True):
    """Compact tool result preview."""
    preview = (result or "")[:180].replace("\n", " ")
    if ok:
        _console.print(f"    -> {preview}", style="tool")
    else:
        _console.print(f"    -> ERROR: {preview}", style="error")


def _print_turn_stats(
    handler: PdfContextHandler,
    history_msgs: List[Message],
    turn_msgs: List[Message],
    turn_num: int,
):
    """Render turn stats as a rich table inside a panel."""
    ctx_tokens = _estimate_tokens(handler.output_messages)
    history_tokens = _estimate_tokens(history_msgs)
    ctx_msg_count = len(handler.output_messages)
    history_count = len(history_msgs)
    pct = ctx_tokens / _CONTEXT_WINDOW * 100

    total_in = total_out = total_cr = total_cw = grand_total = 0
    models: set[str] = set()
    for msg in turn_msgs:
        if isinstance(msg, AIMessage) and msg.metadata:
            u = msg.metadata.usage
            if u:
                total_in += u.prompt_tokens
                total_out += u.completion_tokens
                total_cr += u.cache_read_input_tokens
                total_cw += u.cache_creation_input_tokens
                grand_total += u.total_tokens
            pt = msg.metadata.provider_trace
            if pt and pt.resolved_provider:
                models.add(f"{pt.resolved_provider}/{pt.resolved_model}")

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="stats", no_wrap=True)
    table.add_column()

    table.add_row(
        "Context:",
        f"{ctx_tokens / 1000:.1f}K / {_CONTEXT_WINDOW // 1000}K ({pct:.1f}%)"
        f"  [history: {history_count} msgs / {history_tokens / 1000:.1f}K"
        f" -> sent: {ctx_msg_count} msgs]",
    )
    table.add_row(
        "Tokens:",
        f"in={total_in:,}  out={total_out:,}  cache_r={total_cr:,}"
        f"  cache_w={total_cw:,}  total={grand_total:,}",
    )
    if models:
        table.add_row("Model:", ", ".join(sorted(models)))

    _console.print(Panel(
        table,
        title=f"Turn #{turn_num} Stats",
        title_align="left",
        border_style="stats",
    ))


# ============================================================================
# 4.  Multi-turn main loop
# ============================================================================

_SYSTEM_PROMPT = """\
You are a meticulous document analyst working with large PDFs.

## Core workflow
1. Use Pdf metadata first to understand document size and structure.
2. Use Pdf search to locate relevant sections.
3. Read only the specific pages you need.
4. **Immediately** save distilled findings to your scratchpad via
   scratchpad-save_section(heading, page_range, content) BEFORE reading
   more pages.
5. Answer with precise citations to PDF physical page numbers.

## Scratchpad discipline — MANDATORY
Your reading history is NOT permanent.  The system periodically
**removes large tool results** from your context.  **Only the scratchpad
survives.**  If you read a page and then the content gets pruned before
you saved a note, that information is LOST and you will have to re-read
the page — wasting time, tokens, and risking a loop.

### Rule A — Save what you read, IMMEDIATELY.
Every time you call `pdf-read_pages` or `pdf-search` and get back useful
facts, your **VERY NEXT tool call MUST be** `scratchpad-save_section`.
Do NOT call another `pdf-read_pages` first.  Do NOT call
`pdf-search` first.  Save, then continue.

### Rule B — One topic per section.
Split findings by topic, not by page.  Use descriptive snake_case
headings: `homeready_overview`, `ltv_fico_grid`, `dti_requirements`,
`borrower_eligibility`, etc.  If a topic spans multiple reads,
**overwrite** the existing section (just call `save_section` with the
same heading) instead of creating a duplicate.

### Rule C — Note content guidelines.
- Key facts as bullet points (numeric limits, conditions, thresholds).
- 1–2 short verbatim quotes (≤200 chars) for citation.
- Include the page range you read.
- Keep each section 5–30 lines.  Summarize, do NOT paste raw text.

## Scratchpad anti-patterns (DO NOT DO THESE)
- ❌ Reading multiple pages before saving anything.
- ❌ Re-reading pages you already read because you forgot.
- ❌ Pasting raw PDF text as a section.
- ❌ Calling `scratchpad-list_sections` after `save_section` — the
  response already includes the updated table of contents.
- ❌ Calling `scratchpad-remove_section` then `save_section` — just
  overwrite.

## Answer format
- Answer in Chinese.
- Cite PDF physical pages.  If printed page labels appear, mention them
  separately and never mix them into the same page range.
"""


def _tool_hint(args: dict) -> str:
    """One-line hint for tool-call logging."""
    parts = []
    if "source" in args:
        parts.append(args["source"])
    if "query" in args:
        parts.append(f"query={args['query']!r}")
    sp = args.get("start_page")
    ep = args.get("end_page")
    if sp is not None or ep is not None:
        parts.append(f"pages={sp}-{ep}")
    if "topic" in args:
        parts.append(f"topic={args['topic']!r}")
    if "keyword" in args:
        parts.append(f"keyword={args['keyword']!r}")
    return " | ".join(parts) if parts else str(args)[:160]


async def _run_turn(
    conv: chak.Conversation,
    handler: PdfContextHandler,
    prompt: str,
    turn: int,
):
    """Execute one conversation turn and print stats."""
    start_idx = len(conv.messages)

    _console.print(f"\n[bold]Turn #{turn}[/bold]")
    _print_user(prompt)

    # Collect assistant text fragments; tool calls go to console live.
    assistant_chunks: list[str] = []

    async for event in await conv.asend(prompt, event=True):
        match event:
            case MessageChunk(content=text) if text:
                assistant_chunks.append(text)

            case ToolCallStartEvent(tool_name=name, arguments=args):
                _print_tool_call(name, _tool_hint(args))

            case ToolCallSuccessEvent(tool_name=name, result=res):
                _print_tool_result(name, res, ok=True)

            case ToolCallErrorEvent(tool_name=name, error=err):
                _print_tool_result(name, err, ok=False)

    # Render the full assistant answer in a single panel.
    if assistant_chunks:
        _print_assistant("".join(assistant_chunks))

    turn_msgs = conv.messages[start_idx:]
    _print_turn_stats(handler, conv.messages, turn_msgs, turn)


async def main(pdf_source: str = _DEFAULT_PDF):
    if not _API_KEY:
        print("DEEPSEEK_API_KEY not set.  Check .env")
        return

    # Resolve local paths to absolute so the LLM sees a stable identifier.
    # Remote URLs are passed through unchanged.
    if not pdf_source.startswith(("http://", "https://")):
        p = Path(pdf_source).expanduser().resolve()
        if not p.exists():
            print(f"PDF not found: {p}")
            return
        pdf_source = str(p)

    pdf = Pdf()
    scratchpad = Scratchpad()
    handler = PdfContextHandler()

    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        system_prompt=_SYSTEM_PROMPT,
        tools=[pdf, scratchpad],
        context_handler=handler,
    )
    # Fluent settings API — see examples/conv_setting.py
    conv.tool.loop.max(30)

    # ── Banner ──────────────────────────────────────────────────────
    info = Table(show_header=False, box=None, padding=(0, 1))
    info.add_column(style="bold", no_wrap=True)
    info.add_column()
    info.add_row("Model:", _MODEL)
    info.add_row("Window:", f"{_CONTEXT_WINDOW // 1000}K tokens")
    info.add_row("PDF:", pdf_source)
    info.add_row("Strategy:", "scratchpad notebook + mechanical offload")
    info.add_row("", "Type 'quit' to exit.")
    _console.print(Panel(
        info,
        title="[bold]Chat With PDF Agent[/bold]\nMulti-turn + Contract-driven Context Management",
        title_align="left",
        border_style="bold blue",
    ))

    # ── Turn 0: let the LLM familiarize itself with the PDF ─────────
    setup_prompt = (
        f"You are given a large PDF:\n{pdf_source}\n\n"
        "Use Pdf metadata to understand its size and structure. "
        "Save a brief structural overview to your scratchpad. "
        "Do not answer any questions yet — just get oriented."
    )
    await _run_turn(conv, handler, setup_prompt, 0)

    # ── Turns 1-4: seeded questions (gradient: heavy → detail → new topic
    #    → cross-topic comparison).  This showcases accumulating context
    #    management without manual input, then hands off to the user.
    # ─────────────────────────────────────────────────────────────────
    auto_questions = [
        # Q1 — heavy read, exercises the save-after-each-read contract.
        "Where are the HomeReady Mortgage loan eligibility and borrower "
        "eligibility requirements in the Selling Guide, and what are "
        "the key requirements?",
        # Q2 — follow-up on Q1's detail.  Tests whether the LLM consults
        # its scratchpad instead of re-reading the same PDF pages.
        "What are the maximum LTV, CLTV, and HCLTV ratios for HomeReady? "
        "How do they differ between standard and high-LTV tracks?",
        # Q3 — brand-new topic, forces a fresh search/read cycle and
        # shows context growing across subjects.
        "What are the requirements for First-Time Homebuyer programs "
        "in the Selling Guide?",
        # Q4 — cross-topic synthesis, the strongest test of multi-turn
        # value: the answer requires combining HomeReady + general DTI.
        "What is the general maximum DTI ratio allowed in the Selling "
        "Guide, and does HomeReady have any different DTI treatment?",
    ]
    for i, q in enumerate(auto_questions, 1):
        await _run_turn(conv, handler, q, i)

    # ── Interactive multi-turn loop ─────────────────────────────────
    turn = len(auto_questions) + 1
    while True:
        try:
            # Print the prompt with rich styling, then read input plainly.
            _console.print()
            _console.print("[bold cyan]You >[/bold cyan] ", end="")
            user_input = await asyncio.to_thread(input)
        except (EOFError, KeyboardInterrupt):
            break
        stripped = user_input.strip().lower()
        if stripped in ("quit", "exit", "q"):
            break
        if not user_input.strip():
            continue
        await _run_turn(conv, handler, user_input, turn)
        turn += 1

    _console.print("\n[bold]Bye![/bold]")

    # Dump the LLM's notes so you can inspect what it actually persisted.
    # Use the system temp dir to avoid polluting the project tree.
    dump_path = Path(tempfile.gettempdir()) / f"scratchpad_dump_{os.getpid()}.md"
    msg = scratchpad.dump(dump_path)
    _console.print(f"  {msg}", style="dim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat with a PDF — multi-turn with context management."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=_DEFAULT_PDF,
        help="PDF source: a remote URL or a local file path "
             "(default: the remote Selling Guide).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.pdf))
