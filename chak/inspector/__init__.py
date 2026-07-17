"""
Chak Inspector — live view of Conversation messages in the browser.

Zero-intrusion design: this module does not modify Conversation or any chak
internals. It polls ``conv.messages`` on background threads and pushes updates
to a local browser page via Server-Sent Events. When you never call
``watch()``, this module has no effect on runtime behavior at all.

Two modes (one function)
------------------------
``watch()`` behaves differently depending on whether you pass a conversation:

1. ``watch(conv)`` — attach a single conversation manually.

2. ``watch()`` (no args) — turn on global auto-observation: every
   Conversation created afterwards in this process is attached
   automatically. No need to scatter ``watch(conv)`` through agent code.

Both modes share a single HTTP+SSE server. The viewer page shows a tab bar
(only visible when 2+ convs are attached) so you can switch between them.
Each conversation is polled on its own background thread so a slow/blocked
one never starves the others.

Usage (per-conversation):
    import chak
    from chak.inspector import watch

    conv1 = chak.Conversation(...)
    watch(conv1)                 # starts server, opens browser, watches conv1

    conv2 = chak.Conversation(...)
    watch(conv2)                 # attaches to same server, appears as a tab

Usage (global auto-attach):
    import chak
    from chak.inspector import watch

    watch()                      # starts server, auto-watches all future convs

    # Every Conversation created below is auto-attached — no watch(conv) needed.
    conv1 = chak.Conversation(...)
    conv2 = chak.Conversation(...)

Mixed usage — both modes in the same process:
    watch(conv1)                 # manually watch an existing conv
    watch()                      # flip on global auto-attach for future convs
    conv2 = chak.Conversation(...)  # auto-attached, appears as a new tab
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["watch"]


# ─── Module-level singleton state ────────────────────────────────────────
# One inspector server per process. ``_REG`` holds every conversation that
# has been ``watch()``-ed so far, keyed by ``conv.id``. All mutations go
# through ``_LOCK``. Threads (pollers) and the FastAPI app read via
# ``_REG.snapshot_*()`` helpers which do their own copying.

_LOCK = threading.Lock()


class _ConvSlot:
    """Per-conversation state: the live conv object, its last serialized
    snapshot, and a version counter used by SSE clients to detect changes."""

    __slots__ = (
        "conv", "messages", "last_sig", "version", "attached_at", "is_running",
        "tools", "tools_sig",
    )

    def __init__(self, conv: Any) -> None:
        self.conv = conv
        self.messages: List[dict] = []
        self.last_sig: Optional[tuple] = None
        self.version: int = 0
        self.attached_at: float = time.time()
        # "Running" = a turn is currently in flight. Flipped True/False
        # by hooks we install on ``conv.hook.before_send`` /
        # ``conv.hook.after_send`` (see ``_install_running_hooks``).
        # This is an authoritative signal from chak's own turn lifecycle,
        # not a heuristic — no message-tail guessing required.
        self.is_running: bool = False
        # Snapshot of the OpenAI-shape tool schema that chak is currently
        # willing to send to the LLM (from ``conv._tool_manager._get_openai_tools()``).
        # Refreshed by the poller alongside ``messages`` so the viewer can
        # show the exact contract handed to the model — this is what the
        # user needs to see when debugging "did chak tell the model about
        # parameter types?" style questions.
        self.tools: List[dict] = []
        self.tools_sig: Optional[tuple] = None


class _Registry:
    """Central registry of attached conversations plus server metadata."""

    def __init__(self) -> None:
        self.server_id: str = uuid.uuid4().hex[:12]
        self.started: bool = False
        self.host: str = "127.0.0.1"
        self.port: int = 7878
        self.poll_interval: float = 0.2
        self.slots: Dict[str, _ConvSlot] = {}
        # Monotonic counter for the *set* of attached conv_ids. Bumped on
        # every attach; consumed by the SSE clients to know when to refetch
        # the /convs listing (so a newly-attached conv shows up in the tab
        # bar without a manual refresh).
        self.roster_version: int = 0

    def snapshot_roster(self) -> List[dict]:
        """Public roster snapshot used by the ``/convs`` REST endpoint.

        Callers that already hold ``_LOCK`` (e.g. the SSE generator, which
        acquires the lock to snapshot the slot state atomically) must use
        ``_snapshot_roster_locked`` instead — ``threading.Lock`` is not
        reentrant and re-entering here deadlocks the event loop task,
        which is exactly the failure mode that manifests as the browser
        staring at "loading…" forever waiting on the first SSE frame.
        """
        with _LOCK:
            return self._snapshot_roster_locked()

    def _snapshot_roster_locked(self) -> List[dict]:
        """Roster snapshot assuming the caller already holds ``_LOCK``."""
        out: List[dict] = []
        for cid, slot in self.slots.items():
            model_uri = _infer_model_uri(slot)
            # ``conv.title`` is user-set (via ``conv.title = ...``) so the
            # roster mirrors whatever is currently on the Conversation.
            # ``getattr`` with a default keeps this backward-compatible
            # for stubs / mocks that don't expose the attribute.
            title = getattr(slot.conv, "title", None)
            out.append({
                "id": cid,
                "title": title,
                "model_uri": model_uri,
                "msg_count": len(slot.messages),
                "attached_at": slot.attached_at,
                "running": slot.is_running,
            })
        # Preserve insertion order (Python dicts are ordered).
        return out


_REG = _Registry()

# Guard flag so on() is idempotent: calling it multiple times only
# subscribes the auto-watch listener once.
_AUTO_ON: bool = False


# ─── Helpers ─────────────────────────────────────────────────────────────

def _port_in_use(host: str, port: int) -> bool:
    """Return True if ``(host, port)`` is already bound by another process.

    Used only when we're about to start the server for the first time in
    THIS process; subsequent ``watch()`` calls skip this check because they
    reuse the running server.

    ``SO_REUSEADDR`` is set so a TIME_WAIT socket from a just-killed prior
    run doesn't produce a false positive here.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError:
            pass
        s.settimeout(0.2)
        try:
            s.bind((host, port))
        except OSError:
            return True
    return False


def _serialize(msg: Any) -> dict:
    """Best-effort JSON-safe view of a chak Message."""
    try:
        return msg.model_dump(mode="json")
    except Exception:
        return {
            "id": getattr(msg, "id", None),
            "role": getattr(msg, "role", "?"),
            "content": getattr(msg, "content", None),
        }


def _snapshot_signature(messages: List[Any]) -> tuple:
    """Cheap fingerprint used to detect list changes without full serialization."""
    sig = []
    for m in messages:
        mid = getattr(m, "id", None)
        content = getattr(m, "content", None)
        clen = len(content) if isinstance(content, (str, list)) else 0
        sig.append((mid, getattr(m, "role", None), clen))
    return tuple(sig)


def _snapshot_wire_tools(conv: Any) -> List[dict]:
    """Snapshot the tool schema *as the model actually sees it* for one conv.

    Design rationale — why this is here and not a public method on Provider
    ---------------------------------------------------------------------
    Chak standardizes internally on OpenAI's tool shape:

        {"type": "function", "function": {"name", "description", "parameters"}}

    All providers accept this as chak's canonical form. Almost every provider
    (OpenAI, DeepSeek, Moonshot, Google, Baidu, Bailian, 阿里, Ollama, Azure,
    Mistral, Minimax, Iflytek, ...) forwards it on the wire verbatim because
    their SDKs speak OpenAI-compatible protocol.

    The one exception is Anthropic: it rewires tools into its own shape
    (``{name, description, input_schema}``) inside ``send()`` via a private
    helper ``converter._convert_tools``.

    The inspector cares about "what did the model actually see", so for
    Anthropic conversations we want the ``input_schema`` shape, not the
    OpenAI ``parameters`` shape (otherwise the viewer would show a form
    the model never received, and users would file spurious bugs like
    "why is chak sending OpenAI format to Anthropic??").

    We considered adding a public ``Provider.to_wire_tools(...)`` method
    and overriding it in Anthropic + delegating in ResilientProvider.
    Rejected: inspector is an *observation* tool; letting it dictate the
    shape of the core Provider API inverts the dependency direction. Every
    future inspector feature would carve another notch into core, and the
    surface area drifts.

    Instead we duck-type here: if the provider's converter happens to expose
    ``_convert_tools`` (currently only Anthropic), we call it. Otherwise
    we return the canonical OpenAI shape, which for every non-Anthropic
    provider IS the wire shape. The private-name coupling is confined to
    this single helper; a rename in ``_convert_tools`` would fail the
    inspector's unit test loudly, and Anthropic's shape wouldn't render —
    both are visible failure modes, not silent drift.

    Failure policy: any exception is swallowed and we return ``[]``.
    Inspector is strictly non-intrusive — it must never break the app.
    """
    tm = getattr(conv, "_tool_manager", None)
    if tm is None:
        return []
    try:
        openai_tools = tm._get_openai_tools()
    except Exception:
        return []
    if not openai_tools:
        return []
    # Duck-type check for provider-specific rewiring. Only Anthropic-family
    # providers currently implement this; everyone else eats OpenAI shape.
    # NOTE: ``_convert_tools`` lives on the *provider* (not the converter) —
    # see ``AnthropicCompatibleProvider._convert_tools`` in anthropic_compat.py.
    # Getting this attachment point wrong silently falls back to identity,
    # which is exactly the bug we're guarding against.
    provider = getattr(conv, "provider", None)
    convert_fn = getattr(provider, "_convert_tools", None)
    if callable(convert_fn):
        try:
            return convert_fn(openai_tools)
        except Exception:
            # Conversion blew up — show canonical form rather than nothing,
            # so the user still gets *something* useful. A silent [] here
            # would look like "no tools", which is a misleading signal.
            return openai_tools
    return openai_tools


def _wire_tools_signature(tools: List[dict]) -> tuple:
    """Cheap fingerprint for tool list change detection.

    Sensitive to name, description, and top-level schema keys. Rebuilds
    the whole tool snapshot when any of those change (skill staging,
    dynamic add/remove, etc.).
    """
    sig = []
    for t in tools or []:
        # Handle both OpenAI shape (nested under ``function``) and
        # Anthropic shape (flat name/description/input_schema).
        if isinstance(t, dict):
            fn = t.get("function") or t
            name = fn.get("name") if isinstance(fn, dict) else None
            desc = fn.get("description") if isinstance(fn, dict) else None
            # A schema hash would be ideal but stringifying keys is cheap
            # enough and avoids importing hashlib for a fingerprint.
            schema = fn.get("parameters") or t.get("input_schema") or {}
            props = tuple(sorted((schema.get("properties") or {}).keys()))
            sig.append((name, desc, props))
    return tuple(sig)


def _install_running_hooks(slot: _ConvSlot) -> None:
    """Track turn boundaries via chak's ``before_send`` / ``after_send`` hooks.

    This gives an authoritative "a turn is in flight" signal straight
    from ``Conversation``'s own lifecycle, replacing the older approach
    of guessing from the message-tail shape. ``after_send`` is invoked
    by chak inside ``finally`` blocks on the streaming/event paths and
    directly after the model call on non-stream paths, so both success
    and normal error paths reliably clear the flag.

    Caveat: hooks fire in registration order. Registering here at
    ``watch()`` time means ours run before any hook the user adds
    afterwards — so a later user hook raising cannot prevent our
    marker from running. Users who add hooks *before* ``watch()`` and
    whose hooks raise may leave ``is_running`` stuck True; that is an
    acceptable trade for the massive simplification vs the heuristic.
    """
    async def _mark_running(conv, request, **kwargs):
        with _LOCK:
            slot.is_running = True

    async def _mark_idle(conv, request, **kwargs):
        with _LOCK:
            slot.is_running = False

    slot.conv.hook.before_send(_mark_running)
    slot.conv.hook.after_send(_mark_idle)


def _infer_model_uri(slot: _ConvSlot) -> Optional[str]:
    """Best-effort ``provider/model`` label for a conv.

    First try ``conv.model_uri`` (chak's own attribute), then fall back to
    the latest assistant message's metadata so we still show something
    meaningful even if the conversation object doesn't expose it directly.
    """
    conv = slot.conv
    uri = getattr(conv, "model_uri", None)
    if uri:
        return uri
    # Fall back: scan the newest assistant message's metadata.
    for m in reversed(slot.messages):
        if m.get("role") != "assistant":
            continue
        md = m.get("metadata") or {}
        p, mo = md.get("provider"), md.get("model")
        if p and mo:
            return f"{p}/{mo}"
        return mo or p
    return None


def _make_poller(conv_id: str, slot: _ConvSlot) -> threading.Thread:
    """Spawn a daemon poller for one conversation slot.

    Kept as a per-conv thread instead of a shared loop so a hung/slow
    ``conv.messages`` access never starves siblings. This loop is now
    purely about detecting message-list changes; the running flag is
    driven by hooks (see ``_install_running_hooks``).
    """
    poll_interval = _REG.poll_interval

    def _poll() -> None:
        while True:
            try:
                current = list(slot.conv.messages)
                sig = _snapshot_signature(current)
                # Snapshot tools too — they can change between rounds when
                # skills stage in/out or when the app dynamically adds/removes
                # tools mid-conversation. Cheap: a small list of dicts, and
                # the signature check bails out early when nothing moved.
                wire_tools = _snapshot_wire_tools(slot.conv)
                tools_sig = _wire_tools_signature(wire_tools)
                msgs_changed = sig != slot.last_sig
                tools_changed = tools_sig != slot.tools_sig
                if msgs_changed or tools_changed:
                    with _LOCK:
                        if msgs_changed:
                            slot.messages = [_serialize(m) for m in current]
                            slot.last_sig = sig
                        if tools_changed:
                            slot.tools = wire_tools
                            slot.tools_sig = tools_sig
                        # One version counter for both — SSE clients don't
                        # need to distinguish "which piece changed", they
                        # get a fresh payload either way.
                        slot.version += 1
            except Exception:
                # Never let poller crash — inspector must not affect the app.
                pass
            time.sleep(poll_interval)

    t = threading.Thread(
        target=_poll, daemon=True, name=f"chak-inspector-poll-{conv_id[:8]}"
    )
    t.start()
    return t


# ─── Server lifecycle ────────────────────────────────────────────────────

def _attach_conv(conv: Any, conv_id: str) -> None:
    """Register a Conversation with the inspector: create slot, install
    hooks, and spawn the poller thread.

    Shared by ``watch()`` (manual) and the ``on_create`` signal listener
    (auto-attach via ``on()``).
    """
    slot = _ConvSlot(conv)
    with _LOCK:
        _REG.slots[conv_id] = slot
        _REG.roster_version += 1
    _install_running_hooks(slot)
    _make_poller(conv_id, slot)


def _ensure_server(
    port: int,
    poll_interval: float,
    host: str,
    open_browser: bool,
) -> bool:
    """Start the inspector HTTP+SSE server if not already running.

    Returns ``True`` if this call actually brought up the server, ``False``
    if it was already running or could not bind the port.

    Extracted from ``watch()`` so that ``on()`` can start the server
    immediately (before any Conversation exists) and subsequent
    ``watch()`` / auto-attach calls reuse it.
    """
    with _LOCK:
        if _REG.started:
            # Server already running — warn if the caller asked for a
            # different port; the existing server wins (one server per
            # process by design).
            if port != _REG.port:
                print(
                    f"⚠️  Chak Inspector: server already running on port "
                    f"{_REG.port}; requested port {port} is ignored."
                )
            return False

    # Lazy import — only needed when the server actually starts.
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "chak.inspector requires FastAPI + uvicorn. "
            "Install with: pip install 'chakpy[server]'"
        ) from e

    _bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    if _port_in_use(_bind_host, port):
        url = f"http://{_bind_host}:{port}"
        print(
            f"⚠️  Chak Inspector: port {port} is already in use.\n"
            f"   Another inspector seems to be running at {url}. This\n"
            f"   run will not attach a live poller. Kill the previous\n"
            f"   process or pass port=<other> to force a fresh server."
        )
        return False

    with _LOCK:
        _REG.host = host
        _REG.port = port
        _REG.poll_interval = poll_interval

    app = _build_app(FastAPI, HTMLResponse, JSONResponse, StreamingResponse, HTTPException)

    def _run_server() -> None:
        # log_level="warning" keeps chak's own logs uncluttered.
        uvicorn.run(app, host=host, port=port, log_level="warning")

    threading.Thread(
        target=_run_server, daemon=True, name="chak-inspector-http"
    ).start()
    with _LOCK:
        _REG.started = True

    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{_REG.port}"
    print(f"🔍 Chak Inspector: {url}")
    if open_browser:
        # Small delay so uvicorn has a moment to bind before the browser
        # opens. On headless servers (CI, Linux without a desktop env)
        # webbrowser.open() returns False silently — we don't want that
        # to be noisy, so the return value is simply ignored.
        def _try_open():
            try:
                webbrowser.open(url)
            except Exception:
                pass  # No browser available — that's fine, stay silent.
        threading.Timer(0.5, _try_open).start()
    return True


# ─── Public API ──────────────────────────────────────────────────────────

def watch(
    conv: Any = None,
    port: int = 7878,
    poll_interval: float = 0.2,
    open_browser: bool = True,
    host: str = "127.0.0.1",
) -> None:
    """Observe conversations in the browser — one or all.

    Two modes, controlled by whether *conv* is passed:

    - ``watch(conv)`` — attach a single Conversation to the inspector.
    - ``watch()``     — turn on **global auto-attach**: every Conversation
      created afterwards in this process is attached automatically.
      Ideal for agent apps where convs are created dynamically deep
      inside sub-agents or parallel tasks.

    Both modes share a single HTTP+SSE server. Each new conversation
    appears as a tab in the browser UI. The function returns immediately;
    the server runs in a daemon thread and dies with the interpreter.

    Args:
        conv: The Conversation to observe (manual mode), or ``None`` to
              turn on global auto-attach for all future Conversations.
              Only ``conv.messages`` / ``conv.id`` / ``conv.dump()`` are
              read — no attributes are mutated.
        port: HTTP port for the inspector page. Default 7878. Only honored
              on the first call in the process; subsequent calls reuse
              the running server.
        poll_interval: Seconds between polls of ``conv.messages``. Only
              honored on the first call (applies to all convs uniformly).
        open_browser: If True, auto-open the inspector URL on the first
              call. Ignored on subsequent calls (browser is already open).
              On headless servers (no browser), the call is silently
              ignored — safe to leave as-is in production.
        host: Bind address. Default 127.0.0.1 (local only).

    Examples::

        # Watch one conversation
        watch(conv)

        # Watch all future conversations (global auto-attach)
        watch()

        # Mixed: manual + auto
        watch(conv1)             # manually watch conv1
        watch()                  # auto-attach all future convs
        conv2 = Conversation(...)  # auto-attached as a new tab
    """
    # ── Global auto-attach mode ─────────────────────────────────────
    if conv is None:
        global _AUTO_ON
        if _AUTO_ON:
            return  # Already active — nothing to do.

        # Start the server right away so the browser page is immediately
        # available, even before the first Conversation is created.
        _ensure_server(port, poll_interval, host, open_browser)

        def _auto_watch(new_conv: Any) -> None:
            """Signal listener: attach every new Conversation."""
            watch(
                new_conv,
                port=port,
                poll_interval=poll_interval,
                # Browser was already opened by the initial watch() call;
                # don't re-open for every new conversation.
                open_browser=False,
                host=host,
            )

        # Late import — merely importing chak.inspector must never trigger
        # chak.conversation; the import only happens when the user
        # explicitly calls watch().
        from chak.conversation import Conversation
        Conversation.on_create.subscribe(_auto_watch)
        _AUTO_ON = True
        return

    # ── Per-conversation mode ───────────────────────────────────────
    conv_id = getattr(conv, "id", None) or uuid.uuid4().hex
    # Force-string in case some caller uses non-str ids.
    conv_id = str(conv_id)

    with _LOCK:
        already_attached = conv_id in _REG.slots

    if already_attached:
        # Idempotent: don't double-register the same conv.
        return

    started_now = _ensure_server(port, poll_interval, host, open_browser)

    _attach_conv(conv, conv_id)

    if not started_now:
        url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{_REG.port}"
        print(f"🔍 Chak Inspector: attached conv {conv_id[:8]} → {url}")


# ─── FastAPI app builder ─────────────────────────────────────────────────

def _build_app(FastAPI, HTMLResponse, JSONResponse, StreamingResponse, HTTPException):
    """Construct the FastAPI app. Split out so the imports stay lazy."""
    app = FastAPI(title="Chak Inspector")
    html_path = Path(__file__).parent / "viewer.html"
    html_content = html_path.read_text(encoding="utf-8")
    # Serve markdown-it.min.js from the same origin so the viewer has zero
    # external dependencies. We switched from marked.js because markdown-it
    # handles edge cases (tables with inline code, nested lists, HTML mixed
    # with markdown) more reliably.
    md_lib_path = Path(__file__).parent / "markdown-it.min.js"
    md_lib_content = md_lib_path.read_text(encoding="utf-8") if md_lib_path.exists() else ""

    poll_interval = _REG.poll_interval

    @app.get("/")
    async def _index() -> HTMLResponse:
        return HTMLResponse(html_content)

    @app.get("/markdown-it.min.js")
    async def _md_lib() -> HTMLResponse:
        return HTMLResponse(md_lib_content, media_type="text/javascript")

    @app.get("/convs")
    async def _convs() -> JSONResponse:
        """List every attached conversation. The browser polls this on
        first load and whenever the SSE stream reports a roster_version
        change."""
        with _LOCK:
            rv = _REG.roster_version
            sid = _REG.server_id
        return JSONResponse({
            "server_id": sid,
            "roster_version": rv,
            "convs": _REG.snapshot_roster(),
        })

    @app.get("/dump/{conv_id}")
    async def _dump(conv_id: str) -> JSONResponse:
        """Return conv.dump() plus conv.id so the browser can save a file
        named after the conversation."""
        with _LOCK:
            slot = _REG.slots.get(conv_id)
        if slot is None:
            raise HTTPException(status_code=404, detail=f"conv {conv_id} not found")
        try:
            data = slot.conv.dump()
            actual_id = getattr(slot.conv, "id", None) or conv_id
        except Exception as e:  # pragma: no cover - defensive
            return JSONResponse(
                {"error": f"dump failed: {e}"}, status_code=500
            )
        payload = {"conversation_id": actual_id, "messages": data}
        headers = {
            "Content-Disposition": f'attachment; filename="{actual_id}.json"'
        }
        return JSONResponse(payload, headers=headers)

    @app.get("/stream/{conv_id}")
    async def _stream(conv_id: str) -> StreamingResponse:
        """SSE stream for one conversation. Payload carries server_id and
        roster_version so the browser can detect (a) a process restart
        (auto-reload the tab) and (b) newly-attached siblings (refresh
        the tab bar)."""
        async def _gen():
            last_version = -1
            last_roster_sig: Optional[tuple] = None
            # Track tools separately from ``version`` so a payload that
            # only carries a roster change still doesn't force clients to
            # re-render tools they already have. The very first tick still
            # pushes tools regardless, courtesy of ``first_tick``.
            last_tools_sig: Optional[tuple] = None
            # ``first_tick`` guarantees the very first frame on a new SSE
            # connection carries a full ``messages`` array, even if the
            # slot's version happens to match a stale ``last_version=-1``
            # comparison assumption. Without this, switching to a conv
            # whose messages haven't changed since server boot could
            # leave the UI stuck on "loading…" until the next real edit.
            first_tick = True
            while True:
                with _LOCK:
                    slot = _REG.slots.get(conv_id)
                    sid = _REG.server_id
                    rv = _REG.roster_version
                    if slot is None:
                        payload = {"error": "conv_not_found", "server_id": sid}
                        yield f"data: {json.dumps(payload)}\n\n"
                        return
                    v = slot.version
                    # Snapshot every conv's visible state (msg_count +
                    # running flag). We push the whole roster whenever
                    # any bit of it changes, so the sidebar's spinners
                    # stay in sync with actual per-conv activity — not
                    # just at attach/detach time.
                    #
                    # NOTE: we're already holding ``_LOCK`` here — must use
                    # the ``_locked`` variant, not the public
                    # ``snapshot_roster()`` (which would re-acquire and
                    # deadlock, since ``threading.Lock`` isn't reentrant).
                    roster_snapshot = _REG._snapshot_roster_locked()
                    roster_sig = tuple(
                        (r["id"], r["title"], r["model_uri"], r["msg_count"], r["running"])
                        for r in roster_snapshot
                    )
                    changed_msgs = v != last_version
                    changed_roster = roster_sig != last_roster_sig
                    # Compare against the slot's authoritative tools_sig,
                    # not a recomputation — the poller is the single source
                    # of truth for "tools have really changed".
                    changed_tools = slot.tools_sig != last_tools_sig
                    should_push = first_tick or changed_msgs or changed_roster or changed_tools
                    if should_push:
                        # Send messages / tools on first tick OR whenever
                        # they actually changed; keep other slots as
                        # ``None`` so the client can distinguish "nothing
                        # new to render" from "empty".
                        include_msgs = first_tick or changed_msgs
                        include_tools = first_tick or changed_tools
                        payload = {
                            "server_id": sid,
                            "roster_version": rv,
                            "roster": roster_snapshot,
                            "version": v,
                            "messages": list(slot.messages) if include_msgs else None,
                            # Tools snapshot as the model actually sees them
                            # (Anthropic input_schema shape vs everyone else's
                            # OpenAI shape — see _snapshot_wire_tools).
                            "tools": list(slot.tools) if include_tools else None,
                        }
                        last_version = v
                        last_roster_sig = roster_sig
                        last_tools_sig = slot.tools_sig
                        first_tick = False
                    else:
                        payload = None
                if payload is not None:
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                else:
                    # SSE keep-alive comment; some proxies/browsers drop idle streams.
                    yield ": keep-alive\n\n"
                await asyncio.sleep(poll_interval)

        return StreamingResponse(_gen(), media_type="text/event-stream")

    return app
