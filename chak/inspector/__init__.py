"""
Chak Inspector — live view of Conversation messages in the browser.

Zero-intrusion design: this module does not modify Conversation or any chak
internals. It polls ``conv.messages`` on background threads and pushes updates
to a local browser page via Server-Sent Events. When you never call
``watch(conv)``, this module has no effect on runtime behavior at all.

Multi-conversation support
--------------------------
``watch(conv)`` can be called any number of times in the same process. The
first call starts a single HTTP+SSE server and opens the browser; every
subsequent call simply registers the new conversation onto the running
server. The viewer page shows a tab bar (only visible when 2+ convs are
attached) so you can switch between them. Each conversation is polled on
its own background thread so a slow/blocked one never starves the others.

Usage:
    import chak
    from chak.inspector import watch

    conv1 = chak.Conversation(...)
    watch(conv1)                 # starts server, opens browser

    conv2 = chak.Conversation(...)
    watch(conv2)                 # attaches to same server, appears as a tab
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

    __slots__ = ("conv", "messages", "last_sig", "version", "attached_at", "is_running")

    def __init__(self, conv: Any) -> None:
        self.conv = conv
        self.messages: List[dict] = []
        self.last_sig: Optional[tuple] = None
        self.version: int = 0
        self.attached_at: float = time.time()
        # "Running" = mid-turn. Recomputed on every poll from the current
        # message tail; see ``_infer_running``. Surfaced to the browser so
        # the sidebar can render a spinner next to active convs.
        self.is_running: bool = False


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

    def snapshot_ids(self) -> List[str]:
        # Snapshot under lock; keys() view is unsafe to iterate lazily.
        with _LOCK:
            return list(self.slots.keys())

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
            out.append({
                "id": cid,
                "model_uri": model_uri,
                "msg_count": len(slot.messages),
                "attached_at": slot.attached_at,
                "running": slot.is_running,
            })
        # Preserve insertion order (Python dicts are ordered).
        return out


_REG = _Registry()


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


def _infer_running(messages: List[dict]) -> bool:
    """Heuristic: is the conversation currently mid-turn?

    A chak turn is "complete" when the tail message is an ``assistant``
    reply with no pending tool_calls. Anything else — an unanswered
    user prompt, an assistant message whose tool_calls are still waiting
    for results, or a tool result awaiting the next assistant hop —
    means the agent loop is expected to append at least one more
    message shortly.

    This is a *conservative* signal derived purely from the message
    tail; it doesn't require any hook into chak's internals. If the
    process is genuinely idle but happens to be sitting on a
    tool_call-terminated turn (e.g. user hit Ctrl+C), we'll still
    report "running" — that's fine, the sidebar's spinner just means
    "the model owes us another message here".
    """
    if not messages:
        return False
    last = messages[-1]
    role = last.get("role")
    if role == "assistant":
        # A final assistant reply (no tool_calls) terminates the turn.
        return bool(last.get("tool_calls"))
    if role in ("user", "tool"):
        # The next message is owed by the assistant — still running.
        return True
    if role == "system":
        # A conv containing only a system prompt is idle by definition.
        return False
    # Unknown role — err on the side of "running" so we don't drop the
    # spinner mid-flight for a message shape we don't recognize.
    return True


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


def _make_poller(slot: _ConvSlot) -> threading.Thread:
    """Spawn a daemon poller for one conversation slot.

    Kept as a per-conv thread instead of a shared loop so a hung/slow
    ``conv.messages`` access never starves siblings.
    """
    poll_interval = _REG.poll_interval

    def _poll() -> None:
        while True:
            try:
                current = list(slot.conv.messages)
                sig = _snapshot_signature(current)
                if sig != slot.last_sig:
                    dumped = [_serialize(m) for m in current]
                    running = _infer_running(dumped)
                    with _LOCK:
                        slot.messages = dumped
                        slot.last_sig = sig
                        slot.version += 1
                        slot.is_running = running
            except Exception:
                # Never let poller crash — inspector must not affect the app.
                pass
            time.sleep(poll_interval)

    cid_short = next(
        (cid for cid, s in _REG.slots.items() if s is slot), "?"
    )[:8]
    t = threading.Thread(
        target=_poll, daemon=True, name=f"chak-inspector-poll-{cid_short}"
    )
    t.start()
    return t


# ─── Public API ──────────────────────────────────────────────────────────

def watch(
    conv: Any,
    port: int = 7878,
    poll_interval: float = 0.2,
    open_browser: bool = True,
    host: str = "127.0.0.1",
) -> None:
    """Attach a Conversation to the (singleton) inspector server.

    Args:
        conv: The Conversation instance to observe. Only ``conv.messages``
              (and ``conv.id`` / ``conv.dump()``) are read — no attributes
              are mutated.
        port: HTTP port for the inspector page. Default 7878. Only honored
              on the first call in the process; subsequent calls reuse
              the running server.
        poll_interval: Seconds between polls of ``conv.messages``. Only
              honored on the first call (applies to all convs uniformly).
        open_browser: If True, auto-open the inspector URL on the first
              call. Ignored on subsequent calls (browser is already open).
        host: Bind address. Default 127.0.0.1 (local only).

    Multiple ``watch()`` calls in the same process share one server; each
    additional conversation appears as a tab in the browser UI.

    This function returns immediately; the server runs in a daemon thread
    and dies with the interpreter.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "chak.inspector requires FastAPI + uvicorn. "
            "Install with: pip install 'chakpy[server]'"
        ) from e

    conv_id = getattr(conv, "id", None) or uuid.uuid4().hex
    # Force-string in case some caller uses non-str ids.
    conv_id = str(conv_id)

    with _LOCK:
        already_attached = conv_id in _REG.slots
        first_call = not _REG.started

    if already_attached:
        # Idempotent: don't double-register the same conv.
        return

    # ── First call in this process: bring up the server ─────────────
    if first_call:
        _bind_host = "127.0.0.1" if host == "0.0.0.0" else host
        if _port_in_use(_bind_host, port):
            url = f"http://{_bind_host}:{port}"
            print(
                f"⚠️  Chak Inspector: port {port} is already in use.\n"
                f"   Another inspector seems to be running at {url}. This\n"
                f"   run will not attach a live poller. Kill the previous\n"
                f"   process or pass port=<other> to force a fresh server."
            )
            return

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

    # ── Register the conv (both first-call and subsequent calls) ────
    slot = _ConvSlot(conv)
    with _LOCK:
        _REG.slots[conv_id] = slot
        _REG.roster_version += 1

    _make_poller(slot)

    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{_REG.port}"
    if first_call:
        print(f"🔍 Chak Inspector: {url}")
        if open_browser:
            # Small delay so uvicorn has a moment to bind before the browser opens.
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    else:
        print(f"🔍 Chak Inspector: attached conv {conv_id[:8]} → {url}")


# ─── FastAPI app builder ─────────────────────────────────────────────────

def _build_app(FastAPI, HTMLResponse, JSONResponse, StreamingResponse, HTTPException):
    """Construct the FastAPI app. Split out so the imports stay lazy."""
    app = FastAPI(title="Chak Inspector")
    html_path = Path(__file__).parent / "viewer.html"
    html_content = html_path.read_text(encoding="utf-8")

    poll_interval = _REG.poll_interval

    @app.get("/")
    async def _index() -> HTMLResponse:
        return HTMLResponse(html_content)

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
                        (r["id"], r["msg_count"], r["running"])
                        for r in roster_snapshot
                    )
                    changed_msgs = v != last_version
                    changed_roster = roster_sig != last_roster_sig
                    should_push = first_tick or changed_msgs or changed_roster
                    if should_push:
                        # Send messages on first tick OR whenever they
                        # actually changed; roster-only ticks keep it as
                        # ``None`` so the client can distinguish "nothing
                        # new to render" from "empty conversation".
                        include_msgs = first_tick or changed_msgs
                        payload = {
                            "server_id": sid,
                            "roster_version": rv,
                            "roster": roster_snapshot,
                            "version": v,
                            "messages": list(slot.messages) if include_msgs else None,
                        }
                        last_version = v
                        last_roster_sig = roster_sig
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
