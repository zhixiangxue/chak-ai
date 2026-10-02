"""Unit tests for the optional You.com layer of the std Search tool."""

import json
import types

import httpx
import pytest

import chak.tools.std.search as search_module
from chak.tools.std.search import Search, SearchResult


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes: scripted httpx.AsyncClient replacement (no network access)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.you.com/mcp")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )


class _FakeAsyncClient:
    """Records every POST and returns the queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None, headers=None):
        self.requests.append({"url": url, "json": json, "headers": dict(headers or {})})
        return self._responses.pop(0)


def _install_fake_httpx(monkeypatch, responses):
    client = _FakeAsyncClient(responses)
    fake_httpx = types.SimpleNamespace(
        AsyncClient=lambda timeout=None: client,
        HTTPStatusError=httpx.HTTPStatusError,
    )
    monkeypatch.setattr(search_module, "httpx", fake_httpx)
    return client


def _sse_response(messages):
    """Build an SSE body exactly like the You.com MCP server sends."""
    blocks = []
    for m in messages:
        block = "event: message\ndata: " + json.dumps(m)
        blocks.append(block)
    body = "\n\n".join(blocks)
    return _FakeResponse(status_code=200, text=body, headers={"content-type": "text/event-stream"})


_YOUCOM_RESULT = {
    "results": {
        "web": [
            {
                "title": "Asyncio guide",
                "url": "https://example.com/asyncio",
                "description": "A practical asyncio guide",
            },
            {
                "title": "No description",
                "url": "https://example.com/nodesc",
                "description": "",
                "contents": {"highlights": ["first highlight passage"]},
            },
        ]
    },
    "metadata": {},
}


def _handshake_responses(call_response):
    """initialize (SSE) → initialized (202) → tools/call."""
    return [
        _sse_response([
            {"jsonrpc": "2.0", "id": 1, "result": {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "You.com", "version": "4.0.2"},
            }},
        ]),
        _FakeResponse(status_code=202, text="", headers={"content-type": "application/json"}),
        call_response,
    ]


def _success_call_response():
    return _sse_response([
        {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}},  # no id — skipped
        {"jsonrpc": "2.0", "id": 2, "result": {
            "content": [{"type": "text", "text": json.dumps(_YOUCOM_RESULT)}],
        }},
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_youcom_layer_skipped_by_default(monkeypatch):
    async def fake_ddgs(self, query, n):
        return None, "stub no results"

    monkeypatch.setattr(Search, "_search_ddgs", fake_ddgs)
    search = Search()  # no keys, no youcom

    out = await search.search("python asyncio")

    assert "You.com: skipped (not enabled)" in out
    assert "DuckDuckGo: stub no results" in out
    assert "Tavily: skipped (no tavily_key provided)" in out


async def test_youcom_false_behaves_like_disabled(monkeypatch):
    async def fake_ddgs(self, query, n):
        return [SearchResult("DDG result", "https://duck.example", "ddg snippet")], ""

    monkeypatch.setattr(Search, "_search_ddgs", fake_ddgs)
    search = Search(youcom=False)

    out = await search.search("query")
    parsed = json.loads(out)

    assert parsed[0]["title"] == "DDG result"


async def test_youcom_keyless_free_profile_returns_results(monkeypatch):
    client = _install_fake_httpx(monkeypatch, _handshake_responses(_success_call_response()))
    search = Search(youcom=True)

    out = await search.search("python asyncio", max_results=5)
    parsed = json.loads(out)

    # endpoint: keyless free profile, no auth header anywhere
    assert all(r["url"] == "https://api.you.com/mcp?profile=free" for r in client.requests)
    assert all("Authorization" not in r["headers"] for r in client.requests)

    # request shape: initialize → notifications/initialized → tools/call you-search
    assert client.requests[0]["json"]["method"] == "initialize"
    assert client.requests[1]["json"]["method"] == "notifications/initialized"
    call = client.requests[2]["json"]
    assert call["method"] == "tools/call"
    assert call["params"]["name"] == "you-search"
    assert call["params"]["arguments"] == {"query": "python asyncio", "count": 5}

    # results: title/url/snippet contract, description → snippet,
    # description empty → first highlight
    assert parsed[0] == {
        "title": "Asyncio guide",
        "url": "https://example.com/asyncio",
        "snippet": "A practical asyncio guide",
    }
    assert parsed[1]["snippet"] == "first highlight passage"


async def test_youcom_api_key_uses_authenticated_endpoint(monkeypatch):
    client = _install_fake_httpx(monkeypatch, _handshake_responses(_success_call_response()))
    search = Search(youcom="ydc-test-key")

    out = await search.search("query")
    parsed = json.loads(out)

    assert all(r["url"] == "https://api.you.com/mcp" for r in client.requests)
    assert all(r["headers"].get("Authorization") == "Bearer ydc-test-key" for r in client.requests)
    assert len(parsed) == 2


async def test_youcom_env_var_upgrades_to_authenticated(monkeypatch):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    monkeypatch.setenv("YDC_API_KEY", "env-key")
    client = _install_fake_httpx(monkeypatch, _handshake_responses(_success_call_response()))
    search = Search(youcom=True)

    await search.search("query")

    assert all(r["url"] == "https://api.you.com/mcp" for r in client.requests)
    assert all(r["headers"].get("Authorization") == "Bearer env-key" for r in client.requests)


async def test_youcom_session_id_passthrough(monkeypatch):
    responses = _handshake_responses(_success_call_response())
    responses[0].headers["mcp-session-id"] = "sess-123"
    client = _install_fake_httpx(monkeypatch, responses)
    search = Search(youcom=True)

    await search.search("query")

    assert client.requests[1]["headers"].get("mcp-session-id") == "sess-123"
    assert client.requests[2]["headers"].get("mcp-session-id") == "sess-123"


async def test_youcom_failure_falls_through_to_duckduckgo(monkeypatch):
    responses = [
        _FakeResponse(status_code=500, text="", headers={"content-type": "application/json"}),
    ]
    _install_fake_httpx(monkeypatch, responses)

    async def fake_ddgs(self, query, n):
        return [SearchResult("DDG result", "https://duck.example", "ddg snippet")], ""

    monkeypatch.setattr(Search, "_search_ddgs", fake_ddgs)
    search = Search(youcom=True)

    out = await search.search("query")
    parsed = json.loads(out)

    # You.com layer failed → DuckDuckGo still answers
    assert parsed == [{"title": "DDG result", "url": "https://duck.example", "snippet": "ddg snippet"}]


async def test_youcom_plain_json_response(monkeypatch):
    call = _FakeResponse(
        status_code=200,
        text=json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"content": [{"type": "text", "text": json.dumps(_YOUCOM_RESULT)}]},
        }),
        headers={"content-type": "application/json"},
    )
    _install_fake_httpx(monkeypatch, _handshake_responses(call))
    search = Search(youcom=True)

    out = await search.search("query")
    parsed = json.loads(out)

    assert parsed[0]["title"] == "Asyncio guide"


async def test_youcom_error_result_reports_reason(monkeypatch):
    call = _sse_response([
        {"jsonrpc": "2.0", "id": 2, "error": {"code": -32000, "message": "invalid query"}},
    ])
    _install_fake_httpx(monkeypatch, _handshake_responses(call))
    search = Search(youcom=True)

    results, err = await search._search_youcom("query", 5)

    assert results is None
    assert err == "invalid query"


async def test_empty_query_unchanged(monkeypatch):
    _install_fake_httpx(monkeypatch, [])
    search = Search(youcom=True)

    out = await search.search("   ")

    assert out == "Error: query must not be empty"
    assert repr(search) == "<Search tavily=no brave=no youcom=yes max_results=10>"
