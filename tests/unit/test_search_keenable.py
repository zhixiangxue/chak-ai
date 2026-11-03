"""Unit tests for the opt-in Keenable layer of chak.tools.std.Search."""

import asyncio
import json

import httpx

from chak.tools.std import search as search_mod
from chak.tools.std.search import Search


def _run(coro):
    return asyncio.run(coro)


def _patch_post(monkeypatch, payload, captured):
    async def _post(self, url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)


def test_keenable_disabled_by_default(monkeypatch):
    s = Search()
    called = []

    async def _fail(*a, **k):
        called.append(True)
        return None

    monkeypatch.setattr(s, "_search_keenable", _fail)

    async def _ddgs(query, n):
        return None, "stub"

    monkeypatch.setattr(s, "_search_ddgs", _ddgs)
    _run(s.search("q"))
    assert called == []


def test_keyless_public_endpoint(monkeypatch):
    captured = {}
    _patch_post(
        monkeypatch,
        {
            "results": [
                {"title": "A", "url": "https://a.example", "description": "desc", "snippet": "long"},
                {"title": "No URL", "description": "dropped"},
                {"title": "B", "url": "https://b.example", "snippet": "x" * 2000},
            ]
        },
        captured,
    )
    out = json.loads(_run(Search(use_keenable=True).search("rust", max_results=3)))
    assert captured["url"] == search_mod._KEENABLE_PUBLIC_SEARCH_URL
    assert captured["json"] == {"query": "rust", "max_results": 3}
    assert captured["headers"]["X-Keenable-Title"] == "chak"
    assert "X-API-Key" not in captured["headers"]
    assert [r["url"] for r in out] == ["https://a.example", "https://b.example"]
    assert out[0]["snippet"] == "desc"
    assert len(out[1]["snippet"]) == search_mod._KEENABLE_SNIPPET_MAX_CHARS


def test_key_uses_keyed_endpoint(monkeypatch):
    captured = {}
    _patch_post(monkeypatch, {"results": [{"title": "A", "url": "https://a.example"}]}, captured)
    _run(Search(keenable_key="kn-key").search("rust"))
    assert captured["url"] == search_mod._KEENABLE_SEARCH_URL
    assert captured["headers"]["X-API-Key"] == "kn-key"
