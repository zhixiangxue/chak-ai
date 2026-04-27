"""
Search: Built-in web search tool for chak

Provides one public method exposed as an LLM tool via NativeObjectTool:

    - search: keyword/query → list of results (title, url, snippet)

Search chain (first success wins):
    1. Tavily      — AI-optimized, clean results, requires tavily_key
    2. Brave       — high-quality traditional search, requires brave_key
    3. DuckDuckGo  — free, no key, via duckduckgo-search (ddgs)

Usage:
    from chak.tools.std import Search
    search = Search(tavily_key="tvly-xxx")
    conv = Conversation(model, tools=[search])

Optional dependencies (pip install chakpy[web]):
    tavily-python      — Tavily AI search client (Layer 1, needs API key)
    ddgs               — DuckDuckGo backend (Layer 3, no key, pip install ddgs)
    httpx              — already a core chak dependency (used for Brave)
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional

import httpx

_USER_AGENT = "Mozilla/5.0 (compatible; chak-ai/1.0)"
_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class SearchResult:
    """A single search result."""

    __slots__ = ("title", "url", "snippet")

    def __init__(self, title: str, url: str, snippet: str):
        self.title = title
        self.url = url
        self.snippet = snippet

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SearchResult title={self.title!r} url={self.url!r}>"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class Search:
    """Web search tool.

    Exposes one LLM-callable method via NativeObjectTool:
        - search: free-text query → top-N results (title, url, snippet)

    Search chain (first success wins):
        1. Tavily      — AI-optimized results (requires tavily_key)
        2. Brave       — high-quality traditional search (requires brave_key)
        3. DuckDuckGo  — free, no API key needed

    Example::

        search = Search(tavily_key="tvly-xxx")
        conv = Conversation(model, tools=[search])
    """

    def __init__(
        self,
        tavily_key: Optional[str] = None,
        brave_key: Optional[str] = None,
        max_results: int = 10,
        timeout: int = 15,
        region: str = "wt-wt",
        safe_search: str = "moderate",
    ):
        """
        Args:
            tavily_key:   Tavily API key (optional; Layer 1 skipped when absent).
            brave_key:    Brave Search API key (optional; Layer 2 skipped when absent).
            max_results:  Maximum number of results to return (default 10).
            timeout:      HTTP request timeout in seconds (default 15).
            region:       Search region for DuckDuckGo, e.g. "wt-wt", "us-en", "cn-zh".
            safe_search:  "strict" | "moderate" | "off" (default "moderate").
        """
        self._tavily_key = tavily_key
        self._brave_key = brave_key
        self._max_results = max_results
        self._timeout = timeout
        self._region = region
        self._safe_search = safe_search

    # ------------------------------------------------------------------
    # Public tool
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> str:
        """Search the web and return a list of relevant results.

        Use for: finding current information, looking up facts, discovering
        URLs before fetching pages, research tasks.

        Args:
            query:       Search query string.
            max_results: Number of results to return (default from constructor).

        Returns:
            JSON string with a list of results, each containing
            ``title``, ``url``, ``snippet``.
            Returns an error string on failure.
        """
        try:
            if not query or not query.strip():
                return "Error: query must not be empty"

            n = int(max_results) if max_results is not None else self._max_results
            failures: list[str] = []

            # Layer 1: Tavily (AI-optimized, best for agent use)
            if self._tavily_key:
                results = await asyncio.to_thread(self._search_tavily, query, n)
                if results:
                    return self._format(results)
                failures.append("Tavily: failed (check API key or network)")
            else:
                failures.append("Tavily: skipped (no tavily_key provided)")

            # Layer 2: Brave Search (paid, high quality)
            if self._brave_key:
                results = await self._search_brave(query, n)
                if results:
                    return self._format(results)
                failures.append("Brave: failed (check API key or network)")
            else:
                failures.append("Brave: skipped (no brave_key provided)")

            # Layer 3: DuckDuckGo (free, no key)
            results, ddgs_err = await self._search_ddgs(query, n)
            if results:
                return self._format(results)
            failures.append(f"DuckDuckGo: {ddgs_err}")

            detail = "; ".join(failures)
            return (
                f"Error: all search backends failed for query: {query!r}. "
                f"Reasons: {detail}. "
                f"To fix: install at least one backend — "
                f"`pip install ddgs` (free, no key needed) or "
                f"`pip install tavily-python` (requires TAVILY_API_KEY)."
            )

        except Exception as e:
            return f"Error searching for {query!r}: {e}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format(self, results: List[SearchResult]) -> str:
        """Serialize results to JSON string."""
        return json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2)

    def _search_tavily(self, query: str, n: int) -> Optional[List[SearchResult]]:
        """Query Tavily AI Search (synchronous, run via asyncio.to_thread)."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self._tavily_key)
            resp = client.search(query, max_results=n)
        except ImportError:
            return None
        except Exception:
            return None

        items = resp.get("results", [])
        results = []
        for item in items[:n]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            ))
        return results or None

    async def _search_brave(self, query: str, n: int) -> Optional[List[SearchResult]]:
        """Query Brave Search REST API."""
        safe_map = {"strict": "strict", "moderate": "moderate", "off": "off"}
        params = {
            "q": query,
            "count": min(n, 20),  # Brave max per request is 20
            "safesearch": safe_map.get(self._safe_search, "moderate"),
        }
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._brave_key,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.get(_BRAVE_SEARCH_URL, params=params, headers=headers)
                if r.status_code in (401, 403):
                    return None  # Invalid key — fall through
                r.raise_for_status()
                data = r.json()
        except Exception:
            return None

        items = data.get("web", {}).get("results", [])
        results = []
        for item in items[:n]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            ))
        return results or None

    async def _search_ddgs(
        self, query: str, n: int
    ) -> tuple[Optional[List[SearchResult]], str]:
        """Query DuckDuckGo via duckduckgo-search (ddgs) library.

        Returns:
            (results, error_msg) — results is None on failure;
            error_msg describes the failure reason.
        """
        try:
            from ddgs import DDGS

            def _run() -> list:
                # DDGS is synchronous; run in thread to avoid blocking the event loop.
                with DDGS() as ddgs:
                    return list(ddgs.text(
                        query,
                        region=self._region,
                        safesearch=self._safe_search,
                        max_results=n,
                    ))

            items = await asyncio.to_thread(_run)
        except ImportError:
            return None, "package not installed — run `pip install ddgs`"
        except Exception as e:
            return None, f"runtime error: {e}"

        results = []
        for item in items[:n]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("href", item.get("url", "")),
                snippet=item.get("body", ""),
            ))
        return (results or None), "no results returned"

    def __repr__(self) -> str:
        tv = "yes" if self._tavily_key else "no"
        br = "yes" if self._brave_key else "no"
        return f"<Search tavily={tv} brave={br} max_results={self._max_results}>"
