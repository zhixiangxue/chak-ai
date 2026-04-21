"""
Http: Built-in HTTP client tool for chak

Provides five public methods exposed as LLM tools via NativeObjectTool:

    - get:    HTTP GET  — query APIs, fetch JSON/text resources
    - post:   HTTP POST — submit data, call REST endpoints
    - put:    HTTP PUT  — replace resources
    - patch:  HTTP PATCH — partial update resources
    - delete: HTTP DELETE — delete resources

Usage:
    from chak.tools.std import Http
    http = Http()
    conv = Conversation(model, tools=[http])

Dependencies:
    httpx — already a core chak dependency, no extras needed.
"""

from typing import Optional

import httpx

_USER_AGENT = "Mozilla/5.0 (compatible; chak-ai/1.0)"
_MAX_REDIRECTS = 5


class Http:
    """Full HTTP client tool for calling REST APIs and external services.

    Exposes five LLM-callable methods via NativeObjectTool:
        - get / post / put / patch / delete

    Each method returns a plain-text response block containing the HTTP
    status line and response body, making it easy for the LLM to parse
    the result and detect errors.

    Example::

        http = Http(timeout=30)
        conv = Conversation(model, tools=[http])
    """

    def __init__(
        self,
        timeout: int = 30,
        default_headers: Optional[dict] = None,
    ):
        """
        Args:
            timeout:         Request timeout in seconds (default 30).
            default_headers: Headers added to every request (e.g. auth tokens).
                             Per-request headers override these on conflict.
        """
        self._timeout = timeout
        self._default_headers = default_headers or {}

    # ------------------------------------------------------------------
    # Public tools
    # ------------------------------------------------------------------

    async def get(
        self,
        url: str,
        headers: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> str:
        """Send an HTTP GET request.

        Use for: fetching JSON from REST APIs, reading remote resources,
        calling read-only endpoints.

        Args:
            url:     Target URL.
            headers: Additional request headers (merged with default_headers).
            params:  URL query parameters as a dict, e.g. {"page": "1"}.

        Returns:
            Response block with status line and body, or an error string.
        """
        try:
            async with self._client() as client:
                r = await client.get(url, headers=self._merge(headers), params=params)
                return self._format(r)
        except Exception as e:
            return f"Error GET {url}: {e}"

    async def post(
        self,
        url: str,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        body: Optional[str] = None,
    ) -> str:
        """Send an HTTP POST request.

        Use for: creating resources, submitting forms, calling write endpoints.

        Args:
            url:     Target URL.
            headers: Additional request headers.
            json:    Request body as a JSON-serializable dict.
                     Sets Content-Type: application/json automatically.
            body:    Raw string body (used when json is not provided).

        Returns:
            Response block with status line and body, or an error string.
        """
        try:
            async with self._client() as client:
                r = await client.post(
                    url,
                    headers=self._merge(headers),
                    json=json,
                    content=body.encode() if body and json is None else None,
                )
                return self._format(r)
        except Exception as e:
            return f"Error POST {url}: {e}"

    async def put(
        self,
        url: str,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        body: Optional[str] = None,
    ) -> str:
        """Send an HTTP PUT request.

        Use for: replacing an existing resource entirely.

        Args:
            url:     Target URL.
            headers: Additional request headers.
            json:    Request body as a JSON-serializable dict.
            body:    Raw string body (used when json is not provided).

        Returns:
            Response block with status line and body, or an error string.
        """
        try:
            async with self._client() as client:
                r = await client.put(
                    url,
                    headers=self._merge(headers),
                    json=json,
                    content=body.encode() if body and json is None else None,
                )
                return self._format(r)
        except Exception as e:
            return f"Error PUT {url}: {e}"

    async def patch(
        self,
        url: str,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        body: Optional[str] = None,
    ) -> str:
        """Send an HTTP PATCH request.

        Use for: partially updating an existing resource.

        Args:
            url:     Target URL.
            headers: Additional request headers.
            json:    Request body as a JSON-serializable dict.
            body:    Raw string body (used when json is not provided).

        Returns:
            Response block with status line and body, or an error string.
        """
        try:
            async with self._client() as client:
                r = await client.patch(
                    url,
                    headers=self._merge(headers),
                    json=json,
                    content=body.encode() if body and json is None else None,
                )
                return self._format(r)
        except Exception as e:
            return f"Error PATCH {url}: {e}"

    async def delete(
        self,
        url: str,
        headers: Optional[dict] = None,
    ) -> str:
        """Send an HTTP DELETE request.

        Use for: removing a remote resource.

        Args:
            url:     Target URL.
            headers: Additional request headers.

        Returns:
            Response block with status line and body, or an error string.
        """
        try:
            async with self._client() as client:
                r = await client.delete(url, headers=self._merge(headers))
                return self._format(r)
        except Exception as e:
            return f"Error DELETE {url}: {e}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=_MAX_REDIRECTS,
            timeout=self._timeout,
            headers={"User-Agent": _USER_AGENT},
        )

    def _merge(self, extra: Optional[dict]) -> dict:
        """Merge default_headers with per-request headers (extra wins on conflict)."""
        merged = dict(self._default_headers)
        if extra:
            merged.update(extra)
        return merged

    def _format(self, r: httpx.Response) -> str:
        """Format response as a readable block for LLM consumption."""
        status = f"HTTP {r.status_code} {r.reason_phrase}"
        ctype = r.headers.get("content-type", "")
        return f"{status}\nContent-Type: {ctype}\n\n{r.text}"

    def __repr__(self) -> str:
        return f"<Http timeout={self._timeout}>"
