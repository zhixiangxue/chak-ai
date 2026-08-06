import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest

from chak.tools.std import pdf as pdf_module

pytestmark = pytest.mark.unit


class FakePdfResponse:
    def __init__(self, content: bytes = b"%PDF-1.4\n", content_type: str = "application/pdf"):
        self.content = content
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        pass


def test_resolve_pdf_url_sends_browser_like_headers(monkeypatch):
    source = "https://example.com/file.pdf"
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakePdfResponse()

    monkeypatch.setattr(httpx, "get", fake_get)

    local_path = pdf_module._resolve_pdf(source)

    try:
        assert captured["url"] == source
        assert captured["kwargs"]["follow_redirects"] is True
        assert captured["kwargs"]["timeout"] == 60
        assert captured["kwargs"]["headers"]["User-Agent"].startswith("Mozilla/5.0")
        assert "application/pdf" in captured["kwargs"]["headers"]["Accept"]
        assert os.path.exists(local_path)
    finally:
        os.unlink(local_path)


def test_resolve_pdf_url_falls_back_to_requests_after_httpx_error(monkeypatch):
    source = "https://example.com/file.pdf"
    request = httpx.Request("GET", source)
    captured = {}

    def failing_httpx_get(*args, **kwargs):
        raise httpx.ReadError("[SSL: UNEXPECTED_EOF_WHILE_READING]", request=request)

    def fake_requests_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakePdfResponse()

    monkeypatch.setattr(httpx, "get", failing_httpx_get)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_requests_get))

    local_path = pdf_module._resolve_pdf(source)

    try:
        assert captured["url"] == source
        assert captured["kwargs"]["allow_redirects"] is True
        assert captured["kwargs"]["timeout"] == pdf_module._PDF_DOWNLOAD_TIMEOUT
        assert captured["kwargs"]["headers"] == pdf_module._PDF_DOWNLOAD_HEADERS
        assert os.path.exists(local_path)
    finally:
        os.unlink(local_path)


def test_resolve_pdf_url_retries_requests_without_verify_for_certificate_errors(monkeypatch):
    source = "https://example.com/file.pdf"
    request = httpx.Request("GET", source)
    calls = []

    class FakeRequestsSSLError(Exception):
        pass

    def failing_httpx_get(*args, **kwargs):
        raise httpx.ReadError("[SSL: UNEXPECTED_EOF_WHILE_READING]", request=request)

    def fake_requests_get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise FakeRequestsSSLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        return FakePdfResponse()

    fake_requests = SimpleNamespace(
        get=fake_requests_get,
        exceptions=SimpleNamespace(SSLError=FakeRequestsSSLError),
    )
    monkeypatch.setattr(httpx, "get", failing_httpx_get)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    local_path = pdf_module._resolve_pdf(source)

    try:
        assert len(calls) == 2
        assert "verify" not in calls[0][1]
        assert calls[1][1]["verify"] is False
        assert calls[1][1]["headers"] == pdf_module._PDF_DOWNLOAD_HEADERS
        assert os.path.exists(local_path)
    finally:
        os.unlink(local_path)


def test_resolve_pdf_url_does_not_disable_verify_for_other_ssl_errors(monkeypatch):
    source = "https://example.com/file.pdf"
    request = httpx.Request("GET", source)
    calls = []

    class FakeRequestsSSLError(Exception):
        pass

    def failing_httpx_get(*args, **kwargs):
        raise httpx.ReadError("[SSL: UNEXPECTED_EOF_WHILE_READING]", request=request)

    def fake_requests_get(url, **kwargs):
        calls.append((url, kwargs))
        raise FakeRequestsSSLError("[SSL: WRONG_VERSION_NUMBER] wrong version number")

    fake_requests = SimpleNamespace(
        get=fake_requests_get,
        exceptions=SimpleNamespace(SSLError=FakeRequestsSSLError),
    )
    monkeypatch.setattr(httpx, "get", failing_httpx_get)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    with pytest.raises(FakeRequestsSSLError):
        pdf_module._resolve_pdf(source)

    assert len(calls) == 1
    assert "verify" not in calls[0][1]


def test_strip_code_fences_returns_bare_text_when_no_fence():
    text = "<table><tr><td>x</td></tr></table>"
    assert pdf_module._strip_code_fences(text) == text


def test_strip_code_fences_removes_single_language_fence():
    text = "```markdown\n<table></table>\n```"
    assert pdf_module._strip_code_fences(text) == "<table></table>"


def test_strip_code_fences_removes_doubled_trailing_fence():
    # glm-4.5v was observed emitting a doubled closing fence; a single-layer
    # peel left a stray ``` behind, so both trailing fences must be stripped.
    text = "```markdown\n<table></table>\n```\n```"
    assert pdf_module._strip_code_fences(text) == "<table></table>"


def test_strip_code_fences_handles_leading_fence_without_closing():
    text = "```html\n<table></table>"
    assert pdf_module._strip_code_fences(text) == "<table></table>"


def test_mode_rw_exposes_all_tools_by_default():
    available = pdf_module.Pdf().__available__()
    assert available == {
        "metadata", "outline", "search", "read_pages", "read_all",
        "render_page", "schema", "fill",
    }


def test_mode_r_hides_form_filling_tools():
    available = pdf_module.Pdf(mode="r").__available__()
    # schema is hidden together with fill: it exists solely to serve fill.
    assert "fill" not in available
    assert "schema" not in available
    # render_page stays: it only writes a derived PNG as a reading fallback.
    assert available == {
        "metadata", "outline", "search", "read_pages", "read_all",
        "render_page",
    }


def test_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="must be 'r' or 'rw'"):
        pdf_module.Pdf(mode="w")


class FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self, kind: str) -> str:
        return self._text


class FakeDoc:
    def __init__(self, pages_text: list[str]):
        self._pages = [FakePage(t) for t in pages_text]
        self.page_count = len(self._pages)

    def load_page(self, index: int) -> FakePage:
        return self._pages[index]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_search_coerces_string_numeric_args_from_llm(monkeypatch):
    """LLMs frequently pass numeric tool arguments as strings (e.g. "220"
    instead of 220). Without coercion, ``position - context_chars`` raises
    'unsupported operand type(s) for -: int and str'. search() must coerce
    max_results/context_chars the same way render_page() coerces page/dpi.
    """
    fake_doc = FakeDoc(["some text with the keyword inside it"])
    fake_pymupdf = SimpleNamespace(open=lambda path: fake_doc)

    monkeypatch.setattr(pdf_module, "_require_pdf_libs", lambda: (fake_pymupdf, None))
    monkeypatch.setattr(pdf_module, "_resolve_pdf", lambda source: "/tmp/fake.pdf")

    pdf = pdf_module.Pdf()

    # Simulates the exact failure mode: LLM sends string values for numeric params.
    result = pdf.search(
        source="fake.pdf",
        query="keyword",
        max_results="20",
        context_chars="10",
    )

    payload = json.loads(result)
    assert payload["results"][0]["page"] == 1
    assert "keyword" in payload["results"][0]["context"]

