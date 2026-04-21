"""
Pdf: Built-in PDF reader tool for chak

Extracts text, tables, and structured content from PDF files (local or remote).
Uses pymupdf4llm for high-accuracy local extraction (97% table accuracy, no LLM needed).

Usage:
    from chak.tools.std import Pdf
    pdf = Pdf()
    conv = Conversation(model, tools=[pdf])

Supported output formats:
    markdown  — Markdown with tables and headings (default, best for LLM input)
    txt       — plain text, no syntax
    html      — styled HTML document
    json      — per-page JSON array

Dependencies:
    pymupdf4llm  — pip install pymupdf4llm
    markdown     — pip install markdown  (only needed for html format)
"""

import json
import re
import tempfile
from pathlib import Path
from typing import Literal

_HTML_CSS = """
  body  { font-family: Arial, sans-serif; margin: 0.75in; font-size: 9pt; color: #222; }
  h1   { color: #1F3864; font-size: 14pt; margin-bottom: 1em; }
  h2   { color: #1F3864; font-size: 12pt; font-weight: bold; margin: 1.2em 0 0.4em 0; }
  h3   { color: #1F3864; font-size: 10.5pt; font-weight: bold; margin: 1em 0 0.3em 0; }
  h4, h5, h6 { color: #333; font-size: 9.5pt; font-weight: bold; margin: 0.8em 0 0.2em 0; }
  p    { margin: 0.15em 0 0.4em 0; line-height: 1.6; }
  ul, ol { margin: 0.2em 0 0.4em 1.4em; padding: 0; }
  li   { margin: 0.1em 0; line-height: 1.5; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1em; font-size: 9pt; }
  th   { background: #1F3864; color: white; font-weight: bold;
         padding: 4px 8px; border: 1px solid #ccc; text-align: left; }
  td   { padding: 4px 8px; border: 1px solid #ccc; }
  tr:nth-child(even) td { background: #f5f5f5; }
  hr   { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
"""


def _require_pymupdf4llm():
    try:
        import pymupdf4llm  # noqa: PLC0415
        return pymupdf4llm
    except ImportError:
        raise ImportError("pymupdf4llm is required. Run: pip install pymupdf4llm")


def _md_to_plain(md: str) -> str:
    """Strip Markdown syntax to produce readable plain text."""
    txt = re.sub(r"\|[^\n]+", "", md)            # remove table rows
    txt = re.sub(r"^#+\s+", "", txt, flags=re.M)  # headings
    txt = re.sub(r"\*\*([^*]+)\*\*", r"\1", txt)  # bold
    txt = re.sub(r"\*([^*]+)\*", r"\1", txt)      # italic
    txt = re.sub(r"^[-*]\s+", "", txt, flags=re.M) # bullets
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _md_to_html(md: str, title: str = "PDF") -> str:
    """Convert Markdown to a complete styled HTML document."""
    try:
        import markdown as md_lib  # noqa: PLC0415
    except ImportError:
        raise ImportError("markdown is required for html format. Run: pip install markdown")
    body = md_lib.markdown(md, extensions=["tables", "fenced_code", "nl2br"])
    return (
        f"<!DOCTYPE html>\n<html>\n<head>\n"
        f'<meta charset="utf-8">\n<title>{title}</title>\n'
        f"<style>\n{_HTML_CSS}\n</style>\n</head>\n<body>\n"
        f"<h1>{title}</h1>\n{body}\n</body>\n</html>"
    )


def _resolve_pdf(source: str) -> str:
    """Return a local file path to the PDF, downloading if source is a URL."""
    if source.startswith(("http://", "https://")):
        try:
            import httpx  # noqa: PLC0415
        except ImportError:
            raise ImportError("httpx is required for URL sources. Run: pip install httpx")

        resp = httpx.get(source, follow_redirects=True, timeout=60)
        resp.raise_for_status()

        # Validate Content-Type
        ct = resp.headers.get("content-type", "")
        if "pdf" not in ct and not source.lower().endswith(".pdf"):
            raise ValueError(
                f"Source does not appear to be a PDF (Content-Type: {ct}). "
                "Only PDF files are supported."
            )

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name

    # Local file
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"File '{source}' does not have a .pdf extension. "
            "Only PDF files are supported."
        )
    return str(path)


class Pdf:
    """Read and extract content from PDF files (local path or HTTPS URL).

    Exposes a single LLM-callable method `read` via NativeObjectTool.

    Example::

        pdf = Pdf()
        conv = Conversation(model, tools=[pdf])
    """

    def read(
        self,
        source: str,
        format: str = "markdown",
    ) -> str:
        """Read a PDF file and return its content as text.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.
                    Non-PDF files are rejected with an error.
            format: Output format. One of:
                    - "markdown" (default) — Markdown with tables and headings,
                      best for LLM reasoning and token efficiency.
                    - "txt"      — Plain text, no Markdown syntax.
                    - "html"     — Styled HTML document.
                    - "json"     — JSON array, one object per page:
                                   [{"page": 1, "content": "..."}, ...]

        Returns:
            PDF content as a string in the requested format.

        Raises:
            ValueError:       If source is not a PDF or format is unsupported.
            FileNotFoundError: If a local path does not exist.
        """
        valid_formats = {"markdown", "txt", "html", "json"}
        if format not in valid_formats:
            raise ValueError(
                f"Unsupported format '{format}'. Choose from: {', '.join(sorted(valid_formats))}"
            )

        lib = _require_pymupdf4llm()
        local_path = _resolve_pdf(source)

        if format == "json":
            chunks = lib.to_markdown(local_path, page_chunks=True)
            pages = [
                {"page": c["metadata"].get("page", i + 1), "content": c["text"]}
                for i, c in enumerate(chunks)
            ]
            return json.dumps(pages, ensure_ascii=False, indent=2)

        md = lib.to_markdown(local_path)

        if format == "markdown":
            return md
        if format == "txt":
            return _md_to_plain(md)
        if format == "html":
            title = Path(source).stem
            return _md_to_html(md, title)

        # unreachable
        return md
