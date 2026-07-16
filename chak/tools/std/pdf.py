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
from typing import Any

_PDF_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; chak-pdf/1.0)",
    "Accept": "application/pdf,application/octet-stream,*/*",
}
_PDF_DOWNLOAD_TIMEOUT = 60


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


def _require_pdf_libs():
    """Lazily import and return (pymupdf, pymupdf4llm).

    IMPORTANT: pymupdf.layout must NEVER be imported in this process.
    Its module-level activate() monkey-patches pymupdf._get_layout with an
    ONNX model predictor.  This irreversibly changes PyMuPDF's internal
    behaviour and causes pymupdf4llm 0.x to raise
    ``ValueError: min() iterable argument is empty`` on pages with complex
    tables, silently falling back to plain-text extraction.
    pymupdf4llm already provides its own layout analysis; the external
    pymupdf.layout package is neither needed nor safe to use here.
    """
    try:
        import pymupdf  # noqa: PLC0415
    except ImportError:
        raise ImportError("PyMuPDF is required. Run: pip install PyMuPDF")

    try:
        import pymupdf4llm  # noqa: PLC0415
    except ImportError:
        raise ImportError("pymupdf4llm is required. Run: pip install pymupdf4llm")

    return pymupdf, pymupdf4llm


def _is_certificate_verify_error(error: Exception) -> bool:
    message = str(error).lower()
    return "certificate_verify_failed" in message or "certificate verify failed" in message


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

        try:
            resp = httpx.get(
                source,
                follow_redirects=True,
                timeout=_PDF_DOWNLOAD_TIMEOUT,
                headers=_PDF_DOWNLOAD_HEADERS,
            )
        except httpx.HTTPError as httpx_error:
            try:
                import requests  # noqa: PLC0415
            except ImportError:
                if not _is_certificate_verify_error(httpx_error):
                    raise httpx_error
                resp = httpx.get(
                    source,
                    follow_redirects=True,
                    timeout=_PDF_DOWNLOAD_TIMEOUT,
                    headers=_PDF_DOWNLOAD_HEADERS,
                    verify=False,
                )
            else:
                try:
                    resp = requests.get(
                        source,
                        allow_redirects=True,
                        timeout=_PDF_DOWNLOAD_TIMEOUT,
                        headers=_PDF_DOWNLOAD_HEADERS,
                    )
                except requests.exceptions.SSLError as requests_ssl_error:
                    if not _is_certificate_verify_error(requests_ssl_error):
                        raise
                    resp = requests.get(
                        source,
                        allow_redirects=True,
                        timeout=_PDF_DOWNLOAD_TIMEOUT,
                        headers=_PDF_DOWNLOAD_HEADERS,
                        verify=False,
                    )
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


def _json(data: dict[str, Any] | list[Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)



def _content(text: str, max_chars: int | None) -> str:
    if max_chars is None:
        return text
    return text[:max_chars]


def _file_size_mb(size_bytes: int) -> float:
    return round(size_bytes / (1024 * 1024), 2)


def _text_density(average_chars_per_page: int) -> str:
    if average_chars_per_page <= 0:
        return "none"
    if average_chars_per_page < 500:
        return "low"
    if average_chars_per_page < 2000:
        return "medium"
    return "high"


def _size_category(file_size_mb: float, page_count: int, estimated_total_chars: int) -> str:
    if file_size_mb >= 100 or page_count >= 500 or estimated_total_chars >= 1_000_000:
        return "huge"
    if file_size_mb >= 25 or page_count >= 100 or estimated_total_chars >= 250_000:
        return "large"
    if file_size_mb >= 5 or page_count >= 25 or estimated_total_chars >= 50_000:
        return "medium"
    return "small"


def _likely_scanned_or_image_heavy(file_size_mb: float, page_count: int, average_chars_per_page: int) -> bool:
    if page_count <= 0:
        return False
    return file_size_mb >= 2 and average_chars_per_page < 200


def _page_numbers(start_page: int, end_page: int, page_count: int) -> list[int]:
    if start_page < 1:
        raise ValueError("start_page must be >= 1")
    if end_page < start_page:
        raise ValueError("end_page must be >= start_page")
    if end_page > page_count:
        raise ValueError(f"end_page must be <= total pages ({page_count})")
    return list(range(start_page - 1, end_page))


def _chunk_page(chunk: dict[str, Any], fallback: int) -> int:
    metadata = chunk.get("metadata") or {}
    return metadata.get("page") or metadata.get("page_number") or fallback


def _format_output(markdown_text: str, format: str, title: str = "PDF") -> str:
    if format == "markdown":
        return markdown_text
    if format == "txt":
        return _md_to_plain(markdown_text)
    if format == "html":
        return _md_to_html(markdown_text, title)
    raise ValueError("format must be one of: markdown, txt, html")


def _headings_from_markdown_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        page = _chunk_page(chunk, index + 1)
        for match in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, flags=re.M):
            headings.append(
                {
                    "level": len(match.group(1)),
                    "title": match.group(2).strip(),
                    "page": page,
                }
            )
    return headings


def _looks_like_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 120:
        return False
    if text.endswith((".", ",", ";", ":")):
        return False
    if re.match(r"^(chapter|section|part|appendix)\b", text, flags=re.I):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+\S", text):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if len(words) < 2:
        return False
    title_words = sum(1 for word in words if word[:1].isupper())
    return title_words / len(words) >= 0.6


def _headings_from_plain_text(doc: Any) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for page_index in range(doc.page_count):
        text = doc.load_page(page_index).get_text("text")
        for line in text.splitlines():
            title = line.strip()
            key = (page_index + 1, title)
            if key in seen or not _looks_like_heading(title):
                continue
            seen.add(key)
            headings.append({"level": 1, "title": title, "page": page_index + 1})
    return headings


def _fallback_markdown(doc: Any, pages: list[int]) -> str:
    """Extract text from pages using PyMuPDF native get_text.

    Fallback when pymupdf4llm raises ValueError on pages with empty or
    complex table structures (min() iterable argument is empty).
    """
    parts = []
    for page_index in pages:
        text = doc.load_page(page_index).get_text("text")
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _fallback_chunks(doc: Any, pages: list[int]) -> list[dict[str, Any]]:
    """Extract per-page chunks using PyMuPDF native get_text.

    Fallback when pymupdf4llm raises ValueError on pages with empty or
    complex table structures (min() iterable argument is empty).
    """
    chunks = []
    for page_index in pages:
        text = doc.load_page(page_index).get_text("text")
        chunks.append(
            {
                "text": text,
                "metadata": {"page": page_index + 1},
            }
        )
    return chunks


class Pdf:
    """Navigate and extract PDF content safely for LLM workflows.

    The default API is intentionally navigation-oriented: call metadata first,
    then read page ranges or search for relevant pages. All page fields and
    start_page/end_page arguments are 1-based PDF physical pages, not printed
    page labels shown inside the document.
    """

    def metadata(
        self,
        source: str,
    ) -> str:
        """Return PDF metadata and high-level document overview.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.

        Returns:
            JSON string with source, PDF physical page count, file size,
            structural metadata, TOC count, estimated total characters, sampled
            page text density, and compact document-scale signals. This method
            intentionally does not include page content; use read_pages when
            content is needed.
        """
        pymupdf, _ = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        file_size_bytes = Path(local_path).stat().st_size
        file_size_mb = _file_size_mb(file_size_bytes)
        with pymupdf.open(local_path) as doc:
            sample_pages = min(2, doc.page_count)
            sample_texts = [doc.load_page(i).get_text("text") for i in range(sample_pages)]
            sample_text = "\n\n".join(sample_texts)
            average_chars = int(len(sample_text) / sample_pages) if sample_pages else 0
            estimated_total_chars = average_chars * doc.page_count
            text_density = _text_density(average_chars)
            toc = doc.get_toc(simple=True)
            pdf_metadata = dict(doc.metadata or {})

            payload = {
                "source": source,
                "pages": doc.page_count,
                "file_size_bytes": file_size_bytes,
                "file_size_mb": file_size_mb,
                "pdf_format": pdf_metadata.get("format"),
                "document_title": pdf_metadata.get("title"),
                "document_author": pdf_metadata.get("author"),
                "document_subject": pdf_metadata.get("subject"),
                "document_keywords": pdf_metadata.get("keywords"),
                "pdf_creator": pdf_metadata.get("creator"),
                "pdf_producer": pdf_metadata.get("producer"),
                "creation_date": pdf_metadata.get("creationDate"),
                "modification_date": pdf_metadata.get("modDate"),
                "is_encrypted": bool(pdf_metadata.get("encryption")),
                "toc_items": len(toc),
                "has_toc": bool(toc),
                "sampled_pages": sample_pages,
                "average_chars_per_sampled_page": average_chars,
                "estimated_total_chars": estimated_total_chars,
                "text_density": text_density,
                "size_category": _size_category(file_size_mb, doc.page_count, estimated_total_chars),
                "likely_scanned_or_image_heavy": _likely_scanned_or_image_heavy(
                    file_size_mb,
                    doc.page_count,
                    average_chars,
                ),
            }
        return _json(payload)

    def outline(
        self,
        source: str,
    ) -> str:
        """Return the full PDF outline, falling back to inferred Markdown headings.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.

        Returns:
            JSON string with outline items. Standard PDF TOC items contain level,
            title, and page. The page field is the 1-based PDF physical page,
            not the printed page label shown inside the document. When citing
            locations, prefer this PDF physical page value; printed page labels
            may be mentioned separately if they appear in the content. If the
            PDF has no standard TOC/bookmarks, headings are inferred from the
            full document using layout extraction, with a plain-text heading
            scan as a fallback if layout extraction fails.
        """
        pymupdf, _ = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        with pymupdf.open(local_path) as doc:
            toc = doc.get_toc(simple=True)
            if toc:
                items = [
                    {"level": level, "title": title, "page": page}
                    for level, title, page in toc
                ]
                return _json({"source": source, "type": "toc", "items": items, "total_items": len(toc)})

            # See _require_pdf_libs() docstring for why pymupdf.layout is banned.
            _, lib = _require_pdf_libs()
            try:
                chunks = lib.to_markdown(doc, page_chunks=True)
                headings = _headings_from_markdown_chunks(chunks)
                method = "layout_markdown"
            except Exception:
                headings = _headings_from_plain_text(doc)
                method = "plain_text"

            return _json(
                {
                    "source": source,
                    "type": "inferred_headings",
                    "method": method,
                    "items": headings,
                    "total_items": len(headings),
                }
            )

    def search(
        self,
        source: str,
        query: str,
        max_results: int = 20,
        context_chars: int = 220,
    ) -> str:
        """Search text quickly across the PDF and return page-level matches.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.
            query: Case-insensitive keyword or phrase to search.
            max_results: Maximum matches returned.
            context_chars: Characters of context around each match.

        Returns:
            JSON string with matching page numbers and snippets. The page field
            is the 1-based PDF physical page, not the printed page label shown
            inside the document. When citing search results, prefer this PDF
            physical page value; printed page labels may be mentioned separately
            if they appear in the snippet.
        """
        if not query:
            raise ValueError("query is required")

        pymupdf, _ = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        results = []
        query_lower = query.lower()
        with pymupdf.open(local_path) as doc:
            for page_index in range(doc.page_count):
                text = doc.load_page(page_index).get_text("text")
                position = text.lower().find(query_lower)
                if position < 0:
                    continue
                start = max(0, position - context_chars)
                end = min(len(text), position + len(query) + context_chars)
                results.append(
                    {
                        "page": page_index + 1,
                        "position": position,
                        "context": text[start:end],
                    }
                )
                if len(results) >= max_results:
                    break

            payload = {
                "source": source,
                "query": query,
                "pages": doc.page_count,
                "results": results,
            }
        return _json(payload)

    def read_pages(
        self,
        source: str,
        start_page: int,
        end_page: int,
        format: str = "markdown",
        max_chars: int | None = None,
    ) -> str:
        """Read a page range with layout-aware extraction.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.
            start_page: First PDF physical page to read, 1-based.
            end_page: Last PDF physical page to read, inclusive and 1-based.
            format: Output format: markdown, txt, html, or json.
            max_chars: Maximum characters returned for text formats. If None,
                the full extracted range is returned.

        Returns:
            Text for markdown/txt/html, or JSON page chunks for json. Text
            formats include a JSON-like navigation header followed by content.
            All page values are 1-based PDF physical pages, not printed page
            labels shown inside the document. When citing extracted content,
            prefer the PDF physical page values from the navigation header or
            JSON chunks; printed page labels inside the content may be mentioned
            separately but should not be mixed into the same page range.
        """
        valid_formats = {"markdown", "txt", "html", "json"}
        if format not in valid_formats:
            raise ValueError(f"Unsupported format '{format}'. Choose from: {', '.join(sorted(valid_formats))}")

        # Do NOT import pymupdf.layout here — see _require_pdf_libs() docstring
        # for the full explanation of why it is globally banned.
        pymupdf, lib = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        with pymupdf.open(local_path) as doc:
            pages = _page_numbers(start_page, end_page, doc.page_count)
            if format == "json":
                try:
                    chunks = lib.to_markdown(doc, pages=pages, page_chunks=True)
                    extraction_method = "layout"
                except ValueError:
                    chunks = _fallback_chunks(doc, pages)
                    extraction_method = "plain_text"
                payload = {
                    "source": source,
                    "start_page": start_page,
                    "end_page": end_page,
                    "total_pages": doc.page_count,
                    "extraction_method": extraction_method,
                    "pages": [
                        {
                            "page": _chunk_page(chunk, start_page + index),
                            "content": chunk.get("text", ""),
                        }
                        for index, chunk in enumerate(chunks)
                    ],
                }
                return _json(payload)

            try:
                md = lib.to_markdown(doc, pages=pages)
                extraction_method = "layout"
            except ValueError:
                md = _fallback_markdown(doc, pages)
                extraction_method = "plain_text"
            formatted = _format_output(md, format, Path(source).stem)
            content = _content(formatted, max_chars)
            header = _json(
                {
                    "source": source,
                    "start_page": start_page,
                    "end_page": end_page,
                    "total_pages": doc.page_count,
                    "format": format,
                    "extraction_method": extraction_method,
                    "next_page": end_page + 1 if end_page < doc.page_count else None,
                }
            )
            return f"{header}\n\n{content}"

    def read_all(
        self,
        source: str,
        format: str = "markdown",
        max_chars: int | None = None,
    ) -> str:
        """Read the full PDF.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.
            format: Output format: markdown, txt, or html.
            max_chars: Maximum characters returned. If None, the full extracted
                document is returned. For large PDFs, prefer metadata, outline,
                search, and read_pages unless full text is explicitly required.

        Returns:
            Full-document text with navigation metadata. The pages field is the
            PDF physical page count, not the document's printed page label range.
            When citing content, prefer PDF physical page values over printed
            page labels shown inside the document.
        """
        valid_formats = {"markdown", "txt", "html"}
        if format not in valid_formats:
            raise ValueError(f"Unsupported format '{format}'. Choose from: {', '.join(sorted(valid_formats))}")

        # See _require_pdf_libs() docstring for why pymupdf.layout is banned.
        pymupdf, lib = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        with pymupdf.open(local_path) as doc:
            try:
                md = lib.to_markdown(doc)
                extraction_method = "layout"
            except ValueError:
                md = _fallback_markdown(doc, list(range(doc.page_count)))
                extraction_method = "plain_text"
            formatted = _format_output(md, format, Path(source).stem)
            content = _content(formatted, max_chars)
            header = _json(
                {
                    "source": source,
                    "pages": doc.page_count,
                    "format": format,
                    "extraction_method": extraction_method,
                }
            )
            return f"{header}\n\n{content}"
