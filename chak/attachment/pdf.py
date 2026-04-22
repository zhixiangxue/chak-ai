"""
PDF document attachment with pymupdf4llm readers.

Produces high-quality Markdown output (tables, headings, layout preserved) using
pymupdf4llm, which significantly outperforms plain-text fitz extraction for LLM use.

Metadata fields:
    - pages: Total page count
"""
import asyncio
import os
import tempfile
from typing import Awaitable, Callable, Optional, Union

from .base import Attachment, AttachmentContent, ContentType, MimeType, detect_content_type


def _pymupdf4llm():
    try:
        import pymupdf4llm  # noqa: PLC0415
        return pymupdf4llm
    except ImportError:
        raise ImportError("pymupdf4llm is required. Run: pip install pymupdf4llm")


def _local_path_from_url_sync(url: str) -> str:
    """Download a URL to a temp file and return its path."""
    import urllib.request  # noqa: PLC0415
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


async def _local_path_from_url_async(url: str) -> str:
    """Async-download a URL to a temp file and return its path."""
    try:
        import aiohttp  # noqa: PLC0415
    except ImportError:
        raise ImportError("aiohttp is required for async URL downloads. Run: pip install aiohttp")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} when downloading {url}")
            data = await resp.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(data)
    tmp.close()
    return tmp.name


def _extract(file_path: str) -> AttachmentContent:
    """Run pymupdf4llm extraction and return AttachmentContent."""
    lib = _pymupdf4llm()
    try:
        import fitz  # noqa: PLC0415 — bundled with pymupdf4llm
        md = lib.to_markdown(file_path)
        doc = fitz.open(file_path)
        meta = {
            "pages": doc.page_count,
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
        }
        doc.close()
        return AttachmentContent(content=md, meta=meta)
    except Exception as e:
        return AttachmentContent(content="", meta={"error": str(e)})


def default_pdf_reader(source: str) -> AttachmentContent:
    """Synchronous PDF reader using pymupdf4llm (returns Markdown).

    Args:
        source: Local file path or HTTPS URL to a PDF.

    Returns:
        AttachmentContent with Markdown content and page count metadata.
    """
    content_type = detect_content_type(source)
    if content_type == ContentType.URL:
        file_path = _local_path_from_url_sync(source)
    elif content_type == ContentType.LOCAL_PATH:
        if not os.path.exists(source):
            return AttachmentContent(content="", meta={"error": f"File not found: {source}"})
        file_path = source
    else:
        return AttachmentContent(content="", meta={"error": f"Unsupported content type: {content_type}"})
    return _extract(file_path)


async def default_pdf_reader_async(source: str) -> AttachmentContent:
    """Asynchronous PDF reader using pymupdf4llm (returns Markdown).

    Downloads URLs asynchronously; extraction runs in a thread pool to avoid
    blocking the event loop.

    Args:
        source: Local file path or HTTPS URL to a PDF.

    Returns:
        AttachmentContent with Markdown content and page count metadata.
    """
    content_type = detect_content_type(source)
    if content_type == ContentType.URL:
        file_path = await _local_path_from_url_async(source)
    elif content_type == ContentType.LOCAL_PATH:
        if not os.path.exists(source):
            return AttachmentContent(content="", meta={"error": f"File not found: {source}"})
        file_path = source
    else:
        return AttachmentContent(content="", meta={"error": f"Unsupported content type: {content_type}"})

    # pymupdf4llm is CPU-bound — run in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract, file_path)


class PDF(Attachment):
    """PDF document attachment.

    Reads PDF content as Markdown via pymupdf4llm (tables, headings, layout preserved).
    Pass a custom ``reader`` to override extraction logic.

    Examples:
        >>> PDF("report.pdf")
        >>> PDF("https://example.com/doc.pdf")
        >>> pdf = PDF("document.pdf")
        >>> result = await pdf.aread()
        >>> print(result.meta["pages"])
    """

    def __init__(
        self,
        source: str,
        reader: Optional[Callable[[str], Union[AttachmentContent, Awaitable[AttachmentContent]]]] = None,
    ):
        """
        Args:
            source: Local file path or URL to a PDF file.
            reader: Optional custom reader (defaults to default_pdf_reader_async).
        """
        reader = reader or default_pdf_reader_async
        super().__init__(source, MimeType.PDF, reader)

