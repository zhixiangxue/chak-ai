"""  
PDF document attachment with PyMuPDF (fitz) readers.

Provides high-performance PDF text extraction with rich metadata support.

Metadata fields:
    - pages: Total page count
    - title: Document title
    - author: Document author
    - subject: Document subject
    - creator: Application that created the document
    - producer: PDF producer (converter)
    - creation_date: Document creation date
    - modification_date: Last modification date
"""
from typing import Optional, Callable, Union, Awaitable
import os
import tempfile
from .base import Attachment, MimeType, detect_content_type, AttachmentContent, ContentType


def default_pdf_reader(content: str) -> AttachmentContent:
    """
    Default synchronous PDF reader using PyMuPDF (fitz).
    
    Args:
        content: Local file path or URL to PDF
        
    Returns:
        AttachmentContent with extracted text and metadata
    """
    try:
        import fitz  # PyMuPDF
        
        content_type = detect_content_type(content)
        
        # Handle different content types
        if content_type == ContentType.URL:
            # Download URL to temp file
            import urllib.request
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                urllib.request.urlretrieve(content, tmp.name)
                file_path = tmp.name
        
        elif content_type == ContentType.LOCAL_PATH:
            # Local path: check if file exists
            if not os.path.exists(content):
                return AttachmentContent(content="", meta={"error": f"File not found: {content}"})
            file_path = content
        
        elif content_type == ContentType.TEXT:
            # Plain text: not a valid file path
            return AttachmentContent(content="", meta={"error": "Invalid content: expected file path or URL, got plain text"})
        
        else:  # ContentType.DATA_URI
            # Data URI: not supported for PDF
            return AttachmentContent(content="", meta={"error": "Data URI not supported for PDF files"})
        
        # Now file_path is guaranteed to be a valid local file
        # Read PDF with PyMuPDF
        doc = fitz.open(file_path)
        
        # Extract text from all pages
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        
        text = "\n".join(text_parts)
        
        # Extract rich metadata
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
        
        # Clean up temp file if downloaded from URL
        if content_type == ContentType.URL:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        
        return AttachmentContent(content=text, meta=meta)
    
    except Exception as e:
        return AttachmentContent(content="", meta={"error": str(e)})


async def default_pdf_reader_async(content: str) -> AttachmentContent:
    """
    Default asynchronous PDF reader using PyMuPDF (fitz).
    
    Args:
        content: Local file path or URL to PDF
        
    Returns:
        AttachmentContent with extracted text and metadata
    """
    try:
        import fitz  # PyMuPDF
        import aiohttp
        import aiofiles
        
        content_type = detect_content_type(content)
        
        # Handle different content types
        if content_type == ContentType.URL:
            # Download URL to temp file
            async with aiohttp.ClientSession() as session:
                async with session.get(content) as resp:
                    if resp.status == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(await resp.read())
                            file_path = tmp.name
                    else:
                        return AttachmentContent(content="", meta={"error": f"HTTP {resp.status}"})
        
        elif content_type == ContentType.LOCAL_PATH:
            # Local path: check if file exists
            if not os.path.exists(content):
                return AttachmentContent(content="", meta={"error": f"File not found: {content}"})
            file_path = content
        
        elif content_type == ContentType.TEXT:
            # Plain text: not a valid file path
            return AttachmentContent(content="", meta={"error": "Invalid content: expected file path or URL, got plain text"})
        
        else:  # ContentType.DATA_URI
            # Data URI: not supported for PDF
            return AttachmentContent(content="", meta={"error": "Data URI not supported for PDF files"})
        
        # Now file_path is guaranteed to be a valid local file
        # Read PDF file asynchronously
        async with aiofiles.open(file_path, 'rb') as f:
            content_bytes = await f.read()
        
        # Process PDF with PyMuPDF (CPU-bound, but fast)
        import io
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        
        # Extract text from all pages
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        
        text = "\n".join(text_parts)
        
        # Extract rich metadata
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
        
        # Clean up temp file if downloaded from URL
        if content_type == ContentType.URL:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        
        return AttachmentContent(content=text, meta=meta)
    
    except Exception as e:
        return AttachmentContent(content="", meta={"error": str(e)})


class PDF(Attachment):
    """PDF document attachment"""
    
    def __init__(
        self, 
        source: str,
        reader: Optional[Callable[[str], Union[AttachmentContent, Awaitable[AttachmentContent]]]] = None
    ):
        """
        Create a PDF attachment.
        
        Args:
            source: Local file path or URL to PDF file
            reader: Optional custom reader function (defaults to default_pdf_reader)
        
        Examples:
            >>> PDF("report.pdf")
            >>> PDF("https://example.com/doc.pdf")
            >>> PDF("file.pdf", reader=my_custom_pdf_reader)
            
            # Access metadata after reading
            >>> pdf = PDF("document.pdf")
            >>> result = pdf.read()
            >>> print(f"Pages: {result.meta['pages']}")
            >>> print(f"Author: {result.meta['author']}")
        """
        reader = reader or default_pdf_reader_async
        super().__init__(source, MimeType.PDF, reader)
