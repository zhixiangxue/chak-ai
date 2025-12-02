"""
CSV file attachment with default reader.
"""
from typing import Optional, Callable, Union, Awaitable
import os
import tempfile
from .base import Attachment, MimeType, detect_content_type, AttachmentContent, ContentType


def default_csv_reader(content: str) -> AttachmentContent:
    """
    Default synchronous CSV reader.
    
    Args:
        content: Local file path or URL to CSV file
        
    Returns:
        AttachmentContent with extracted text and metadata
    """
    try:
        import csv
        import urllib.request
        
        content_type = detect_content_type(content)
        
        # Handle different content types
        if content_type == ContentType.URL:
            # Download URL to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w+b') as tmp:
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
            # Data URI: not supported for CSV
            return AttachmentContent(content="", meta={"error": "Data URI not supported for CSV files"})
        
        # Now file_path is guaranteed to be a valid local file
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Format as tab-separated text
        text_parts = []
        for row in rows:
            text_parts.append("\t".join(row))
        
        text = "\n".join(text_parts)
        
        meta = {
            "rows": len(rows),
            "columns": len(rows[0]) if rows else 0,
        }
        
        # Clean up temp file if downloaded from URL
        if content_type == ContentType.URL:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        
        return AttachmentContent(content=text, meta=meta)
    
    except Exception as e:
        return AttachmentContent(content="", meta={"error": str(e)})


async def default_csv_reader_async(content: str) -> AttachmentContent:
    """
    Default asynchronous CSV reader.
    
    Args:
        content: Local file path or URL to CSV file
        
    Returns:
        AttachmentContent with extracted text and metadata
    """
    try:
        import csv
        import aiohttp
        import aiofiles
        
        content_type = detect_content_type(content)
        
        # Handle different content types
        if content_type == ContentType.URL:
            # Download URL to temp file asynchronously
            async with aiohttp.ClientSession() as session:
                async with session.get(content) as response:
                    response.raise_for_status()
                    data = await response.read()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='wb') as tmp:
                        tmp.write(data)
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
            # Data URI: not supported for CSV
            return AttachmentContent(content="", meta={"error": "Data URI not supported for CSV files"})
        
        # Now file_path is guaranteed to be a valid local file
        async with aiofiles.open(file_path, 'r', encoding='utf-8', newline='') as f:
            content_str = await f.read()
            reader = csv.reader(content_str.splitlines())
            rows = list(reader)
        
        # Format as tab-separated text
        text_parts = []
        for row in rows:
            text_parts.append("\t".join(row))
        
        text = "\n".join(text_parts)
        
        meta = {
            "rows": len(rows),
            "columns": len(rows[0]) if rows else 0,
        }
        
        # Clean up temp file if downloaded from URL
        if content_type == ContentType.URL:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        
        return AttachmentContent(content=text, meta=meta)
    
    except Exception as e:
        return AttachmentContent(content="", meta={"error": str(e)})


class CSV(Attachment):
    """CSV file attachment"""
    
    def __init__(
        self, 
        source: str,
        reader: Optional[Callable[[str], Union[AttachmentContent, Awaitable[AttachmentContent]]]] = None
    ):
        """
        Create a CSV file attachment.
        
        Args:
            source: Local file path or URL to CSV file
            reader: Optional custom reader function (defaults to default_csv_reader)
        
        Examples:
            >>> CSV("data.csv")
            >>> CSV("https://example.com/dataset.csv")
        """
        reader = reader or default_csv_reader_async
        super().__init__(source, MimeType.CSV, reader)
