"""
Plain text file attachment with default reader.
"""
from typing import Optional, Callable, Union, Awaitable
import os
from .base import Attachment, MimeType, detect_content_type, AttachmentContent, ContentType


def default_txt_reader(content: str) -> AttachmentContent:
    """
    Default synchronous TXT reader.
    
    Args:
        content: Local file path or URL to text file
        
    Returns:
        AttachmentContent with extracted text and metadata
    """
    try:
        content_type = detect_content_type(content)
        
        # Handle different content types
        if content_type == ContentType.URL:
            # Download URL
            import urllib.request
            with urllib.request.urlopen(content) as response:
                text = response.read().decode('utf-8')
        
        elif content_type == ContentType.LOCAL_PATH:
            # Local path: check if file exists
            if not os.path.exists(content):
                return AttachmentContent(content="", meta={"error": f"File not found: {content}"})
            with open(content, 'r', encoding='utf-8') as f:
                text = f.read()
        
        elif content_type == ContentType.TEXT:
            # Plain text: treat as direct text content
            text = content
        
        else:  # ContentType.DATA_URI
            # Data URI: not supported for TXT
            return AttachmentContent(content="", meta={"error": "Data URI not supported for TXT files"})
        
        meta = {
            "length": len(text),
            "lines": text.count('\n') + 1,
        }
        
        return AttachmentContent(content=text, meta=meta)
    
    except Exception as e:
        return AttachmentContent(content="", meta={"error": str(e)})


async def default_txt_reader_async(content: str) -> AttachmentContent:
    """
    Default asynchronous TXT reader.
    
    Args:
        content: Local file path or URL to text file
        
    Returns:
        AttachmentContent with extracted text and metadata
    """
    try:
        import aiohttp
        import aiofiles
        
        content_type = detect_content_type(content)
        
        # Handle different content types
        if content_type == ContentType.URL:
            # Download URL asynchronously
            async with aiohttp.ClientSession() as session:
                async with session.get(content) as response:
                    response.raise_for_status()
                    text = await response.text()
        
        elif content_type == ContentType.LOCAL_PATH:
            # Local path: check if file exists
            if not os.path.exists(content):
                return AttachmentContent(content="", meta={"error": f"File not found: {content}"})
            async with aiofiles.open(content, 'r', encoding='utf-8') as f:
                text = await f.read()
        
        elif content_type == ContentType.TEXT:
            # Plain text: treat as direct text content
            text = content
        
        else:  # ContentType.DATA_URI
            # Data URI: not supported for TXT
            return AttachmentContent(content="", meta={"error": "Data URI not supported for TXT files"})
        
        meta = {
            "length": len(text),
            "lines": text.count('\n') + 1,
        }
        
        return AttachmentContent(content=text, meta=meta)
    
    except Exception as e:
        return AttachmentContent(content="", meta={"error": str(e)})


class TXT(Attachment):
    """Plain text file attachment"""
    
    def __init__(
        self, 
        source: str,
        reader: Optional[Callable[[str], Union[AttachmentContent, Awaitable[AttachmentContent]]]] = None
    ):
        """
        Create a text file attachment.
        
        Args:
            source: Local file path or URL to text file
            reader: Optional custom reader function (defaults to default_txt_reader)
        
        Examples:
            >>> TXT("notes.txt")
            >>> TXT("https://example.com/readme.txt")
        """
        reader = reader or default_txt_reader_async
        super().__init__(source, MimeType.TXT, reader)
