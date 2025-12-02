"""
Web link attachment with default reader.
"""
from typing import Optional, Callable, Union, Awaitable
from .base import Attachment, MimeType, AttachmentContent


def default_link_reader(content: str) -> AttachmentContent:
    """
    Default synchronous link reader using requests.
    
    Args:
        content: URL to fetch
        
    Returns:
        AttachmentContent with extracted text and metadata
    """
    try:
        import requests
        
        response = requests.get(content, timeout=30)
        response.raise_for_status()
        
        # Get text content
        text = response.text
        
        meta = {
            "url": content,
            "status_code": response.status_code,
            "content_type": response.headers.get('Content-Type', ''),
            "content_length": len(text),
        }
        
        return AttachmentContent(content=text, meta=meta)
    
    except Exception as e:
        return AttachmentContent(content="", meta={"error": str(e), "url": content})


async def default_link_reader_async(content: str) -> AttachmentContent:
    """
    Default asynchronous link reader using aiohttp.
    
    Args:
        content: URL to fetch
        
    Returns:
        AttachmentContent with extracted text and metadata
    """
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(content, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                text = await response.text()
                
                meta = {
                    "url": content,
                    "status_code": response.status,
                    "content_type": response.headers.get('Content-Type', ''),
                    "content_length": len(text),
                }
                
                return AttachmentContent(content=text, meta=meta)
    
    except Exception as e:
        return AttachmentContent(content="", meta={"error": str(e), "url": content})


class Link(Attachment):
    """Web link attachment for fetching online content"""
    
    def __init__(
        self, 
        source: str,
        reader: Optional[Callable[[str], Union[AttachmentContent, Awaitable[AttachmentContent]]]] = None
    ):
        """
        Create a link attachment.
        
        Args:
            source: URL to fetch
            reader: Optional custom reader function (defaults to default_link_reader)
        
        Examples:
            >>> Link("https://example.com/article")
            >>> Link("https://api.example.com/data", reader=my_custom_reader)
        """
        reader = reader or default_link_reader_async
        super().__init__(source, MimeType.PLAIN, reader)
