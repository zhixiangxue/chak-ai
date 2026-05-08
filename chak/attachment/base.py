"""
Base classes and utilities for attachment handling.
"""
from enum import Enum
from typing import Optional, Callable, Union, Dict, Any, Awaitable
import asyncio
import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field


class ContentType(str, Enum):
    """Content type classification for attachment sources"""
    DATA_URI = "data_uri"        # Base64-encoded data URI (e.g., data:image/png;base64,...)
    URL = "url"                   # HTTP/HTTPS/FTP URL
    LOCAL_PATH = "local_path"     # Existing local file path
    TEXT = "text"                 # Plain text content (default fallback)


class MimeType(str, Enum):
    """Supported MIME types for multimodal inputs"""
    # Text
    PLAIN = "text/plain"
    
    # Images
    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    WEBP = "image/webp"
    
    # Audio
    WAV = "audio/wav"
    MP3 = "audio/mpeg"
    OGG = "audio/ogg"
    
    # Video (future support)
    MP4 = "video/mp4"
    WEBM = "video/webm"
    
    # Documents
    PDF = "application/pdf"
    DOC = "application/msword"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    XLS = "application/vnd.ms-excel"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    CSV = "text/csv"
    TXT = "text/plain"  # Alias for PLAIN
    
    @property
    def media_type(self) -> str:
        """Get media type (primary type) from MIME type
        
        Returns:
            Media type (e.g., 'text', 'image', 'audio', 'video')
        """
        return self.value.split('/')[0]
    
    @property
    def subtype(self) -> str:
        """Get subtype from MIME type
        
        Returns:
            Subtype (e.g., 'plain', 'jpeg', 'wav', 'mp4')
        """
        return self.value.split('/')[1]
    
    def is_text(self) -> bool:
        """Check if this is a text MIME type"""
        return self.media_type == 'text'
    
    def is_image(self) -> bool:
        """Check if this is an image MIME type"""
        return self.media_type == 'image'
    
    def is_audio(self) -> bool:
        """Check if this is an audio MIME type"""
        return self.media_type == 'audio'
    
    def is_video(self) -> bool:
        """Check if this is a video MIME type"""
        return self.media_type == 'video'
    
    def is_document(self) -> bool:
        """Check if this is a document MIME type"""
        return self in {
            MimeType.PDF, MimeType.DOC, MimeType.DOCX,
            MimeType.XLS, MimeType.XLSX, MimeType.CSV
        } or (self.media_type == 'text' and self != MimeType.PLAIN)


class AttachmentContent(BaseModel):
    """Content extracted from an attachment"""
    content: str = Field(description="Extracted text content")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    
    model_config = ConfigDict(extra="allow")


def detect_content_type(content: str) -> ContentType:
    """
    Detect if content is a URL, local file path, data URI, or plain text.
    
    Args:
        content: Content string to detect
        
    Returns:
        ContentType enum value:
        - ContentType.DATA_URI: Base64-encoded data URI (e.g., data:image/png;base64,...)
        - ContentType.URL: HTTP/HTTPS/FTP URL
        - ContentType.LOCAL_PATH: Existing local file path
        - ContentType.TEXT: Plain text content (default fallback)
    """
    if not isinstance(content, str):
        return ContentType.TEXT
    
    # Check for data URI (starts with "data:")
    if content.startswith("data:"):
        return ContentType.DATA_URI
    
    # Check for URL using urlparse (more robust than startswith)
    try:
        from urllib.parse import urlparse
        parsed = urlparse(content)
        if parsed.scheme in ('http', 'https', 'ftp', 'file'):
            return ContentType.URL
    except Exception:
        pass
    
    # Check if it's a local file path (exists on filesystem)
    if os.path.exists(content):
        return ContentType.LOCAL_PATH
    
    # Try to parse as Path object (could be relative path)
    try:
        path = Path(content)
        if path.exists():
            return ContentType.LOCAL_PATH
    except (OSError, ValueError):
        pass
    
    # Default: treat as plain text content
    return ContentType.TEXT


class Attachment:
    """Base attachment class for multimodal content"""
    
    def __init__(
        self, 
        source: str, 
        mime_type: Optional[MimeType] = None,
        reader: Optional[Callable[[str], Union['AttachmentContent', Awaitable['AttachmentContent']]]] = None
    ):
        """
        Create an attachment.
        
        Args:
            source: URL, local file path, or base64-encoded data URI
            mime_type: MIME type (optional, auto-detected if not specified)
            reader: Optional callable to read and parse content (for document types)
                   Returns AttachmentContent with content and meta
                   Can be sync or async function
        """
        self.source = source
        self.mime_type = mime_type
        self.reader = reader
        self._cached_result: Optional['AttachmentContent'] = None
    
    @property
    def content(self) -> Optional[str]:
        """Get extracted text content (lazy loaded, only available after read/aread)"""
        if self._cached_result is None:
            return None
        return self._cached_result.content
    
    @property
    def meta(self) -> Optional[Dict[str, Any]]:
        """Get metadata from reader (only available after read/aread)"""
        if self._cached_result is None:
            return None
        return self._cached_result.meta
    
    @property
    def is_loaded(self) -> bool:
        """Check if content has been loaded"""
        return self._cached_result is not None

    @property
    def link(self) -> str:
        """Canonical attachment link string for frontend rendering.

        Format: [attachment://<source>]

        Example:
            >>> att = Attachment(source="https://s3.amazonaws.com/bucket/file.pdf")
            >>> att.link
            '[attachment://https://s3.amazonaws.com/bucket/file.pdf]'
        """
        return f"[attachment://{self.source}]"
        
    def _validate_mime_type(self, expected_prefix: str):
        """Validate MIME type matches expected media type"""
        if self.mime_type and not self.mime_type.startswith(expected_prefix):
            raise ValueError(
                f"Invalid MIME type for {expected_prefix}: got {self.mime_type}"
            )
    
    def read(self) -> Optional['AttachmentContent']:
        """
        Synchronously read and parse attachment content (for document types).
        Only works with synchronous readers.
        
        Returns:
            AttachmentContent with content and meta, or None if not a document
            
        Raises:
            RuntimeError: If reader is async (use aread() instead)
        """
        if self._cached_result is not None:
            return self._cached_result
            
        if self.reader is None:
            return None
        
        # Check if reader is async
        if asyncio.iscoroutinefunction(self.reader):
            raise RuntimeError(
                f"Cannot use sync read() with async reader. "
                f"Use 'await attachment.aread()' instead."
            )
            
        try:
            result = self.reader(self.source)  # Use self.source instead of self.content
            
            # Ensure result is AttachmentContent
            if isinstance(result, AttachmentContent):
                self._cached_result = result
            else:
                # Convert dict to AttachmentContent for backward compatibility
                # Support both 'text'/'metadata' and 'content'/'meta'
                if 'text' in result:
                    self._cached_result = AttachmentContent(
                        content=result.get('text', ''),
                        meta=result.get('metadata', {})
                    )
                else:
                    self._cached_result = AttachmentContent(**result)
            
            return self._cached_result
        except Exception as e:
            # Return empty result on error (as per user requirement)
            return AttachmentContent(content="", meta={"error": str(e)})
    
    async def aread(self) -> Optional['AttachmentContent']:
        """
        Asynchronously read and parse attachment content (for document types).
        Works with both sync and async readers.
        
        Returns:
            AttachmentContent with content and meta, or None if not a document
        """
        if self._cached_result is not None:
            return self._cached_result
            
        if self.reader is None:
            return None
            
        try:
            # Support both sync and async readers
            if asyncio.iscoroutinefunction(self.reader):
                result = await self.reader(self.source)  # Use self.source instead of self.content
            else:
                result = self.reader(self.source)  # Use self.source instead of self.content
            
            # Ensure result is AttachmentContent
            if isinstance(result, AttachmentContent):
                self._cached_result = result
            else:
                # Convert dict to AttachmentContent for backward compatibility
                # Support both 'text'/'metadata' and 'content'/'meta'
                if 'text' in result:
                    self._cached_result = AttachmentContent(
                        content=result.get('text', ''),
                        meta=result.get('metadata', {})
                    )
                else:
                    self._cached_result = AttachmentContent(**result)
            
            return self._cached_result
        except Exception as e:
            # Return empty result on error (as per user requirement)
            return AttachmentContent(content="", meta={"error": str(e)})
