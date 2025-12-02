"""
Image attachment type.
"""
from typing import Optional
from .base import Attachment, MimeType


class Image(Attachment):
    """Image attachment for vision-enabled models"""
    
    def __init__(self, source: str, mime_type: Optional[MimeType] = None):
        """
        Create an image attachment.
        
        Args:
            source: Image URL or base64 data URI (e.g., "data:image/jpeg;base64,...")
            mime_type: Image MIME type (auto-detected from URL extension if not specified)
        
        Examples:
            >>> Image("https://example.com/photo.jpg")
            >>> Image("https://example.com/icon.png", MimeType.PNG)
            >>> Image("data:image/jpeg;base64,/9j/4AAQ...")
        """
        if mime_type is None:
            mime_type = self._auto_detect(source)
        super().__init__(source, mime_type)
        self._validate_mime_type("image/")
    
    @staticmethod
    def _auto_detect(source: str) -> MimeType:
        """Auto-detect image MIME type from URL or data URI"""
        if source.startswith("data:image/"):
            # Extract from data URI
            mime_str = source.split(";")[0].replace("data:", "")
            try:
                return MimeType(mime_str)
            except ValueError:
                return MimeType.JPEG  # Default for unknown formats
        elif source.lower().endswith((".jpg", ".jpeg")):
            return MimeType.JPEG
        elif source.lower().endswith(".png"):
            return MimeType.PNG
        elif source.lower().endswith(".gif"):
            return MimeType.GIF
        elif source.lower().endswith(".webp"):
            return MimeType.WEBP
        else:
            return MimeType.JPEG  # Default
