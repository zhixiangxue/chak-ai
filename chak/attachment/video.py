"""
Video attachment type.
"""
from typing import Optional
from .base import Attachment, MimeType


class Video(Attachment):
    """Video attachment (future support)"""
    
    def __init__(self, source: str, mime_type: Optional[MimeType] = None):
        """
        Create a video attachment.
        
        Args:
            source: Video URL or base64 data URI
            mime_type: Video MIME type (auto-detected from URL extension if not specified)
        
        Note:
            Video support is experimental and may not work with all providers.
        
        Examples:
            >>> Video("https://example.com/clip.mp4")
            >>> Video("https://example.com/video.webm", MimeType.WEBM)
        """
        if mime_type is None:
            mime_type = self._auto_detect(source)
        super().__init__(source, mime_type)
        self._validate_mime_type("video/")
    
    @staticmethod
    def _auto_detect(source: str) -> MimeType:
        """Auto-detect video MIME type from URL or data URI"""
        if source.startswith("data:video/"):
            mime_str = source.split(";")[0].replace("data:", "")
            try:
                return MimeType(mime_str)
            except ValueError:
                return MimeType.MP4  # Default
        elif source.lower().endswith(".mp4"):
            return MimeType.MP4
        elif source.lower().endswith(".webm"):
            return MimeType.WEBM
        else:
            return MimeType.MP4  # Default
