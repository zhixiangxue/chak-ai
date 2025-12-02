"""
Audio attachment type.
"""
from typing import Optional
from .base import Attachment, MimeType


class Audio(Attachment):
    """Audio attachment for audio-enabled models"""
    
    def __init__(self, source: str, mime_type: Optional[MimeType] = None):
        """
        Create an audio attachment.
        
        Args:
            source: Audio URL or base64 data URI
            mime_type: Audio MIME type (auto-detected from URL extension if not specified)
        
        Examples:
            >>> Audio("https://example.com/voice.mp3")
            >>> Audio("https://example.com/sound.wav", MimeType.WAV)
            >>> Audio("data:audio/wav;base64,UklGR...")
        """
        if mime_type is None:
            mime_type = self._auto_detect(source)
        super().__init__(source, mime_type)
        self._validate_mime_type("audio/")
    
    @staticmethod
    def _auto_detect(source: str) -> MimeType:
        """Auto-detect audio MIME type from URL or data URI"""
        if source.startswith("data:audio/"):
            mime_str = source.split(";")[0].replace("data:", "")
            try:
                return MimeType(mime_str)
            except ValueError:
                return MimeType.WAV  # Default
        elif source.lower().endswith(".wav"):
            return MimeType.WAV
        elif source.lower().endswith(".mp3"):
            return MimeType.MP3
        elif source.lower().endswith(".ogg"):
            return MimeType.OGG
        else:
            return MimeType.WAV  # Default
