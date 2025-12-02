"""
Multimodal attachment types for chak.

This module provides type-safe attachment classes for sending images, audio,
video, and documents in conversations.
"""

# Base classes and utilities
from .base import MimeType, Attachment, detect_content_type, AttachmentContent, ContentType

# Media attachments
from .image import Image
from .audio import Audio
from .video import Video

# Document attachments
from .pdf import PDF, default_pdf_reader, default_pdf_reader_async
from .doc import DOC, default_docx_reader, default_docx_reader_async
from .excel import Excel, default_xlsx_reader, default_xlsx_reader_async
from .csv import CSV, default_csv_reader, default_csv_reader_async
from .txt import TXT, default_txt_reader, default_txt_reader_async
from .link import Link

__all__ = [
    # Base
    'MimeType',
    'Attachment',
    'detect_content_type',
    'AttachmentContent',
    'ContentType',
    
    # Media
    'Image',
    'Audio',
    'Video',
    
    # Documents
    'PDF',
    'DOC',
    'Excel',
    'CSV',
    'TXT',
    'Link',
    
    # Document readers (sync)
    'default_pdf_reader',
    'default_docx_reader',
    'default_xlsx_reader',
    'default_csv_reader',
    'default_txt_reader',
    
    # Document readers (async)
    'default_pdf_reader_async',
    'default_docx_reader_async',
    'default_xlsx_reader_async',
    'default_csv_reader_async',
    'default_txt_reader_async',
]
