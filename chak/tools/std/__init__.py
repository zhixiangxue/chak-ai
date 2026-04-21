"""
chak.tools.std — chak standard built-in tools

All first-party atomic tools that ship with chak live here:

  - Bash:       execute shell commands (cross-platform)
  - Python:     execute Python code snippets via the active venv interpreter
  - FileSystem: atomic filesystem operations (read/write/edit/tree/list/delete)
  - Web:        fetch web pages (Firecrawl → Jina → httpx+readability fallback)
  - Search:     search the web (Tavily → Brave → DuckDuckGo fallback)
  - Http:       full HTTP client (GET/POST/PUT/PATCH/DELETE)
  - Pdf:        extract text/tables from PDF files (local path or URL)
"""

from .bash import Bash
from .python import Python
from .filesystem import FileSystem
from .web import Web
from .search import Search
from .http import Http
from .pdf import Pdf

__all__ = ["Bash", "Python", "FileSystem", "Web", "Search", "Http", "Pdf"]
