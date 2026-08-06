"""
chak.tools.std — chak standard built-in tools

All first-party atomic tools that ship with chak live here:

  - Bash:       execute shell commands (cross-platform)
  - Python:     execute Python code snippets via the active venv interpreter
  - FileSystem: atomic filesystem operations (read/write/edit/move/find/grep/tree/list/delete)
  - Web:        fetch web pages (Firecrawl → Jina → httpx+readability fallback)
  - Search:     search the web (Tavily → Brave → DuckDuckGo fallback)
  - Http:       full HTTP client (GET/POST/PUT/PATCH/DELETE)
  - Pdf:        extract text/tables from PDF files and fill PDF forms (local path or URL)
  - Sandbox:    execute multi-file code projects in an isolated e2b cloud sandbox
  - SQL:        query and modify SQLite / PostgreSQL / MySQL databases
  - Excel:      read and write .xlsx and .csv spreadsheets
  - Scratchpad: disk-backed working memory for agents (section CRUD + search)
  - Notebook:   persistent, searchable notebook for agents (note + recall)
"""

from .bash import Bash
from .python import Python
from .filesystem import FileSystem
from .web import Web
from .search import Search
from .http import Http
from .pdf import Pdf
from .sandbox import Sandbox
from .sql import SQL
from .excel import Excel
from .scratchpad import Scratchpad
# TODO: re-enable after seeka 0.2.1+ is installable (currently blocked by zvec)
# from .notebook import Notebook, NotebookBackend

__all__ = ["Bash", "Python", "FileSystem", "Web", "Search", "Http", "Pdf", "Sandbox", "SQL", "Excel", "Scratchpad"]  # Notebook, NotebookBackend — TODO: re-enable
