"""Scratchpad: disk-backed working memory for LLM agents.

A scratchpad stores concise, section-based notes in one JSON file. It is meant
for task/session scoped working memory: agents can offload distilled findings
from the chat context and recall only the specific section they need later.

Design principle: each section is a ``(heading, content)`` key-value pair. The
LLM should store short summaries, conclusions, and key quotes — not raw document
or tool-output dumps. A good section is usually 5-30 lines.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

_SECTION_LINE_SOFT_LIMIT = 30
_DEFAULT_SEARCH_LIMIT = 10
_SNIPPET_CHARS = 240


class Scratchpad:
    """Agent working memory backed by a JSON file.

    Args:
        path: JSON file path used to store scratchpad sections.
        mode: ``"r"`` exposes read/search methods only; ``"rw"`` exposes full
            CRUD methods. Defaults to ``"rw"``.
    """

    def __init__(self, path: str | Path, mode: str = "rw"):
        if mode not in ("r", "rw"):
            raise ValueError(f"Invalid mode '{mode}': must be 'r' or 'rw'.")

        self._file = Path(path).expanduser().resolve()
        if self._file.exists() and self._file.is_dir():
            raise ValueError(f"Scratchpad path must be a file, got directory: {self._file}")

        self._mode = mode
        # Protect read-modify-write operations when providers dispatch multiple
        # tool calls concurrently in the same round.
        self._lock = threading.Lock()

    def __available__(self) -> frozenset[str]:
        """Return method names to expose as LLM tools based on current mode."""
        read_methods = {"list_sections", "read_section", "search_sections"}
        if self._mode == "r":
            return frozenset(read_methods)
        return frozenset(read_methods | {"save_section", "remove_section", "clear"})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, str]:
        """Load all sections as an ordered ``{heading: content}`` dict."""
        if not self._file.exists():
            return {}
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, sections: dict[str, str]) -> None:
        """Persist all sections to disk atomically."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._file.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(
                json.dumps(sections, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._file)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _format_toc(sections: dict[str, str]) -> str:
        """Format a table-of-contents string from a sections dict."""
        if not sections:
            return "(empty — no notes saved yet)"
        lines: list[str] = []
        for idx, (heading, body) in enumerate(sections.items(), 1):
            n_lines = len(body.splitlines()) if body else 0
            lines.append(f'{idx}. "{heading}" ({n_lines} lines)')
        return "\n".join(lines)

    @staticmethod
    def _snippet(text: str, needle: str) -> str:
        """Return a compact snippet around the first match."""
        lower_text = text.lower()
        lower_needle = needle.lower()
        pos = lower_text.find(lower_needle)
        if pos < 0:
            return text[:_SNIPPET_CHARS].strip()
        start = max(0, pos - _SNIPPET_CHARS // 3)
        end = min(len(text), pos + len(needle) + _SNIPPET_CHARS // 2)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        return snippet

    def _ensure_writable(self) -> str | None:
        """Return an error message when the scratchpad is read-only."""
        if self._mode == "r":
            return "Error: scratchpad is read-only."
        return None

    # ------------------------------------------------------------------
    # Public API — each method is exposed as an LLM tool
    # ------------------------------------------------------------------

    def list_sections(self) -> str:
        """List all section headings with line counts.

        Use this lightweight table of contents to orient yourself before
        deciding which specific section to read.
        """
        return self._format_toc(self._load())

    def read_section(self, heading: str) -> str:
        """Read the content of one specific section by heading.

        Args:
            heading: The exact section title shown by ``list_sections``.
        """
        heading = heading.strip()
        if not heading:
            return "Error: heading must not be empty."

        sections = self._load()
        if heading in sections:
            body = sections[heading]
            return body if body else "(section exists but is empty)"

        available = list(sections.keys())
        if available:
            return f"Error: section '{heading}' not found. Available sections: {available}"
        return f"Error: section '{heading}' not found (scratchpad is empty)."

    def search_sections(self, query: str, max_results: int = _DEFAULT_SEARCH_LIMIT) -> str:
        """Search section headings and content for a keyword or phrase.

        Use this before reading sections when you remember a term but not the
        exact heading. Search returns compact snippets, not full section bodies.

        Args:
            query: Keyword or phrase to search for.
            max_results: Maximum matching sections to return.
        """
        query = query.strip()
        if not query:
            return "Error: query must not be empty."
        if max_results < 1:
            return "Error: max_results must be at least 1."

        sections = self._load()
        matches: list[str] = []
        needle = query.lower()
        for heading, body in sections.items():
            haystack = f"{heading}\n{body}".lower()
            if needle not in haystack:
                continue
            snippet = self._snippet(body, query)
            matches.append(f'- "{heading}": {snippet}')
            if len(matches) >= max_results:
                break

        if not matches:
            return f"No sections matched query: {query}"
        return "Matching sections:\n" + "\n".join(matches)

    def save_section(self, heading: str, content: str) -> str:
        """Create or overwrite a section with concise findings.

        Store conclusions, key facts, and short quotes. Avoid raw document text
        or bulky tool outputs. If the heading already exists, this replaces the
        entire section.

        Args:
            heading: Descriptive section title, preferably snake_case.
            content: Concise note content.
        """
        error = self._ensure_writable()
        if error:
            return error

        heading = heading.strip()
        if not heading:
            return "Error: heading must not be empty."
        content = content.strip()
        if not content:
            return "Error: content must not be empty."

        with self._lock:
            sections = self._load()
            replaced = heading in sections
            sections[heading] = content
            self._save(sections)

        n_lines = len(content.splitlines())
        action = "Updated" if replaced else "Saved new"
        msg = f"{action} section '{heading}' ({n_lines} lines)."
        if n_lines > _SECTION_LINE_SOFT_LIMIT:
            msg += (
                f" ⚠️ Warning: section is {n_lines} lines "
                f"(recommended ≤{_SECTION_LINE_SOFT_LIMIT}). "
                "Consider splitting into multiple sections or summarizing further."
            )
        msg += f"\n\nCurrent sections:\n{self._format_toc(sections)}"
        return msg

    def remove_section(self, heading: str) -> str:
        """Delete a section that is no longer needed.

        Args:
            heading: The exact section title to remove.
        """
        error = self._ensure_writable()
        if error:
            return error

        heading = heading.strip()
        if not heading:
            return "Error: heading must not be empty."

        with self._lock:
            sections = self._load()
            if heading not in sections:
                available = list(sections.keys())
                if available:
                    return f"Error: section '{heading}' not found. Available: {available}"
                return f"Error: section '{heading}' not found (scratchpad is empty)."

            del sections[heading]
            self._save(sections)

        return f"Removed section '{heading}'.\n\nCurrent sections:\n{self._format_toc(sections)}"

    def clear(self) -> str:
        """Remove all sections at once."""
        error = self._ensure_writable()
        if error:
            return error

        with self._lock:
            sections = self._load()
            if not sections:
                return "Scratchpad is already empty."
            count = len(sections)
            self._save({})

        return f"Cleared all {count} section(s). Scratchpad is now empty."
