"""
FileSystem: Built-in filesystem tool for chak

Provides atomic operations on files and directories.
Register via NativeObjectTool to expose all methods as individual LLM tools:

    from chak.tools.std import FileSystem
    from chak.tools import NativeObjectTool

    fs = FileSystem(workdir="./workspace")
    conv = Conversation(model, tools=[NativeObjectTool(fs)])

Security:
    Pass workdir to restrict all operations to that directory tree.
    Without workdir, absolute and relative paths are both accepted.
"""

import difflib
import mimetypes
import re
import shutil
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Directories to skip automatically in list_dir / tree
# ---------------------------------------------------------------------------
_IGNORE_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".coverage", "htmlcov", "eggs", ".eggs",
    "*.egg-info",
})

_MAX_READ_CHARS    = 128_000
_DEFAULT_READ_LIMIT = 250
_DEFAULT_LIST_MAX   = 200

# Extensions treated as readable text (everything else is rejected as binary)
_TEXT_EXTENSIONS = frozenset({
    ".txt", ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".json",
    ".yaml", ".yml", ".html", ".css", ".scss", ".sh", ".bat",
    ".csv", ".log", ".xml", ".sql", ".env", ".ini", ".cfg",
    ".java", ".cpp", ".c", ".h", ".go", ".rs", ".php", ".rb",
    ".toml", ".lock", ".gitignore", ".dockerignore", ".editorconfig",
    ".r", ".jl", ".kt", ".swift", ".cs", ".vb", ".f90", ".m",
    "",   # extensionless files: Makefile, Dockerfile, etc.
})


# ---------------------------------------------------------------------------
# edit_file helpers
# ---------------------------------------------------------------------------

def _find_match(content: str, old_text: str) -> tuple:
    """Locate old_text in content.

    Tries exact match first, then a whitespace-tolerant line-level sliding
    window so minor indentation differences don't block the replacement.
    Both inputs must use LF line endings (caller normalises CRLF).

    Returns:
        (matched_fragment: str | None, count: int)
    """
    if old_text in content:
        return old_text, content.count(old_text)

    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0

    stripped_old   = [l.strip() for l in old_lines]
    content_lines  = content.splitlines()
    candidates: List[str] = []

    for i in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[i: i + len(stripped_old)]
        if [l.strip() for l in window] == stripped_old:
            candidates.append("\n".join(window))

    if candidates:
        return candidates[0], len(candidates)
    return None, 0


def _not_found_msg(old_text: str, content: str, path: str) -> str:
    """Return a helpful error with the closest difflib match."""
    lines      = content.splitlines(keepends=True)
    old_lines  = old_text.splitlines(keepends=True)
    window     = len(old_lines)

    best_ratio, best_start = 0.0, 0
    for i in range(max(1, len(lines) - window + 1)):
        ratio = difflib.SequenceMatcher(None, old_lines, lines[i: i + window]).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i

    if best_ratio > 0.5:
        diff = "\n".join(difflib.unified_diff(
            old_lines,
            lines[best_start: best_start + window],
            fromfile="old_text (provided)",
            tofile=f"{path} (actual, line {best_start + 1})",
            lineterm="",
        ))
        return (
            f"Error: old_text not found in {path}.\n"
            f"Best match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
        )
    return f"Error: old_text not found in {path}. No similar text found. Verify the file content."


# ---------------------------------------------------------------------------
# FileSystem
# ---------------------------------------------------------------------------

class FileSystem:
    """Atomic filesystem tool: read, write, edit, move, find, grep, tree, list, delete.

    Register with NativeObjectTool to expose all methods as individual
    LLM-callable tools:

        fs = FileSystem(workdir="./workspace")
        conv = Conversation(model, tools=[NativeObjectTool(fs)])

    All paths may be absolute or relative; relative paths resolve against
    workdir (or the process cwd when workdir is not set).
    """

    def __init__(
        self,
        workdir: Optional[str] = None,
        allowed_dirs: Optional[List[str]] = None,
        mode: str = "rw",
    ):
        """
        Args:
            workdir: Optional root directory.  All operations are restricted
                     to this tree when set.  Relative paths resolve against it.
            allowed_dirs: Additional directories outside workdir that are
                          permitted (e.g. a shared assets folder).
            mode: Tool visibility mode — ``"r"`` (read-only, only read methods
                  exposed to LLM), ``"rw"`` (read+write, all methods, default),
                  or ``"w"`` (write-only, only write methods exposed).
        """
        if mode not in ("r", "rw", "w"):
            raise ValueError(
                f"Invalid mode '{mode}': must be 'r', 'rw', or 'w'."
            )
        self._mode = mode
        self._workdir: Optional[Path] = Path(workdir).resolve() if workdir else None
        self._allowed: List[Path] = []
        if self._workdir:
            self._allowed.append(self._workdir)
        for d in (allowed_dirs or []):
            self._allowed.append(Path(d).resolve())
    
    def __available__(self) -> frozenset:
        """Return method names to expose as LLM tools based on current mode.

        This is part of the ``__available__`` protocol consumed by
        ``NativeObjectTool`` — objects may override it to declare which
        public methods should be registered as tools.
        """
        if self._mode == "r":
            return frozenset({"read_file", "tree", "list_dir", "find", "grep"})
        if self._mode == "w":
            return frozenset({"write_file", "create_file", "edit_file", "move", "delete_file"})
        return frozenset({
            "read_file", "write_file", "create_file", "edit_file",
            "tree", "list_dir", "move", "find", "grep", "delete_file",
        })

    # ------------------------------------------------------------------
    # Path resolution & security
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Resolve path and enforce workdir restriction (if set)."""
        p = Path(path).expanduser()
        if not p.is_absolute() and self._workdir:
            p = self._workdir / p
        resolved = p.resolve()
        if self._allowed:
            if not any(self._is_under(resolved, d) for d in self._allowed):
                raise PermissionError(
                    f"Path '{path}' is outside allowed directories."
                )
        return resolved

    @staticmethod
    def _is_under(path: Path, directory: Path) -> bool:
        try:
            path.relative_to(directory.resolve())
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    def read_file(
        self,
        path: str,
        offset: int = 1,
        limit: int = _DEFAULT_READ_LIMIT,
    ) -> str:
        """Read a text file and return its content with line numbers.

        Output format: "LINE_NUM→ line content" for every line returned.
        Use offset + limit to page through large files (max ~128K chars).

        Args:
            path:   File path (absolute or relative to workdir).
            offset: First line to return, 1-based (default 1).
            limit:  Maximum number of lines to return (default 250).

        Returns:
            Numbered file content, or an error string.
        """
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            if not fp.is_file():
                return f"Error: Not a file: {path}"

            raw = fp.read_bytes()
            if not raw:
                return f"(Empty file: {path})"

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return f"Error: Cannot read binary file: {path}"

            all_lines = text.splitlines()
            total     = len(all_lines)

            if offset < 1:
                offset = 1
            if offset > total:
                return f"Error: offset {offset} beyond end of file ({total} lines)"

            start    = offset - 1
            end      = min(start + limit, total)
            numbered = [
                f"{start + i + 1:6}→{line}"
                for i, line in enumerate(all_lines[start:end])
            ]
            result = "\n".join(numbered)

            # Hard cap at 128K chars
            if len(result) > _MAX_READ_CHARS:
                trimmed, chars = [], 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > _MAX_READ_CHARS:
                        break
                    trimmed.append(line)
                end    = start + len(trimmed)
                result = "\n".join(trimmed)

            if end < total:
                result += f"\n\n(Showing lines {offset}–{end} of {total}. Use offset={end + 1} to continue.)"
            else:
                result += f"\n\n(End of file — {total} lines total)"
            return result

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str = "") -> str:
        """Write a file, replacing any existing content entirely.

        Creates parent directories as needed. To modify only a portion
        of an existing file, use edit_file instead.

        Args:
            path:    File path (absolute or relative to workdir).
            content: Complete file content to write (UTF-8). Must be the
                     full text, not a fragment or diff. Defaults to "".

        Returns:
            Success message or error string.
        """
        try:
            fp = self._resolve(path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            if content:
                return f"Written {len(content)} chars to {fp}"
            return f"Written empty file {fp}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {e}"

    # ------------------------------------------------------------------
    # create_file
    # ------------------------------------------------------------------

    def create_file(self, path: str, content: str = "") -> str:
        """Create a new file. Fails if the file already exists.

        Creates parent directories as needed. Use write_file to overwrite
        an existing file.

        Args:
            path:    File path (absolute or relative to workdir).
            content: Complete file content to write (UTF-8). Must be the
                     full text of the new file. Defaults to "".

        Returns:
            Success message or error string.
        """
        try:
            fp = self._resolve(path)
            if fp.exists():
                return f"Error: File already exists: {path}. Use write_file to overwrite."
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            if content:
                return f"Created {fp} ({len(content)} chars)"
            return f"Created empty file {fp}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error creating file: {e}"

    # ------------------------------------------------------------------
    # edit_file
    # ------------------------------------------------------------------

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        """Edit a file by replacing old_text with new_text.

        Matching is whitespace-tolerant: minor indentation differences are
        accepted.  On failure, a diff of the closest matching block is shown.
        CRLF line endings in the original file are preserved after editing.

        Args:
            path:        File path (absolute or relative to workdir).
            old_text:    Text to find and replace. Must be unique unless
                         replace_all is True.
            new_text:    Replacement text.
            replace_all: Replace all occurrences (default False).

        Returns:
            Success message or detailed error string with diff hint.
        """
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"

            raw       = fp.read_bytes()
            uses_crlf = b"\r\n" in raw
            content   = raw.decode("utf-8").replace("\r\n", "\n")

            match, count = _find_match(content, old_text.replace("\r\n", "\n"))

            if match is None:
                return _not_found_msg(old_text, content, path)
            if count > 1 and not replace_all:
                return (
                    f"Warning: old_text appears {count} times. "
                    "Provide more context to make it unique, or set replace_all=true."
                )

            norm_new    = new_text.replace("\r\n", "\n")
            new_content = (
                content.replace(match, norm_new)
                if replace_all
                else content.replace(match, norm_new, 1)
            )
            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")

            fp.write_bytes(new_content.encode("utf-8"))
            return f"Successfully edited {fp}"

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {e}"

    # ------------------------------------------------------------------
    # tree
    # ------------------------------------------------------------------

    def tree(
        self,
        path: str,
        max_depth: int = 3,
        max_entries: int = 300,
    ) -> str:
        """Show directory structure as an ASCII tree.

        Noise directories (.git, node_modules, __pycache__, .venv, etc.)
        are automatically skipped.

        Args:
            path:        Root directory (absolute or relative to workdir).
            max_depth:   Maximum recursion depth (default 3).
            max_entries: Maximum total entries to show (default 300).

        Returns:
            ASCII tree, e.g.:
                workspace/
                ├── src/
                │   ├── main.py
                │   └── utils.py
                └── tests/
                    └── test_main.py
        """
        try:
            dp = self._resolve(path)
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            lines: list[str] = [dp.name + "/"]
            counter = [0]

            def _walk(directory: Path, prefix: str, depth: int) -> None:
                if depth > max_depth or counter[0] >= max_entries:
                    return
                entries = sorted(
                    [e for e in directory.iterdir() if e.name not in _IGNORE_DIRS],
                    key=lambda e: (e.is_file(), e.name.lower()),
                )
                for i, entry in enumerate(entries):
                    if counter[0] >= max_entries:
                        lines.append(prefix + "└── ... (truncated)")
                        break
                    is_last   = (i == len(entries) - 1)
                    connector = "└── " if is_last else "├── "
                    label     = entry.name + ("/" if entry.is_dir() else "")
                    lines.append(prefix + connector + label)
                    counter[0] += 1
                    if entry.is_dir():
                        extension = "    " if is_last else "│   "
                        _walk(entry, prefix + extension, depth + 1)

            _walk(dp, "", 1)
            return "\n".join(lines)

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error building tree: {e}"

    # ------------------------------------------------------------------
    # list_dir
    # ------------------------------------------------------------------

    def list_dir(
        self,
        path: str,
        recursive: bool = False,
        max_entries: int = _DEFAULT_LIST_MAX,
    ) -> str:
        """List directory contents.

        Noise directories (.git, node_modules, __pycache__, .venv, etc.)
        are automatically skipped.

        Args:
            path:        Directory path (absolute or relative to workdir).
            recursive:   Recursively list all nested entries (default False).
            max_entries: Maximum number of entries to return (default 200).

        Returns:
            Formatted directory listing or error string.
        """
        try:
            dp = self._resolve(path)
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            items: list = []
            total = 0

            if recursive:
                for item in sorted(dp.rglob("*")):
                    if any(p in _IGNORE_DIRS for p in item.parts):
                        continue
                    total += 1
                    if len(items) < max_entries:
                        rel = item.relative_to(dp)
                        items.append(
                            "  " + str(rel) + ("/" if item.is_dir() else "")
                        )
            else:
                for item in sorted(dp.iterdir()):
                    if item.name in _IGNORE_DIRS:
                        continue
                    total += 1
                    if len(items) < max_entries:
                        pfx = "[dir]  " if item.is_dir() else "[file] "
                        items.append(pfx + item.name)

            if total == 0:
                return f"(empty directory: {path})"

            result = "\n".join(items)
            if total > max_entries:
                result += f"\n\n(truncated: showing {max_entries} of {total} entries)"
            return result

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"

    # ------------------------------------------------------------------
    # move
    # ------------------------------------------------------------------

    def move(self, src: str, dst: str) -> str:
        """Move or rename a file or directory.

        Works across directories and drives. If dst is an existing directory,
        src is moved inside it. Otherwise src is renamed to dst.
        Use this for both ``mv`` and ``rename`` operations.

        Args:
            src: Source path (absolute or relative to workdir).
            dst: Destination path (absolute or relative to workdir).

        Returns:
            Success message or error string.
        """
        try:
            sp = self._resolve(src)
            dp = self._resolve(dst)
            if not sp.exists():
                return f"Error: Source not found: {src}"
            dp.parent.mkdir(parents=True, exist_ok=True)
            result = shutil.move(str(sp), str(dp))
            return f"Moved {sp} -> {result}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error moving: {e}"

    # ------------------------------------------------------------------
    # find
    # ------------------------------------------------------------------

    def find(
        self,
        path: str,
        pattern: str = "*",
        file_type: str = "any",
        max_results: int = 100,
    ) -> str:
        """Find files or directories matching a glob pattern.

        Searches recursively. Noise directories (.git, node_modules,
        __pycache__, .venv, etc.) are automatically skipped.
        Use this whenever you need to locate files by name or extension.

        Args:
            path:        Root directory to search from.
            pattern:     Glob pattern, e.g. ``"*.py"``, ``"test_*.py"``,
                         ``"**/*.json"`` (default ``"*"`` matches everything).
            file_type:   ``"file"``, ``"dir"``, or ``"any"`` (default ``"any"``).
            max_results: Maximum number of matches to return (default 100).

        Returns:
            Newline-separated list of matching relative paths, or error string.
        """
        try:
            dp = self._resolve(path)
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            matches: list[str] = []
            for p in sorted(dp.rglob(pattern)):
                if any(part in _IGNORE_DIRS for part in p.parts):
                    continue
                if file_type == "file" and not p.is_file():
                    continue
                if file_type == "dir" and not p.is_dir():
                    continue
                matches.append(str(p.relative_to(dp)))
                if len(matches) >= max_results:
                    break

            if not matches:
                return f"No matches for pattern '{pattern}' in {path}"
            result = "\n".join(matches)
            if len(matches) >= max_results:
                result += f"\n\n(showing first {max_results} matches — there may be more)"
            return result

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error finding files: {e}"

    # ------------------------------------------------------------------
    # grep
    # ------------------------------------------------------------------

    def grep(
        self,
        path: str,
        pattern: str,
        file_pattern: str = "*",
        case_sensitive: bool = True,
        max_results: int = 50,
    ) -> str:
        """Search for a regex or literal string pattern inside files.

        Scans text files recursively. Noise directories are skipped.
        Binary files are silently skipped. Use this to find where a symbol,
        function, or string is used across a codebase.

        Args:
            path:           Root directory to search in.
            pattern:        Regex or literal string to search for.
            file_pattern:   Glob pattern to filter which files to scan,
                            e.g. ``"*.py"``, ``"*.{js,ts}"`` (default ``"*"``).
            case_sensitive: Case-sensitive match (default True).
            max_results:    Maximum number of matching lines to return
                            (default 50).

        Returns:
            Matching lines in ``file:line_num: content`` format, or error string.
        """
        try:
            dp = self._resolve(path)
            if not dp.exists():
                return f"Error: Directory not found: {path}"
            if not dp.is_dir():
                return f"Error: Not a directory: {path}"

            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"

            results: list[str] = []
            for fp in sorted(dp.rglob(file_pattern)):
                if not fp.is_file():
                    continue
                if any(part in _IGNORE_DIRS for part in fp.parts):
                    continue
                if fp.suffix not in _TEXT_EXTENSIONS:
                    continue
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                rel = str(fp.relative_to(dp))
                for lineno, line in enumerate(text.splitlines(), 1):
                    if compiled.search(line):
                        results.append(f"{rel}:{lineno}: {line.rstrip()}")
                        if len(results) >= max_results:
                            break
                if len(results) >= max_results:
                    break

            if not results:
                return f"No matches for '{pattern}' in {path}"
            result = "\n".join(results)
            if len(results) >= max_results:
                result += f"\n\n(showing first {max_results} matches — there may be more)"
            return result

        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error searching files: {e}"

    # ------------------------------------------------------------------
    # delete_file
    # ------------------------------------------------------------------

    def delete_file(self, path: str) -> str:
        """Delete a file.

        Args:
            path: File path (absolute or relative to workdir).

        Returns:
            Success message or error string.
        """
        try:
            fp = self._resolve(path)
            if not fp.exists():
                return f"Error: File not found: {path}"
            if fp.is_dir():
                return f"Error: Path is a directory, not a file: {path}"
            fp.unlink()
            return f"Deleted {fp}"
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error deleting file: {e}"

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        wd = str(self._workdir) if self._workdir else "unrestricted"
        return f"<FileSystem workdir={wd} mode={self._mode}>"

