"""
Pdf: Built-in PDF reader & form filler tool for chak

Extracts text, tables, and structured content from PDF files (local or remote).
Uses pymupdf4llm for high-accuracy local extraction on ordinary layouts.
Also fills AcroForm PDF forms (e.g. the URLA 1003 loan application) through a
three-step, LLM-friendly workflow:

    metadata  — reports is_fillable_form plus field counts and pages
    schema    — maps every field name to its meaning, type, options, value
    fill      — writes a {field_name: value} mapping into a new editable PDF,
                so the form can be completed incrementally over multiple calls

Complex tables (merged cells, vertically rotated axis labels, footnote
exceptions) cannot be faithfully flattened by a linear text extractor. This
tool detects such pages automatically and, when a vision model is configured,
transparently re-reads the page from a rendered image so the returned tables
keep their row/column associations and footnote anchors. The caller never
needs to know which pages are complex.

Vision model recommendation for complex tables:

    Model                          Complex tables    Notes
    -----------------------------  ---------------   ----------------------------
    anthropic/claude-sonnet-4-6    Recommended       Most reliable table structure
    minimax/MiniMax-M3             Recommended       Strong; faster and cheaper
    qwen-vl-max                    Not recommended   Values ok, structure slips

Keep vision_dpi at 300 (the tool default); lower resolution degrades every
model's structural accuracy.

Usage:
    from chak.tools.std import Pdf
    pdf = Pdf()                                        # plain-text extraction only
    pdf = Pdf(vision="anthropic/claude-sonnet-4-6")    # recommended: frontier vision model
    conv = Conversation(model, tools=[pdf])

Supported output formats:
    markdown  — Markdown with tables and headings (default, best for LLM input)
    txt       — plain text, no syntax
    html      — styled HTML document
    json      — per-page JSON array

Dependencies:
    pymupdf4llm  — pip install pymupdf4llm
    markdown     — pip install markdown  (only needed for html format)
    PyPDFForm    — pip install PyPDFForm (only needed for form filling)
"""

import base64
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

_PDF_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; chak-pdf/1.0)",
    "Accept": "application/pdf,application/octet-stream,*/*",
}
_PDF_DOWNLOAD_TIMEOUT = 60

# --- Complex-table detection & vision fallback tuning -------------------------
# A text line counts as "rotated" (a vertical axis label — the classic structure
# that linear extraction shreds) when the vertical component of its writing
# direction exceeds this threshold. Horizontal text has dir ~= (1, 0).
_ROTATED_DIR_THRESHOLD = 0.3
# Ignore 1-2 char rotated fragments (stamps, glyph noise); genuine axis labels
# are longer, so this keeps detection high-precision.
_ROTATED_MIN_CHARS = 3
# A table is "heavily merged" when at least this fraction of its grid cells are
# absorbed into spans. Kept conservative so ordinary header merges alone do not
# route a page through the (slower, paid) vision model.
_MERGED_CELL_RATIO = 0.25
# DPI used when rendering a page to an image for the vision model / render_page.
# 200 is the hard floor for correct merged-cell spans (150 reproducibly broke a
# rowspan in testing); 300 is the reliable default for the deeply nested tables
# that trigger the vision path, at negligible extra cost (tiled models are
# fixed-token; glm-class is near its per-image token cap by 300 anyway).
_DEFAULT_VISION_DPI = 300

# Sentinel distinguishing "vision provider not yet resolved" from "resolved to
# None" (unconfigured or construction failed), so we resolve at most once.
_UNSET = object()

# --- Form (AcroForm) support ---------------------------------------------------
# Map PyMuPDF widget type strings to the compact type vocabulary exposed to the
# LLM. Push buttons are deliberately absent: they hold no fillable value.
_WIDGET_TYPE_MAP = {
    "Text": "text",
    "CheckBox": "checkbox",
    "RadioButton": "radio",
    "ComboBox": "dropdown",
    "ListBox": "dropdown",
    "Signature": "signature",
}
# LLMs routinely send checkbox values as strings; accept the common spellings
# instead of failing the field.
_CHECKBOX_TRUE = {"true", "yes", "on", "1", "x", "checked"}
_CHECKBOX_FALSE = {"false", "no", "off", "0", "unchecked"}
# Cap for the printed-text hint extracted around an unlabeled widget.
_NEARBY_TEXT_MAX_CHARS = 80
# A tooltip shared by this many fields no longer discriminates between them
# (real-world forms ship copy-paste TU errors, e.g. the 2021 URLA labels all
# 16 Section 5 declaration radios "Asset 2 Deposited/Not Deposited"), so a
# nearby-text hint is appended alongside such labels.
_AMBIGUOUS_LABEL_MIN_COUNT = 4

_VISION_SYSTEM_PROMPT = (
    "You are a precise table data extractor. You are shown an image of a single "
    "PDF page. Your job is to capture every value and its full row/column context "
    "as clean, fully rectangular tables optimized for data consumption — not to "
    "reproduce the original visual layout."
)

_VISION_USER_PROMPT = (
    "Extract page {page} from the image. First decide what each block IS, then "
    "choose its shape:\n"
    "\n"
    "SEGMENT BY MEANING, NOT BY GRID LINES. One visual grid is NOT necessarily one "
    "table: PDF authors often cram a small rule box or a differently-structured "
    "block into the same borders as a big matrix. Whenever a group of rows or "
    "columns has DIFFERENT axes or a different meaning from its neighbors (e.g. a "
    "little 'Max LTV / Min DSCR / Max Loan Amount' box tacked onto the bottom or "
    "side of a pricing matrix), emit it as its OWN separate table with its own "
    "header instead of forcing it into the neighboring grid. It is better to output "
    "several small correct tables than one big table with wrong semantics.\n"
    "\n"
    "(A) DATA MATRIX — a grid of values indexed by row and column (e.g. FICO × LTV). "
    "Render it as a GitHub-flavored Markdown pipe table where every row has the "
    "exact same number of columns as the header row.\n"
    "  - Do NOT use merged cells. Wherever the original merges a cell, uses a "
    "vertical/rotated axis label, or a value that heads a group of rows/columns, "
    "REPEAT that exact value in every row/column it actually covers so each row is "
    "self-contained. This repetition applies ONLY to genuinely merged/spanning "
    "labels.\n"
    "  - Flatten multi-level headers into ONE header row: prefix EVERY sub-column "
    "with its FULL group-header path joined by ' / ' (e.g. a 'DSCR >= 1.00' band "
    "sitting over a 'Purchase' column becomes the single header "
    "'DSCR >= 1.00 / Purchase'). Never drop a group header, and never turn a group "
    "header into its own separate data column.\n"
    "  - For a cell that is genuinely empty or not applicable, write its printed "
    "value (e.g. 'n/a') or leave it blank. NEVER invent or duplicate a value just "
    "to fill a column. If making a row rectangular would force you to copy ONE "
    "value across columns it does not actually apply to (e.g. a single "
    "'Max LTV: 70%' becoming 70% under Purchase AND R&T AND Cash-Out), STOP — that "
    "is a sign the block is really a key-value block (B); split it out instead of "
    "duplicating.\n"
    "\n"
    "(B) KEY-VALUE / REQUIREMENTS block — a list of 'label: description' entries "
    "(e.g. a 'General Requirements' section, or a small 'Max LTV / Min DSCR / Max "
    "Loan Amount' rule box). Render it as a simple TWO-column table "
    "'| Field | Details |': the label in column 1, its full content in column 2 "
    "(use <br> for line breaks inside the cell). Do NOT widen it into many columns "
    "and do NOT pad or repeat values to make it look like a matrix.\n"
    "\n"
    "- Keep footnote markers (*, **, superscripts) inline in the cell they annotate, "
    "then list every footnote and its meaning as a plain list under the table.\n"
    "- Transcribe non-table prose as plain markdown text in reading order.\n"
    "- Copy every printed value exactly; never invent, merge, or drop data.\n"
    "- Output only the tables and text. No commentary, no code fences."
)


_HTML_CSS = """
  body  { font-family: Arial, sans-serif; margin: 0.75in; font-size: 9pt; color: #222; }
  h1   { color: #1F3864; font-size: 14pt; margin-bottom: 1em; }
  h2   { color: #1F3864; font-size: 12pt; font-weight: bold; margin: 1.2em 0 0.4em 0; }
  h3   { color: #1F3864; font-size: 10.5pt; font-weight: bold; margin: 1em 0 0.3em 0; }
  h4, h5, h6 { color: #333; font-size: 9.5pt; font-weight: bold; margin: 0.8em 0 0.2em 0; }
  p    { margin: 0.15em 0 0.4em 0; line-height: 1.6; }
  ul, ol { margin: 0.2em 0 0.4em 1.4em; padding: 0; }
  li   { margin: 0.1em 0; line-height: 1.5; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1em; font-size: 9pt; }
  th   { background: #1F3864; color: white; font-weight: bold;
         padding: 4px 8px; border: 1px solid #ccc; text-align: left; }
  td   { padding: 4px 8px; border: 1px solid #ccc; }
  tr:nth-child(even) td { background: #f5f5f5; }
  hr   { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
"""


def _require_pdf_libs():
    """Lazily import and return (pymupdf, pymupdf4llm).

    IMPORTANT: pymupdf.layout must NEVER be imported in this process.
    Its module-level activate() monkey-patches pymupdf._get_layout with an
    ONNX model predictor.  This irreversibly changes PyMuPDF's internal
    behaviour and causes pymupdf4llm 0.x to raise
    ``ValueError: min() iterable argument is empty`` on pages with complex
    tables, silently falling back to plain-text extraction.
    pymupdf4llm already provides its own layout analysis; the external
    pymupdf.layout package is neither needed nor safe to use here.
    """
    try:
        import pymupdf  # noqa: PLC0415
    except ImportError:
        raise ImportError("PyMuPDF is required. Run: pip install PyMuPDF")

    try:
        import pymupdf4llm  # noqa: PLC0415
    except ImportError:
        raise ImportError("pymupdf4llm is required. Run: pip install pymupdf4llm")

    return pymupdf, pymupdf4llm


def _is_certificate_verify_error(error: Exception) -> bool:
    message = str(error).lower()
    return "certificate_verify_failed" in message or "certificate verify failed" in message


def _md_to_plain(md: str) -> str:
    """Strip Markdown syntax to produce readable plain text."""
    txt = re.sub(r"\|[^\n]+", "", md)            # remove table rows
    txt = re.sub(r"^#+\s+", "", txt, flags=re.M)  # headings
    txt = re.sub(r"\*\*([^*]+)\*\*", r"\1", txt)  # bold
    txt = re.sub(r"\*([^*]+)\*", r"\1", txt)      # italic
    txt = re.sub(r"^[-*]\s+", "", txt, flags=re.M) # bullets
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _md_to_html(md: str, title: str = "PDF") -> str:
    """Convert Markdown to a complete styled HTML document."""
    try:
        import markdown as md_lib  # noqa: PLC0415
    except ImportError:
        raise ImportError("markdown is required for html format. Run: pip install markdown")
    body = md_lib.markdown(md, extensions=["tables", "fenced_code", "nl2br"])
    return (
        f"<!DOCTYPE html>\n<html>\n<head>\n"
        f'<meta charset="utf-8">\n<title>{title}</title>\n'
        f"<style>\n{_HTML_CSS}\n</style>\n</head>\n<body>\n"
        f"<h1>{title}</h1>\n{body}\n</body>\n</html>"
    )


def _resolve_pdf(source: str) -> str:
    """Return a local file path to the PDF, downloading if source is a URL."""
    if source.startswith(("http://", "https://")):
        try:
            import httpx  # noqa: PLC0415
        except ImportError:
            raise ImportError("httpx is required for URL sources. Run: pip install httpx")

        try:
            resp = httpx.get(
                source,
                follow_redirects=True,
                timeout=_PDF_DOWNLOAD_TIMEOUT,
                headers=_PDF_DOWNLOAD_HEADERS,
            )
        except httpx.HTTPError as httpx_error:
            try:
                import requests  # noqa: PLC0415
            except ImportError:
                if not _is_certificate_verify_error(httpx_error):
                    raise httpx_error
                resp = httpx.get(
                    source,
                    follow_redirects=True,
                    timeout=_PDF_DOWNLOAD_TIMEOUT,
                    headers=_PDF_DOWNLOAD_HEADERS,
                    verify=False,
                )
            else:
                try:
                    resp = requests.get(
                        source,
                        allow_redirects=True,
                        timeout=_PDF_DOWNLOAD_TIMEOUT,
                        headers=_PDF_DOWNLOAD_HEADERS,
                    )
                except requests.exceptions.SSLError as requests_ssl_error:
                    if not _is_certificate_verify_error(requests_ssl_error):
                        raise
                    resp = requests.get(
                        source,
                        allow_redirects=True,
                        timeout=_PDF_DOWNLOAD_TIMEOUT,
                        headers=_PDF_DOWNLOAD_HEADERS,
                        verify=False,
                    )
        resp.raise_for_status()

        # Validate Content-Type
        ct = resp.headers.get("content-type", "")
        if "pdf" not in ct and not source.lower().endswith(".pdf"):
            raise ValueError(
                f"Source does not appear to be a PDF (Content-Type: {ct}). "
                "Only PDF files are supported."
            )

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name

    # Local file
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"File '{source}' does not have a .pdf extension. "
            "Only PDF files are supported."
        )
    return str(path)


def _json(data: dict[str, Any] | list[Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)



def _content(text: str, max_chars: int | None) -> str:
    if max_chars is None:
        return text
    return text[:max_chars]


def _file_size_mb(size_bytes: int) -> float:
    return round(size_bytes / (1024 * 1024), 2)


def _text_density(average_chars_per_page: int) -> str:
    if average_chars_per_page <= 0:
        return "none"
    if average_chars_per_page < 500:
        return "low"
    if average_chars_per_page < 2000:
        return "medium"
    return "high"


def _size_category(file_size_mb: float, page_count: int, estimated_total_chars: int) -> str:
    if file_size_mb >= 100 or page_count >= 500 or estimated_total_chars >= 1_000_000:
        return "huge"
    if file_size_mb >= 25 or page_count >= 100 or estimated_total_chars >= 250_000:
        return "large"
    if file_size_mb >= 5 or page_count >= 25 or estimated_total_chars >= 50_000:
        return "medium"
    return "small"


def _likely_scanned_or_image_heavy(file_size_mb: float, page_count: int, average_chars_per_page: int) -> bool:
    if page_count <= 0:
        return False
    return file_size_mb >= 2 and average_chars_per_page < 200


def _page_numbers(start_page: int, end_page: int, page_count: int) -> list[int]:
    if start_page < 1:
        raise ValueError("start_page must be >= 1")
    if end_page < start_page:
        raise ValueError("end_page must be >= start_page")
    if end_page > page_count:
        raise ValueError(f"end_page must be <= total pages ({page_count})")
    return list(range(start_page - 1, end_page))


def _chunk_page(chunk: dict[str, Any], fallback: int) -> int:
    metadata = chunk.get("metadata") or {}
    return metadata.get("page") or metadata.get("page_number") or fallback


def _format_output(markdown_text: str, format: str, title: str = "PDF") -> str:
    if format == "markdown":
        return markdown_text
    if format == "txt":
        return _md_to_plain(markdown_text)
    if format == "html":
        return _md_to_html(markdown_text, title)
    raise ValueError("format must be one of: markdown, txt, html")


def _headings_from_markdown_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        page = _chunk_page(chunk, index + 1)
        for match in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, flags=re.M):
            headings.append(
                {
                    "level": len(match.group(1)),
                    "title": match.group(2).strip(),
                    "page": page,
                }
            )
    return headings


def _looks_like_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 120:
        return False
    if text.endswith((".", ",", ";", ":")):
        return False
    if re.match(r"^(chapter|section|part|appendix)\b", text, flags=re.I):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+\S", text):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if len(words) < 2:
        return False
    title_words = sum(1 for word in words if word[:1].isupper())
    return title_words / len(words) >= 0.6


def _headings_from_plain_text(doc: Any) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for page_index in range(doc.page_count):
        text = doc.load_page(page_index).get_text("text")
        for line in text.splitlines():
            title = line.strip()
            key = (page_index + 1, title)
            if key in seen or not _looks_like_heading(title):
                continue
            seen.add(key)
            headings.append({"level": 1, "title": title, "page": page_index + 1})
    return headings


def _fallback_markdown(doc: Any, pages: list[int]) -> str:
    """Extract text from pages using PyMuPDF native get_text.

    Fallback when pymupdf4llm raises ValueError on pages with empty or
    complex table structures (min() iterable argument is empty).
    """
    parts = []
    for page_index in pages:
        text = doc.load_page(page_index).get_text("text")
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _fallback_chunks(doc: Any, pages: list[int]) -> list[dict[str, Any]]:
    """Extract per-page chunks using PyMuPDF native get_text.

    Fallback when pymupdf4llm raises ValueError on pages with empty or
    complex table structures (min() iterable argument is empty).
    """
    chunks = []
    for page_index in pages:
        text = doc.load_page(page_index).get_text("text")
        chunks.append(
            {
                "text": text,
                "metadata": {"page": page_index + 1},
            }
        )
    return chunks


def _count_rotated_lines(page: Any) -> int:
    """Count text lines with a non-horizontal writing direction on ``page``.

    Vertically rotated labels (e.g. an occupancy axis printed sideways across a
    matrix) are the single strongest fingerprint of a table whose 2D structure
    linear extraction cannot recover. Short fragments are ignored to avoid
    firing on stray glyphs.
    """
    try:
        info = page.get_text("dict")
    except Exception:
        return 0
    count = 0
    for block in info.get("blocks", []):
        for line in block.get("lines", []):
            direction = line.get("dir", (1, 0))
            if len(direction) != 2 or abs(direction[1]) <= _ROTATED_DIR_THRESHOLD:
                continue
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if len(text.strip()) >= _ROTATED_MIN_CHARS:
                count += 1
    return count


def _has_heavily_merged_table(page: Any) -> bool:
    """Return True if any detected table has a high fraction of merged cells.

    PyMuPDF exposes one rectangle per physical cell, so a merged span shows up
    as fewer cells than ``row_count * col_count``. A large merge ratio means the
    grid is genuinely two-dimensional (spanning headers), which markdown's flat
    pipe tables cannot represent faithfully.
    """
    try:
        tables = page.find_tables().tables
    except Exception:
        return False
    for table in tables:
        try:
            expected = (table.row_count or 0) * (table.col_count or 0)
            actual = len(table.cells)
        except Exception:
            continue
        if expected and actual < expected:
            if (expected - actual) / expected >= _MERGED_CELL_RATIO:
                return True
    return False


def _has_tables(page: Any) -> bool:
    try:
        return bool(page.find_tables().tables)
    except Exception:
        return False


def _complex_table_reasons(page: Any) -> list[str]:
    """Diagnose why a page's tables are unsafe for linear extraction.

    Returns a list of human-readable reasons (empty when the page is safe). A
    page only qualifies when it actually contains a detected table AND exhibits
    a structure — rotated axis labels or heavy cell merging — that flattening
    would corrupt. This keeps ordinary tables on the fast plain-text path.
    """
    if not _has_tables(page):
        return []
    reasons: list[str] = []
    if _count_rotated_lines(page) > 0:
        reasons.append("rotated axis labels")
    if _has_heavily_merged_table(page):
        reasons.append("merged/spanning cells")
    return reasons


def _render_page_png(page: Any, dpi: int) -> bytes:
    """Render a single page to PNG bytes at the requested DPI."""
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.tobytes("png")


def _strip_code_fences(text: str) -> str:
    """Strip any wrapping ```lang ... ``` fence(s) the model may have added.

    Vision models frequently wrap the HTML table in a markdown code fence
    (e.g. ```` ```markdown ... ``` ````), and some (observed with glm-4.5v) emit
    doubled/nested closing fences. Peeling only a single layer left a stray
    ``` behind, so we drop fence lines from both ends until none remain. Our
    expected payload is an HTML table, which never legitimately starts or ends
    with a ``` line, so edge-only stripping is safe.
    """
    lines = text.strip().splitlines()
    while lines and lines[0].strip().startswith("```"):
        lines.pop(0)
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def _wrap_complex_vision(page_no: int, vision_text: str, reasons: list[str]) -> str:
    """Prefix a vision-reconstructed page with a short provenance note."""
    note = (
        f"[PDF page {page_no}: this page contains a complex table "
        f"({', '.join(reasons)}). The content below was reconstructed from a "
        f"rendered image with a vision model to preserve column/row structure and "
        f"footnote exceptions. Trust this reconstruction over any flat text layout.]"
    )
    return f"{note}\n\n{vision_text}"


def _wrap_complex_warning(
    page_no: int, base_md: str, reasons: list[str], vision_configured: bool
) -> str:
    """Prefix a not-reconstructed complex page with an explicit reliability warning."""
    if vision_configured:
        hint = "the configured vision model was unavailable; retry or render_page and read the image manually"
    else:
        hint = (
            "render this page with render_page and read the image using a "
            "vision-capable model, or configure a vision model on the Pdf tool"
        )
    warning = (
        f"> WARNING — COMPLEX TABLE ON PDF PAGE {page_no}: contains "
        f"{', '.join(reasons)} that linear text extraction cannot represent "
        f"faithfully. Column/row associations and footnote exceptions in the text "
        f"below may be WRONG. To read it accurately, {hint}."
    )
    return f"{warning}\n\n{base_md}"


def _summarize_methods(per_page: list[dict[str, Any]]) -> str:
    """Join the distinct per-page extraction methods (e.g. ``layout+vision``)."""
    methods = sorted({record["method"] for record in per_page})
    return "+".join(methods) if methods else "none"


# --- Form (AcroForm) helpers ---------------------------------------------------


def _require_pypdfform():
    """Lazily import and return PyPDFForm's PdfWrapper."""
    try:
        from PyPDFForm import PdfWrapper  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "PyPDFForm is required for PDF form filling. Run: pip install PyPDFForm"
        )
    return PdfWrapper


def _has_xfa(doc: Any) -> bool:
    """Detect an XFA form. PyPDFForm (and this tool) can only fill AcroForm."""
    try:
        key_type, _ = doc.xref_get_key(doc.pdf_catalog(), "AcroForm/XFA")
        return key_type not in (None, "null")
    except Exception:
        return False


def _decode_pdf_name(value: str) -> str:
    """Decode ``#XX`` hex escapes in a PDF name (e.g. ``U.S.#20Citizen``).

    Radio export values are PDF name objects, whose special characters are
    hex-escaped. Show the LLM the human-readable form; filling goes through
    zero-based indices anyway, so the raw spelling is never needed.
    """
    if "#" not in value:
        return value
    try:
        return re.sub(
            r"#([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), value
        )
    except Exception:
        return value


def _widget_on_state(widget: Any) -> str | None:
    """Return the on-state (export value) of a checkbox/radio kid widget."""
    try:
        states = widget.button_states() or {}
    except Exception:
        return None
    for key in ("normal", "down"):
        for state in states.get(key) or []:
            if state != "Off":
                return _decode_pdf_name(state)
    return None


def _dropdown_choices(widget: Any) -> list[str]:
    """Normalize combo/list box choices; entries may be str or (export, display)."""
    labels = []
    for choice in widget.choice_values or []:
        if isinstance(choice, (list, tuple)) and choice:
            labels.append(str(choice[-1]))
        else:
            labels.append(str(choice))
    return labels


def _collect_form_fields(doc: Any) -> list[dict[str, Any]]:
    """Group widgets into logical fields, in document encounter order.

    Radio kids share one field name, so a group collapses into a single record
    whose options follow kid encounter order — the same zero-based index
    PyPDFForm uses for filling. Internal keys (``_loc``, ``_pages``) support
    later enrichment and are stripped before serialization.
    """
    fields: dict[str, dict[str, Any]] = {}
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        for widget in page.widgets():
            name = widget.field_name
            field_type = _WIDGET_TYPE_MAP.get(widget.field_type_string)
            if not name or field_type is None:
                continue
            record = fields.get(name)
            if record is None:
                record = {
                    "name": name,
                    "type": field_type,
                    "page": page_index + 1,
                    "label": None,
                    "options": [],
                    "current_value": None,
                    "max_length": None,
                    "_loc": (page_index, tuple(widget.rect)),
                    "_pages": set(),
                }
                fields[name] = record
            record["_pages"].add(page_index + 1)

            label = (widget.field_label or "").strip()
            if label and not record["label"]:
                record["label"] = label

            value = widget.field_value
            if field_type == "radio":
                state = _widget_on_state(widget)
                if state is not None and state not in record["options"]:
                    record["options"].append(state)
                # The selected kid reports its on-state; unselected kids "Off".
                if value and value != "Off":
                    record["current_value"] = _decode_pdf_name(value)
            elif field_type == "checkbox":
                record["current_value"] = bool(value) and value != "Off"
            elif field_type == "dropdown":
                if not record["options"]:
                    record["options"] = _dropdown_choices(widget)
                if value not in (None, ""):
                    record["current_value"] = value
            else:
                if widget.text_maxlen:
                    record["max_length"] = widget.text_maxlen
                if value not in (None, ""):
                    record["current_value"] = value
    return list(fields.values())


def _nearby_label_text(doc: Any, record: dict[str, Any]) -> str | None:
    """Best-effort semantic hint from printed text around a widget.

    Form labels conventionally sit to the right of checkboxes, to the left of
    a radio group's buttons (the question text; each button only carries its
    option label), and above or to the left of text boxes; clip-extract
    printed text from those spots, in order of likelihood.
    """
    pymupdf, _ = _require_pdf_libs()
    page_index, (x0, y0, x1, y1) = record["_loc"]
    page = doc.load_page(page_index)
    if record["type"] == "radio":
        clips = [
            pymupdf.Rect(x0 - 300, y0 - 6, x0 - 1, y1 + 6),
            pymupdf.Rect(x1 + 1, y0 - 3, x1 + 220, y1 + 3),
        ]
    elif record["type"] == "checkbox":
        clips = [
            pymupdf.Rect(x1 + 1, y0 - 3, x1 + 220, y1 + 3),
            pymupdf.Rect(x0 - 300, y0 - 3, x0 - 1, y1 + 3),
        ]
    else:
        clips = [
            pymupdf.Rect(x0 - 4, y0 - 24, x1 + 60, y0 - 1),
            pymupdf.Rect(x0 - 240, y0 - 3, x0 - 1, y1 + 3),
        ]
    for clip in clips:
        if clip.is_empty:
            continue
        text = " ".join(page.get_text("text", clip=clip).split())
        if text:
            return text[:_NEARBY_TEXT_MAX_CHARS]
    return None


def _compress_page_ranges(pages: list[int]) -> str:
    """Compress sorted page numbers to a compact range string like ``1-3,5``."""
    ranges: list[str] = []
    start = prev = None
    for page in pages:
        if start is None:
            start = prev = page
        elif page == prev + 1:
            prev = page
        else:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = page
    if start is not None:
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _serialize_field(record: dict[str, Any], fillable_keys: set[str] | None) -> dict[str, Any]:
    """Compact one field record for the LLM: omit empty/irrelevant keys."""
    entry: dict[str, Any] = {
        "name": record["name"],
        "type": record["type"],
        "page": record["page"],
    }
    if record["label"]:
        entry["label"] = record["label"]
    # nearby_text is only populated when the label is missing or ambiguous,
    # so emitting both never bloats the common case.
    if record.get("nearby_text"):
        entry["nearby_text"] = record["nearby_text"]
    if record["options"]:
        entry["options"] = {str(i): label for i, label in enumerate(record["options"])}
    value = record["current_value"]
    # Identity check for False: `0 in (None, "", False)` is True in Python,
    # which would wrongly drop a legitimate value of 0.
    if not (value is None or value == "" or value is False):
        entry["current_value"] = value
    if record["max_length"]:
        entry["max_length"] = record["max_length"]
    # Flag names PyPDFForm cannot address so the LLM does not waste a fill call.
    if fillable_keys is not None and record["name"] not in fillable_keys:
        entry["fillable"] = False
    return entry


def _coerce_form_value(record: dict[str, Any], value: Any) -> tuple[Any, str | None]:
    """Coerce an LLM-supplied value to what PyPDFForm expects for the field.

    Returns ``(coerced, None)`` on success or ``(None, reason)`` on rejection.
    Radios and dropdowns are always resolved to a zero-based index here —
    passing an unknown string through would make PyPDFForm append it as a new
    dropdown option, which silently amplifies model hallucinations.
    """
    field_type = record["type"]
    if value is None:
        return None, "value is null"

    if field_type == "text":
        return value if isinstance(value, str) else str(value), None

    if field_type == "checkbox":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value), None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _CHECKBOX_TRUE:
                return True, None
            if lowered in _CHECKBOX_FALSE:
                return False, None
        return None, "checkbox expects true or false"

    if field_type in ("radio", "dropdown"):
        options = record["options"]
        if isinstance(value, bool):
            return None, f"expects a zero-based option index or one of: {options}"
        if isinstance(value, int) or (
            isinstance(value, str) and value.strip().lstrip("-").isdigit()
        ):
            index = int(value)
            if 0 <= index < len(options):
                return index, None
            return None, f"index {index} out of range; options are: {options}"
        if isinstance(value, str):
            text = value.strip()
            for i, option in enumerate(options):
                if option == text:
                    return i, None
            lowered = text.casefold()
            for i, option in enumerate(options):
                if option.casefold() == lowered:
                    return i, None
            return None, f"'{text}' is not an option; choose one of: {options}"
        return None, f"expects a zero-based option index or one of: {options}"

    if field_type == "signature":
        if isinstance(value, str) and Path(value).expanduser().exists():
            return str(Path(value).expanduser()), None
        return None, "signature expects the path of an existing image file"

    return None, f"field type '{field_type}' cannot be filled"


def _form_wrapper(PdfWrapper: Any, local_path: str, names: set[str]) -> Any:
    """Build the PdfWrapper whose fill keys best match PyMuPDF's full names.

    Forms with hierarchical field names need ``use_full_widget_name=True`` for
    the keys to line up; flat forms work either way. Pick empirically.
    """
    wrapper = PdfWrapper(local_path, need_appearances=True)
    keys = set((wrapper.schema or {}).get("properties", {}))
    if names - keys:
        full = PdfWrapper(local_path, need_appearances=True, use_full_widget_name=True)
        full_keys = set((full.schema or {}).get("properties", {}))
        if len(names & full_keys) > len(names & keys):
            return full
    return wrapper


class Pdf:
    """Navigate and extract PDF content safely for LLM workflows.

    The default API is intentionally navigation-oriented: call metadata first,
    then read page ranges or search for relevant pages. All page fields and
    start_page/end_page arguments are 1-based PDF physical pages, not printed
    page labels shown inside the document.
    """

    def __init__(
        self,
        vision: str | None = None,
        vision_api_key: str | None = None,
        vision_dpi: int = _DEFAULT_VISION_DPI,
    ):
        """Configure the PDF reader.

        Args:
            vision: Optional model URI (e.g. ``"anthropic/claude-sonnet-4-6"``
                or a full ``provider@base_url:model`` URI) of a vision-capable
                model. When set, pages whose tables are unsafe for linear
                extraction (rotated axis labels, merged/spanning cells) are
                rendered to an image and re-read by this model so the returned
                tables stay structurally faithful. This is fully transparent to
                callers. Prefer a frontier model: cell VALUES are transcribed
                accurately by ``claude-sonnet-4-6``, ``gpt-4o`` and the
                domestically-reachable ``zhipu/glm-4.5v`` and
                ``bailian/qwen-vl-max`` alike, but MERGED-CELL structure
                (spanning cells, tall rotated labels) is only fully reliable on
                frontier models — mid-tier models occasionally miss a span by
                one row on the most deeply nested tables. ``qwen-vl-max`` works
                well ONLY at high DPI: at low DPI it shifts whole columns and
                produces wrong values, so never pair a qwen model with DPI
                below 300 (it is also markedly slower, ~80-120s/page). If you
                have no adequate model, leave vision unset and rely on the
                inline complex-table warning plus render_page instead.
            vision_api_key: API key for the vision model. When omitted, it is
                resolved from the ``{PROVIDER}_API_KEY`` environment variable
                matching the vision URI's provider.
            vision_dpi: Render resolution for the vision fallback and
                render_page. 200 DPI is the HARD FLOOR for correct merged-cell
                spans — below it, even strong models mis-judge rowspan/colspan
                (150 DPI reproducibly broke a span in testing). Defaults to 300,
                the reliable resolution for the deeply nested tables that
                trigger the vision path; higher values rarely help because
                tiled models are fixed-token and glm-class hits its per-image
                token cap around 300.
        """
        self.vision = vision
        self.vision_api_key = vision_api_key
        self.vision_dpi = vision_dpi
        # Resolved lazily and cached; _UNSET means "not yet attempted".
        self._vision_provider_cache: Any = _UNSET

    def __available__(self) -> frozenset[str]:
        """Explicitly declare which methods are exposed as LLM tools.

        Keeps the tool surface stable and prevents internal helpers from ever
        leaking into the tool schema.
        """
        return frozenset(
            {
                "metadata",
                "outline",
                "search",
                "read_pages",
                "read_all",
                "render_page",
                "schema",
                "fill",
            }
        )

    def _vision_configured(self) -> bool:
        return bool(self.vision)

    def _vision_provider(self) -> Any:
        """Resolve (once) and cache the vision provider, or None if unavailable.

        Never raises: a missing key or bad URI degrades to None so extraction
        falls back to plain text with an explicit warning rather than failing.
        """
        if not self.vision:
            return None
        if self._vision_provider_cache is not _UNSET:
            return self._vision_provider_cache

        from ...utils.logger import logger  # noqa: PLC0415

        provider = None
        try:
            from ...providers import create_provider  # noqa: PLC0415
            from ...providers.types import ProviderCategory  # noqa: PLC0415
            from ...utils.uri import parse as parse_uri  # noqa: PLC0415

            parsed = parse_uri(self.vision)
            api_key = self.vision_api_key or os.getenv(f"{parsed['provider'].upper()}_API_KEY")
            if not api_key:
                logger.warning(
                    f"[Pdf] vision model '{self.vision}' configured but no API key found "
                    f"(pass vision_api_key or set {parsed['provider'].upper()}_API_KEY); "
                    f"complex tables will fall back to plain text with a warning."
                )
            else:
                config: dict[str, Any] = {"api_key": api_key, "model": parsed["model"]}
                if parsed.get("base_url"):
                    config["base_url"] = parsed["base_url"]
                provider = create_provider(parsed["provider"], config, ProviderCategory.LLM)
        except Exception as error:
            logger.warning(f"[Pdf] failed to initialize vision model '{self.vision}': {error}")
            provider = None

        self._vision_provider_cache = provider
        return provider

    def _vision_transcribe_png(self, png: bytes | None, page_no: int) -> str | None:
        """Ask the vision model to transcribe a pre-rendered page image.

        The PNG must be rendered by the caller BEFORE pymupdf4llm.to_markdown
        runs: that call's OCR fallback mutates page state and degrades a later
        re-render (see _extract_pages). Returns None when no vision model is
        available, no image was captured, or the call fails, so the caller can
        degrade gracefully.
        """
        provider = self._vision_provider()
        if provider is None or not png:
            return None

        from ...utils.logger import logger  # noqa: PLC0415
        from ...message import HumanMessage, SystemMessage  # noqa: PLC0415

        try:
            data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
            messages = [
                SystemMessage(content=_VISION_SYSTEM_PROMPT),
                HumanMessage(
                    content=[
                        {"type": "text", "text": _VISION_USER_PROMPT.format(page=page_no)},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ]
                ),
            ]
            response = provider.send(messages=messages, stream=False)
            text = getattr(response, "content", "") or ""
            text = _strip_code_fences(text)
            return text or None
        except Exception as error:
            logger.warning(f"[Pdf] vision transcription failed for page {page_no}: {error}")
            return None

    def _extract_pages(self, doc: Any, pages: list[int], lib: Any) -> list[dict[str, Any]]:
        """Extract each requested page, transparently upgrading complex tables.

        For every page: use fast layout markdown by default; when the page holds
        a table unsafe for flattening, re-read it with the vision model (if
        configured) or annotate it with a reliability warning otherwise.

        Returns a list of records: ``{page, content, method, complex}`` where
        ``method`` is one of ``layout`` / ``plain_text`` / ``vision``.
        """
        # Detect complex tables AND render their page images BEFORE to_markdown.
        # pymupdf4llm's OCR fallback mutates page state: it changes what
        # find_tables() reports (breaking complexity detection) and degrades a
        # later re-render (a 469KB page collapsed to a 70KB blurred image in
        # testing), which would corrupt the vision transcription. Snapshot both
        # signals up front on the clean document. See _require_pdf_libs().
        vision_ready = self._vision_configured()
        reasons_by_page: dict[int, list[str]] = {}
        png_by_page: dict[int, bytes] = {}
        for page_index in pages:
            page = doc.load_page(page_index)
            reasons = _complex_table_reasons(page)
            reasons_by_page[page_index] = reasons
            if reasons and vision_ready:
                try:
                    png_by_page[page_index] = _render_page_png(page, self.vision_dpi)
                except Exception:
                    pass

        try:
            chunks = lib.to_markdown(doc, pages=pages, page_chunks=True)
            base_method = "layout"
        except ValueError:
            # pymupdf4llm raises on some complex/empty table layouts; fall back.
            chunks = _fallback_chunks(doc, pages)
            base_method = "plain_text"

        results: list[dict[str, Any]] = []
        for index, page_index in enumerate(pages):
            page_no = page_index + 1
            base_md = chunks[index].get("text", "") if index < len(chunks) else ""
            reasons = reasons_by_page.get(page_index, [])

            if not reasons:
                results.append(
                    {"page": page_no, "content": base_md, "method": base_method, "complex": False}
                )
                continue

            vision_text = self._vision_transcribe_png(png_by_page.get(page_index), page_no)
            if vision_text:
                results.append(
                    {
                        "page": page_no,
                        "content": _wrap_complex_vision(page_no, vision_text, reasons),
                        "method": "vision",
                        "complex": True,
                    }
                )
            else:
                results.append(
                    {
                        "page": page_no,
                        "content": _wrap_complex_warning(
                            page_no, base_md, reasons, self._vision_configured()
                        ),
                        "method": base_method,
                        "complex": True,
                    }
                )
        return results

    def metadata(
        self,
        source: str,
    ) -> str:
        """Return PDF metadata and high-level document overview.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.

        Returns:
            JSON string with source, PDF physical page count, file size,
            structural metadata, TOC count, estimated total characters, sampled
            page text density, and compact document-scale signals. When the PDF
            is a fillable form, also includes form-field counts and a hint to
            call schema next. This method intentionally does not include page
            content; use read_pages when content is needed.
        """
        pymupdf, _ = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        file_size_bytes = Path(local_path).stat().st_size
        file_size_mb = _file_size_mb(file_size_bytes)
        with pymupdf.open(local_path) as doc:
            sample_pages = min(2, doc.page_count)
            sample_texts = [doc.load_page(i).get_text("text") for i in range(sample_pages)]
            sample_text = "\n\n".join(sample_texts)
            average_chars = int(len(sample_text) / sample_pages) if sample_pages else 0
            estimated_total_chars = average_chars * doc.page_count
            text_density = _text_density(average_chars)
            toc = doc.get_toc(simple=True)
            pdf_metadata = dict(doc.metadata or {})
            form_fields = _collect_form_fields(doc)
            has_xfa = _has_xfa(doc)

            payload = {
                "source": source,
                "pages": doc.page_count,
                "file_size_bytes": file_size_bytes,
                "file_size_mb": file_size_mb,
                "pdf_format": pdf_metadata.get("format"),
                "document_title": pdf_metadata.get("title"),
                "document_author": pdf_metadata.get("author"),
                "document_subject": pdf_metadata.get("subject"),
                "document_keywords": pdf_metadata.get("keywords"),
                "pdf_creator": pdf_metadata.get("creator"),
                "pdf_producer": pdf_metadata.get("producer"),
                "creation_date": pdf_metadata.get("creationDate"),
                "modification_date": pdf_metadata.get("modDate"),
                "is_encrypted": bool(pdf_metadata.get("encryption")),
                "toc_items": len(toc),
                "has_toc": bool(toc),
                "sampled_pages": sample_pages,
                "average_chars_per_sampled_page": average_chars,
                "estimated_total_chars": estimated_total_chars,
                "text_density": text_density,
                "size_category": _size_category(file_size_mb, doc.page_count, estimated_total_chars),
                "likely_scanned_or_image_heavy": _likely_scanned_or_image_heavy(
                    file_size_mb,
                    doc.page_count,
                    average_chars,
                ),
                "is_fillable_form": bool(form_fields),
            }
            if form_fields:
                by_type: dict[str, int] = {}
                widget_pages: set[int] = set()
                for record in form_fields:
                    by_type[record["type"]] = by_type.get(record["type"], 0) + 1
                    widget_pages.update(record["_pages"])
                payload["form_field_count"] = len(form_fields)
                payload["form_fields_by_type"] = by_type
                payload["form_pages"] = _compress_page_ranges(sorted(widget_pages))
                payload["form_hint"] = (
                    "This PDF is a fillable form. Call schema to get every field's "
                    "name, meaning, and options, then call fill with a "
                    "{field_name: value} mapping to fill it."
                )
            if has_xfa:
                payload["has_xfa"] = True
                payload["form_hint"] = (
                    "This PDF uses an XFA form, which this tool cannot fill; "
                    "only its AcroForm fields (if any) are accessible."
                )
        return _json(payload)

    def outline(
        self,
        source: str,
    ) -> str:
        """Return the full PDF outline, falling back to inferred Markdown headings.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.

        Returns:
            JSON string with outline items. Standard PDF TOC items contain level,
            title, and page. The page field is the 1-based PDF physical page,
            not the printed page label shown inside the document. When citing
            locations, prefer this PDF physical page value; printed page labels
            may be mentioned separately if they appear in the content. If the
            PDF has no standard TOC/bookmarks, headings are inferred from the
            full document using layout extraction, with a plain-text heading
            scan as a fallback if layout extraction fails.
        """
        pymupdf, _ = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        with pymupdf.open(local_path) as doc:
            toc = doc.get_toc(simple=True)
            if toc:
                items = [
                    {"level": level, "title": title, "page": page}
                    for level, title, page in toc
                ]
                return _json({"source": source, "type": "toc", "items": items, "total_items": len(toc)})

            # See _require_pdf_libs() docstring for why pymupdf.layout is banned.
            _, lib = _require_pdf_libs()
            try:
                chunks = lib.to_markdown(doc, page_chunks=True)
                headings = _headings_from_markdown_chunks(chunks)
                method = "layout_markdown"
            except Exception:
                headings = _headings_from_plain_text(doc)
                method = "plain_text"

            return _json(
                {
                    "source": source,
                    "type": "inferred_headings",
                    "method": method,
                    "items": headings,
                    "total_items": len(headings),
                }
            )

    def search(
        self,
        source: str,
        query: str,
        max_results: int = 20,
        context_chars: int = 220,
    ) -> str:
        """Search text quickly across the PDF and return page-level matches.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.
            query: Case-insensitive keyword or phrase to search.
            max_results: Maximum matches returned.
            context_chars: Characters of context around each match.

        Returns:
            JSON string with matching page numbers and snippets. The page field
            is the 1-based PDF physical page, not the printed page label shown
            inside the document. When citing search results, prefer this PDF
            physical page value; printed page labels may be mentioned separately
            if they appear in the snippet.
        """
        if not query:
            raise ValueError("query is required")

        pymupdf, _ = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        results = []
        query_lower = query.lower()
        with pymupdf.open(local_path) as doc:
            for page_index in range(doc.page_count):
                text = doc.load_page(page_index).get_text("text")
                position = text.lower().find(query_lower)
                if position < 0:
                    continue
                start = max(0, position - context_chars)
                end = min(len(text), position + len(query) + context_chars)
                results.append(
                    {
                        "page": page_index + 1,
                        "position": position,
                        "context": text[start:end],
                    }
                )
                if len(results) >= max_results:
                    break

            payload = {
                "source": source,
                "query": query,
                "pages": doc.page_count,
                "results": results,
            }
        return _json(payload)

    def read_pages(
        self,
        source: str,
        start_page: int,
        end_page: int,
        format: str = "markdown",
        max_chars: int | None = None,
    ) -> str:
        """Read a page range with layout-aware extraction.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.
            start_page: First PDF physical page to read, 1-based.
            end_page: Last PDF physical page to read, inclusive and 1-based.
            format: Output format: markdown, txt, html, or json.
            max_chars: Maximum characters returned for text formats. If None,
                the full extracted range is returned.

        Returns:
            Text for markdown/txt/html, or JSON page chunks for json. Text
            formats include a JSON-like navigation header followed by content.
            All page values are 1-based PDF physical pages, not printed page
            labels shown inside the document. When citing extracted content,
            prefer the PDF physical page values from the navigation header or
            JSON chunks; printed page labels inside the content may be mentioned
            separately but should not be mixed into the same page range.
        """
        valid_formats = {"markdown", "txt", "html", "json"}
        if format not in valid_formats:
            raise ValueError(f"Unsupported format '{format}'. Choose from: {', '.join(sorted(valid_formats))}")

        # LLMs often pass numeric args as strings; coerce so page comparisons
        # and slicing don't raise TypeError on str.
        start_page = int(start_page)
        end_page = int(end_page)
        if max_chars is not None:
            max_chars = int(max_chars)

        # Do NOT import pymupdf.layout here — see _require_pdf_libs() docstring
        # for the full explanation of why it is globally banned.
        pymupdf, lib = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        with pymupdf.open(local_path) as doc:
            pages = _page_numbers(start_page, end_page, doc.page_count)
            # Per-page extraction transparently upgrades complex-table pages to
            # a vision transcription (or annotates them) so callers never have
            # to know which pages are structurally hard.
            per_page = self._extract_pages(doc, pages, lib)
            complex_pages = [r["page"] for r in per_page if r["complex"]]
            vision_pages = [r["page"] for r in per_page if r["method"] == "vision"]

            if format == "json":
                payload = {
                    "source": source,
                    "start_page": start_page,
                    "end_page": end_page,
                    "total_pages": doc.page_count,
                    "extraction_method": _summarize_methods(per_page),
                    "complex_table_pages": complex_pages,
                    "vision_pages": vision_pages,
                    "pages": [
                        {
                            "page": record["page"],
                            "content": record["content"],
                            "extraction_method": record["method"],
                            "complex_table": record["complex"],
                        }
                        for record in per_page
                    ],
                }
                return _json(payload)

            md = "\n\n".join(record["content"] for record in per_page)
            formatted = _format_output(md, format, Path(source).stem)
            content = _content(formatted, max_chars)
            header = _json(
                {
                    "source": source,
                    "start_page": start_page,
                    "end_page": end_page,
                    "total_pages": doc.page_count,
                    "format": format,
                    "extraction_method": _summarize_methods(per_page),
                    "complex_table_pages": complex_pages,
                    "vision_pages": vision_pages,
                    "next_page": end_page + 1 if end_page < doc.page_count else None,
                }
            )
            return f"{header}\n\n{content}"

    def read_all(
        self,
        source: str,
        format: str = "markdown",
        max_chars: int | None = None,
    ) -> str:
        """Read the full PDF.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.
            format: Output format: markdown, txt, or html.
            max_chars: Maximum characters returned. If None, the full extracted
                document is returned. For large PDFs, prefer metadata, outline,
                search, and read_pages unless full text is explicitly required.

        Returns:
            Full-document text with navigation metadata. The pages field is the
            PDF physical page count, not the document's printed page label range.
            When citing content, prefer PDF physical page values over printed
            page labels shown inside the document.
        """
        valid_formats = {"markdown", "txt", "html"}
        if format not in valid_formats:
            raise ValueError(f"Unsupported format '{format}'. Choose from: {', '.join(sorted(valid_formats))}")

        # See _require_pdf_libs() docstring for why pymupdf.layout is banned.
        pymupdf, lib = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        with pymupdf.open(local_path) as doc:
            pages = list(range(doc.page_count))
            per_page = self._extract_pages(doc, pages, lib)
            complex_pages = [r["page"] for r in per_page if r["complex"]]
            vision_pages = [r["page"] for r in per_page if r["method"] == "vision"]
            md = "\n\n".join(record["content"] for record in per_page)
            formatted = _format_output(md, format, Path(source).stem)
            content = _content(formatted, max_chars)
            header = _json(
                {
                    "source": source,
                    "pages": doc.page_count,
                    "format": format,
                    "extraction_method": _summarize_methods(per_page),
                    "complex_table_pages": complex_pages,
                    "vision_pages": vision_pages,
                }
            )
            return f"{header}\n\n{content}"

    def render_page(
        self,
        source: str,
        page: int,
        dpi: int | None = None,
        output_path: str | None = None,
    ) -> str:
        """Render a single PDF page to a PNG image on disk.

        Use this as a fallback when a page contains a complex table (merged
        cells, rotated axis labels, footnote exceptions) that text extraction
        cannot represent, and no vision model is configured on the tool. The
        saved image can then be read by a vision-capable model or inspected
        manually to recover the true 2D structure.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.
            page: PDF physical page to render, 1-based.
            dpi: Render resolution. Higher improves small-text fidelity at the
                cost of a larger image. Defaults to the tool's configured DPI.
            output_path: Where to write the PNG. When omitted, a temp file is
                created and its path returned.

        Returns:
            JSON string with source, page, dpi, image_path, and image byte size.
        """
        pymupdf, _ = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        # LLMs often pass numeric args as strings; coerce so downstream
        # arithmetic (page - 1) and PyMuPDF (dpi / 72) don't blow up on str.
        page = int(page)
        render_dpi = int(dpi) if dpi is not None else self.vision_dpi
        with pymupdf.open(local_path) as doc:
            if page < 1 or page > doc.page_count:
                raise ValueError(
                    f"page must be between 1 and {doc.page_count} (got {page})"
                )
            png = _render_page_png(doc.load_page(page - 1), render_dpi)

        if output_path:
            image_path = str(Path(output_path).expanduser())
            Path(image_path).parent.mkdir(parents=True, exist_ok=True)
            with open(image_path, "wb") as handle:
                handle.write(png)
        else:
            tmp = tempfile.NamedTemporaryFile(
                suffix=f".page{page}.png", delete=False
            )
            tmp.write(png)
            tmp.close()
            image_path = tmp.name

        return _json(
            {
                "source": source,
                "page": page,
                "dpi": render_dpi,
                "image_path": image_path,
                "bytes": len(png),
            }
        )

    def schema(
        self,
        source: str,
    ) -> str:
        """Describe every fillable form field of a PDF form.

        Call metadata first; when it reports is_fillable_form, use this method
        to learn what data the form needs, then call fill.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF file.

        Returns:
            JSON string with total_fields and a fields array. Each entry has:
            name (the exact key to pass to fill), type (text, checkbox, radio,
            dropdown, or signature), page (1-based PDF physical page), label
            (the field's meaning, from its tooltip) and/or nearby_text
            (printed text next to the field, provided when the label is
            missing or shared by many fields), options (for radio/dropdown: a
            zero-based index to option-label map; fill with the index or the
            exact label), current_value (present only when already filled),
            and max_length for length-limited text fields. Entries marked
            "fillable": false cannot be written by fill. For a PDF without
            form fields, returns is_fillable_form false instead.
        """
        pymupdf, _ = _require_pdf_libs()
        local_path = _resolve_pdf(source)
        with pymupdf.open(local_path) as doc:
            records = _collect_form_fields(doc)
            if not records:
                return _json(
                    {
                        "source": source,
                        "is_fillable_form": False,
                        "hint": (
                            "This PDF has no fillable form fields; use read_pages "
                            "to read its content instead."
                        ),
                    }
                )
            # Only trust a tooltip when it discriminates: a label shared by
            # many fields (copy-paste TU errors, boilerplate like "State")
            # gets a nearby-text hint appended so the LLM can tell them apart.
            label_counts = Counter(r["label"] for r in records if r["label"])
            for record in records:
                label = record["label"]
                if not label or label_counts[label] >= _AMBIGUOUS_LABEL_MIN_COUNT:
                    record["nearby_text"] = _nearby_label_text(doc, record)

        # Cross-check against PyPDFForm so the LLM never wastes a fill call on
        # a field the filler cannot address. Inspection alone must not require
        # PyPDFForm, so degrade to "no cross-check" when it is missing.
        fillable_keys: set[str] | None = None
        try:
            PdfWrapper = _require_pypdfform()
            names = {record["name"] for record in records}
            wrapper = _form_wrapper(PdfWrapper, local_path, names)
            fillable_keys = set((wrapper.schema or {}).get("properties", {}))
        except Exception:
            pass

        return _json(
            {
                "source": source,
                "total_fields": len(records),
                "fields": [_serialize_field(record, fillable_keys) for record in records],
            }
        )

    def fill(
        self,
        source: str,
        data: dict[str, Any] | str,
        output_path: str | None = None,
    ) -> str:
        """Fill form fields of a PDF form and write the result to a new PDF.

        Use field names exactly as returned by schema. The output stays
        editable, so the form can be completed incrementally: pass a previous
        output as source (or set output_path to it) and fill more fields.

        Args:
            source: Local file path or HTTPS URL pointing to a PDF form.
            data: Mapping of field name to value. Value by field type —
                text: a string; checkbox: true or false; radio and dropdown:
                the zero-based option index or the exact option label from
                schema's options; signature: path of a local image file.
            output_path: Where to write the filled PDF. When omitted, a new
                file is created next to a temp directory and its path returned.
                May equal source to fill the working copy in place.

        Returns:
            JSON string with output_path and filled (number of fields
            written). Rejected fields appear in errors as {field_name: reason};
            fix them and call fill again on the output.
        """
        PdfWrapper = _require_pypdfform()
        pymupdf, _ = _require_pdf_libs()
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict) or not data:
            raise ValueError("data must be a non-empty {field_name: value} object")

        local_path = _resolve_pdf(source)
        with pymupdf.open(local_path) as doc:
            records = {record["name"]: record for record in _collect_form_fields(doc)}
        if not records:
            raise ValueError("This PDF has no fillable form fields.")

        wrapper = _form_wrapper(PdfWrapper, local_path, set(records))
        fillable_keys = set((wrapper.schema or {}).get("properties", {}))

        valid: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for name, value in data.items():
            record = records.get(name)
            if record is None or name not in fillable_keys:
                errors[name] = "unknown field name; use names exactly as returned by schema"
                continue
            coerced, reason = _coerce_form_value(record, value)
            if reason is not None:
                errors[name] = reason
                continue
            valid[name] = coerced

        if not valid:
            return _json({"filled": 0, "errors": errors})

        if output_path:
            out = Path(output_path).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out_path = str(out)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".filled.pdf", delete=False)
            tmp.close()
            out_path = tmp.name

        wrapper.fill(valid).write(out_path)

        payload: dict[str, Any] = {"output_path": out_path, "filled": len(valid)}
        if errors:
            payload["errors"] = errors
        return _json(payload)
