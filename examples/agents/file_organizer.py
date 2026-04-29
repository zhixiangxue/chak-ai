"""
File Organizer Agent Demo
==========================
Give the LLM a messy directory.  It will autonomously:
  1. Scan all files and identify their types
  2. Design a category structure (does NOT touch sub-directories)
  3. Create category sub-directories
  4. Move files into the right categories
  5. Rename obviously junk-named files where possible
  6. Write a detailed organisation report

Tools: FileSystem + Bash
No human guidance after the initial prompt.

WARNING: This demo MOVES files on disk.  Files are NOT deleted.
         Undo by moving them back from the category sub-directories.
"""

import asyncio
import os
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import chak
from chak.tools.std import FileSystem, Bash
from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent

_MODEL   = "anthropic/claude-sonnet-4-6"
_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# Target directory
# Override via env var:  DEMO_DOWNLOAD_DIR=/your/messy/folder
# Default: the OS Downloads folder (~\Downloads on Windows/macOS/Linux)
# ---------------------------------------------------------------------------
TARGET_DIR  = os.getenv("DEMO_DOWNLOAD_DIR", str(Path.home() / "Downloads"))
REPORT_FILE = str(Path(TARGET_DIR) / "__organiser_report.md")


async def main():
    fs   = FileSystem(workdir=TARGET_DIR)
    bash = Bash(timeout=60)

    conv = chak.Conversation(
        _MODEL,
        api_key=_API_KEY,
        tools=[fs, bash],
    )
    conv._tool_manager.max_iterations = 120

    prompt = f"""You are a professional file organiser agent. Your mission is to tidy up a
messy Windows downloads folder end-to-end — no human help, no confirmations.

**Target directory**: {TARGET_DIR}
**Report file**     : {REPORT_FILE}

---

## Rules

### What to organise
- Organise **loose files only** — do NOT touch any existing sub-directories
  (i.e., items whose type is "directory").  Leave them exactly where they are.
- Skip the report file itself once you create it.

### Category structure
Create these sub-directories inside {TARGET_DIR} and move files accordingly:

| Sub-directory       | Extensions / pattern                                    |
|---------------------|---------------------------------------------------------|
| `01_documents/`     | `.pdf`, `.docx`, `.doc`, `.txt`, `.md`, `.pem`          |
| `02_spreadsheets/`  | `.xlsx`, `.xls`, `.csv`                                 |
| `03_images/`        | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.svg`|
| `04_code/`          | `.py`, `.js`, `.ts`, `.sh`, `.json`, `.xml`, `.yaml`,   |
|                     | `.yml`, `.sql`, `.toml`                                 |
| `05_archives/`      | `.zip`, `.tar.gz`, `.gz`, `.rar`, `.7z`, `.tar`         |
| `06_executables/`   | `.exe`, `.msi`, `.dmg`, `.deb`, `.rpm`                  |
| `07_snapshots/`     | `*.snapshot` (any file whose name ends in .snapshot)    |
| `08_videos/`        | `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`                  |
| `09_misc/`          | everything else that does not fit the above             |

### Naming conflicts
If a destination file already exists (same name), append `_(2)`, `_(3)` etc.
before moving — do NOT overwrite.

### Important constraints
- Use **Windows-style paths** with backslashes for all Bash commands.
- Use the filesystem `move` tool for all file moves (not Bash mv/move).
- Create directories with Bash: `mkdir "D:\\downloads\\01_documents"` etc.
  (use `2>nul` to suppress "already exists" errors on Windows).
- Never delete any file.

---

## Steps

1. **Scan** — list all items in {TARGET_DIR}.  Separate files from
   directories.  Print a summary: total files, total directories.

2. **Plan** — decide which category each file belongs to.  Print the plan as
   a table: filename | category.

3. **Create directories** — mkdir each of the 9 category sub-directories.

4. **Move files** — move every loose file to its category sub-directory,
   one by one.  Print a one-liner for each move: `[MOVE] filename → category/`

5. **Write report** — save a Markdown report to {REPORT_FILE} containing:
   - Generation timestamp
   - Summary table: category | file count | file names
   - List of any files that could NOT be moved (with reason)
   - 3–5 observations about the directory (e.g. "47 JSON response dumps
     suggest automated API testing artefacts that could be archived")
   - Recommendations for further cleanup

6. **Print console summary** — after saving the report, print a short
   human-readable summary to stdout.
"""

    print("=" * 70)
    print(f"  File Organiser Agent — {TARGET_DIR}")
    print("=" * 70)
    print()

    async for event in await conv.asend(prompt, event=True):
        match event:
            case MessageChunk(content=text) if text:
                print(text, end="", flush=True)

            case ToolCallStartEvent(tool_name=name, arguments=args):
                hint = ""
                if "command" in args:
                    hint = args["command"][:120].replace("\n", " ")
                elif "source" in args:
                    hint = f"{args.get('source','')[:60]} → {args.get('destination','')[:40]}"
                elif "path" in args:
                    hint = args["path"]
                elif "uri" in args:
                    hint = args["uri"]
                print(f"\n\n>>> [{name}] {hint}")

            case ToolCallSuccessEvent(tool_name=name, result=res):
                preview = (res or "")[:200].replace("\n", " ")
                print(f"<<< {preview}")

            case ToolCallErrorEvent(tool_name=name, error=err):
                print(f"<<< ERROR: {err}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
