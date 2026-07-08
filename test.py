import importlib.util
import subprocess
import sys
from pathlib import Path

PROVIDERS = [
    {"mark": "deepseek", "region": "domestic"},
    {"mark": "qwen",     "region": "domestic"},
    {"mark": "openai",   "region": "foreign"},
    {"mark": "claude",   "region": "foreign"},
    {"mark": "minimax",  "region": "foreign"},
]

_ALL_MARKS = {p["mark"] for p in PROVIDERS}
_DEFAULT_REGION = "domestic"

SCENARIO_MARKS = {
    "streaming": "streaming",
    "tools": "tools",
    "structured": "structured",
    "skill": "skill",
}


def load_dotenv_if_available(root: Path) -> None:
    try:
        import dotenv
    except ImportError:
        return
    dotenv.load_dotenv(root / ".env")


def ensure_pytest_available() -> bool:
    if importlib.util.find_spec("pytest") is not None:
        return True
    print("Error: pytest is not installed in the current Python environment.")
    print("Install test dependencies first:")
    print("  python -m pip install -e .[test]")
    return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_region(passthrough: list) -> tuple:
    """Pop ``--region <name>`` or ``--region=<name>`` from *passthrough*."""
    region = _DEFAULT_REGION
    out = []
    skip = False
    for i, arg in enumerate(passthrough):
        if skip:
            skip = False
            continue
        if arg == "--region" and i + 1 < len(passthrough):
            region = passthrough[i + 1]
            skip = True
            continue
        if arg.startswith("--region="):
            region = arg.split("=", 1)[1]
            continue
        out.append(arg)
    return region, out


def _filter(region: str) -> list:
    """Return PROVIDERS entries for *region* (``domestic``, ``foreign``, ``all``)."""
    if region == "all":
        return PROVIDERS
    if region in ("domestic", "foreign"):
        return [p for p in PROVIDERS if p["region"] == region]
    print(f"Error: unknown region '{region}'. Choose: domestic, foreign, all")
    return []


def _run(cmd: list, cwd: Path) -> int:
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=cwd)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    root = Path(__file__).resolve().parent
    if not (root / "chak").is_dir():
        print("Error: test.py must be run from the chak-ai project root.")
        return 2

    load_dotenv_if_available(root)
    if not ensure_pytest_available():
        return 2

    args = sys.argv[1:]
    command = args[0] if args else "default"
    passthrough = args[1:] if args else []

    # --- release: unit (all) + live (region-filtered) -----------------------

    if command == "release":
        region, passthrough = _resolve_region(passthrough)
        entries = _filter(region)
        if not entries:
            return 2
        marks = [p["mark"] for p in entries]
        marker = " or ".join(marks)
        print(f"Region: {region} ({', '.join(marks)})\n")

        print("--- unit ---")
        ret = _run([sys.executable, "-m", "pytest", "tests/unit", "-q", *passthrough], root)
        if ret != 0:
            return ret

        print("\n--- live ---")
        return _run([sys.executable, "-m", "pytest", "tests/live", "-q", "-m", marker, *passthrough], root)

    # --- unit ---------------------------------------------------------------

    if command == "unit":
        return _run([sys.executable, "-m", "pytest", "tests/unit", "-q", *passthrough], root)

    # --- live: region-filtered ----------------------------------------------

    if command == "live":
        region, passthrough = _resolve_region(passthrough)
        entries = _filter(region)
        if not entries:
            return 2
        marks = [p["mark"] for p in entries]
        marker = " or ".join(marks)
        print(f"Region: {region} ({', '.join(marks)})\n")
        return _run([sys.executable, "-m", "pytest", "tests/live", "-q", "-m", marker, *passthrough], root)

    # --- quick --------------------------------------------------------------

    if command == "quick":
        return _run([sys.executable, "-m", "pytest",
            "tests/unit/test_provider_error.py",
            "tests/unit/test_provider_base.py",
            "tests/unit/test_resilient_provider_policy.py",
            "-q", *passthrough], root)

    # --- provider: single provider ------------------------------------------

    if command == "provider":
        if not passthrough:
            print(f"Error: provider command requires one of: {', '.join(sorted(_ALL_MARKS))}")
            return 2
        mark = passthrough[0]
        if mark not in _ALL_MARKS:
            print(f"Error: unknown provider '{mark}'. Choose: {', '.join(sorted(_ALL_MARKS))}")
            return 2
        return _run([sys.executable, "-m", "pytest", "tests/live", "-q", "-m", mark, *passthrough[1:]], root)

    # --- scenario -----------------------------------------------------------

    if command == "scenario":
        if not passthrough:
            print(f"Error: scenario command requires one of: {', '.join(sorted(SCENARIO_MARKS))}")
            return 2
        s = passthrough[0]
        mark = SCENARIO_MARKS.get(s)
        if mark is None:
            print(f"Error: unknown scenario '{s}'. Choose: {', '.join(sorted(SCENARIO_MARKS))}")
            return 2
        return _run([sys.executable, "-m", "pytest", "tests/live", "-q", "-m", mark, *passthrough[1:]], root)

    # --- default / all / raw passthrough ------------------------------------

    if command in {"default", "all"}:
        return _run([sys.executable, "-m", "pytest", "tests/unit", "tests/live", "-q", *passthrough], root)

    return _run([sys.executable, "-m", "pytest", command, *passthrough], root)


if __name__ == "__main__":
    raise SystemExit(main())
