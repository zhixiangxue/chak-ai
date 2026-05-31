import importlib.util
import os
import subprocess
import sys
from pathlib import Path

CORE_API_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "BAILIAN_API_KEY",
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}

SCENARIO_MARKS = {
    "streaming": "streaming",
    "tools": "tools",
    "structured": "structured",
    "skill": "skill",
}

PROVIDER_MARKS = set(CORE_API_KEYS)


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

    if command == "release":
        missing = [name for name in CORE_API_KEYS.values() if not os.getenv(name)]
        if missing:
            print("Error: release mode requires these environment variables:")
            for name in missing:
                print(f"  - {name}")
            return 2
        pytest_args = ["tests/unit", "tests/live", "-q"]
    elif command == "unit":
        pytest_args = ["tests/unit", "-q"]
    elif command == "live":
        pytest_args = ["tests/live", "-q"]
    elif command == "quick":
        pytest_args = ["tests/unit/test_provider_error.py", "tests/unit/test_provider_base.py", "tests/unit/test_resilient_provider_policy.py", "-q"]
    elif command == "provider":
        if not passthrough:
            print("Error: provider command requires one of: deepseek, qwen, openai, claude")
            return 2
        provider = passthrough[0]
        if provider not in PROVIDER_MARKS:
            print("Error: unknown provider. Choose one of: deepseek, qwen, openai, claude")
            return 2
        pytest_args = ["tests/live", "-q", "-m", provider]
        passthrough = passthrough[1:]
    elif command == "scenario":
        if not passthrough:
            print("Error: scenario command requires one of: streaming, tools, structured, skill")
            return 2
        scenario = passthrough[0]
        marker = SCENARIO_MARKS.get(scenario)
        if marker is None:
            print("Error: unknown scenario. Choose one of: streaming, tools, structured, skill")
            return 2
        pytest_args = ["tests/live", "-q", "-m", marker]
        passthrough = passthrough[1:]
    elif command in {"default", "all"}:
        pytest_args = ["tests/unit", "tests/live", "-q"]
    else:
        pytest_args = [command, *passthrough]
        passthrough = []

    full_command = [sys.executable, "-m", "pytest", *pytest_args, *passthrough]
    print("Running:", " ".join(full_command))
    return subprocess.call(full_command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
