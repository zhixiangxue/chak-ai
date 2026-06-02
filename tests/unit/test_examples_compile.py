import py_compile

import pytest

pytestmark = pytest.mark.unit


def test_examples_are_syntax_valid(project_root):
    example_files = sorted((project_root / "examples").rglob("*.py"))
    assert example_files

    for path in example_files:
        py_compile.compile(str(path), doraise=True)


def test_resilient_provider_example_uses_three_core_provider_types(project_root):
    example = project_root / "examples" / "resilient_provider_failover.py"
    content = example.read_text(encoding="utf-8")

    assert "anthropic@http://127.0.0.1:9" in content
    assert "openai@http://127.0.0.1:9/v1" in content
    assert "deepseek/deepseek-v4-flash" in content
