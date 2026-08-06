"""Unit tests for ToolManager duplicate tool-name validation."""

import pytest

from chak.tools.manager import ToolManager
from chak.tools.native.object import NativeObjectTool
from chak.tools.skills import SkillBase, SkillObjectTool


pytestmark = pytest.mark.unit


class DuckTool:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    async def call(self, arguments: dict, **kwargs) -> str:
        return "ok"


def test_duplicate_regular_tool_names_raise_clear_error():
    with pytest.raises(ValueError, match="Duplicate tool name 'python'"):
        ToolManager([DuckTool("python"), DuckTool("python")])


def test_duplicate_native_object_method_and_regular_tool_name_raise():
    class Worker:
        def run(self) -> str:
            return "done"

    with pytest.raises(ValueError, match="Duplicate tool name 'worker-run'"):
        ToolManager([NativeObjectTool(Worker()), DuckTool("worker-run")])


def test_duplicate_skill_entry_and_regular_tool_name_raise():
    class ReportSkill(SkillBase):
        name = "report"
        description = "Generate reports"

        def build(self) -> str:
            return "report"

    with pytest.raises(ValueError, match="Duplicate tool name 'report'"):
        ToolManager([DuckTool("report"), SkillObjectTool(ReportSkill())])


def test_duplicate_skill_method_and_regular_tool_name_raise():
    class ReportSkill(SkillBase):
        name = "report"
        description = "Generate reports"

        def build(self) -> str:
            return "report"

    with pytest.raises(ValueError, match="Duplicate tool name 'report-build'"):
        ToolManager([DuckTool("report-build"), SkillObjectTool(ReportSkill())])
