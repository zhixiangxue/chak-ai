import pytest

import chak
from chak.tools import ClaudeSkill, SkillBase, wrap_tools

pytestmark = [pytest.mark.live, pytest.mark.tools]


async def collect_tool_events(conv, prompt):
    events = []
    async for event in await conv.asend(prompt, event=True, timeout=120):
        events.append(event)
    return events


def assert_tool_success(events, expected_tool_name):
    starts = [event for event in events if isinstance(event, chak.ToolCallStartEvent)]
    successes = [event for event in events if isinstance(event, chak.ToolCallSuccessEvent)]

    assert any(event.tool_name == expected_tool_name for event in starts)
    assert any(event.tool_name == expected_tool_name for event in successes)


@pytest.mark.asyncio
async def test_function_tool_calling(core_provider):
    def add_numbers(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        return a + b

    conv = chak.Conversation(
        core_provider.model_uri,
        api_key=core_provider.api_key,
        tools=wrap_tools([add_numbers]),
        timeout=120,
    )

    events = await collect_tool_events(
        conv,
        "You must call the add_numbers tool with a=7 and b=5. Do not answer directly before calling it.",
    )

    assert_tool_success(events, "add_numbers")


@pytest.mark.asyncio
async def test_object_tool_calling(core_provider):
    class Calculator:
        def multiply(self, a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

    conv = chak.Conversation(
        core_provider.model_uri,
        api_key=core_provider.api_key,
        tools=wrap_tools([Calculator()]),
        timeout=120,
    )

    events = await collect_tool_events(
        conv,
        "You must call the calculator-multiply tool with a=6 and b=7. Do not answer directly before calling it.",
    )

    assert_tool_success(events, "calculator-multiply")


@pytest.mark.asyncio
async def test_async_tool_calling(core_provider):
    async def async_add(a: int, b: int) -> int:
        """Add two integers asynchronously."""
        return a + b

    conv = chak.Conversation(
        core_provider.model_uri,
        api_key=core_provider.api_key,
        tools=wrap_tools([async_add]),
        timeout=120,
    )

    events = await collect_tool_events(
        conv,
        "You must call the async_add tool with a=3 and b=9. Do not answer directly before calling it.",
    )

    assert_tool_success(events, "async_add")


@pytest.mark.asyncio
async def test_skillbase_tool_calling(core_provider):
    class MathSkill(SkillBase):
        name = "math_skill"
        description = "Perform simple math operations."

        def subtract(self, a: int, b: int) -> int:
            """Subtract b from a."""
            return a - b

    conv = chak.Conversation(
        core_provider.model_uri,
        api_key=core_provider.api_key,
        tools=wrap_tools([MathSkill()]),
        timeout=120,
    )

    events = await collect_tool_events(
        conv,
        "You must use math_skill to select and call subtract with a=10 and b=4. Do not answer directly before calling tools.",
    )

    assert any(isinstance(event, chak.ToolCallSuccessEvent) for event in events)


@pytest.mark.skill
@pytest.mark.asyncio
async def test_claude_skill_shape_tool_calling(core_provider, tmp_path):
    skill_dir = tmp_path / "tiny_math_skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: tiny_math_skill\n"
        "description: Use this skill when asked to follow the tiny math skill instructions.\n"
        "---\n\n"
        "When activated, respond that the tiny math skill was activated.\n",
        encoding="utf-8",
    )

    skill = ClaudeSkill(str(skill_dir))
    conv = chak.Conversation(
        core_provider.model_uri,
        api_key=core_provider.api_key,
        tools=wrap_tools([skill]),
        timeout=120,
    )

    events = await collect_tool_events(
        conv,
        "You must call the tiny_math_skill tool now. Do not answer directly before calling it.",
    )

    assert_tool_success(events, "tiny_math_skill")
