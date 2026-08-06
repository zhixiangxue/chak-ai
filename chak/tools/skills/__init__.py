"""Skill-based tool grouping for chak.tools.

This module provides:
- SkillBase: base class for grouping related tools as a "skill"
- SkillObjectTool: wrapper for progressive disclosure
"""

from typing import Any

from .object import SkillObjectTool
from .claude import ClaudeSkill
from .runner import ScriptRunner, PyRunner


class SkillBase:
    """Base class for skill-style tool groups.

    Subclasses should define:
    - name: short skill name shown to the LLM
    - description: one-line description of what this skill can do
    """

    # Simple defaults so users are not forced to override immediately
    name: str
    description: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            cls.name = cls.__name__
        if not hasattr(cls, "description"):
            cls.description = (cls.__doc__ or cls.__name__).strip().split("\n")[0]


__all__ = ["SkillBase", "SkillObjectTool", "ClaudeSkill", "ScriptRunner", "PyRunner"]
