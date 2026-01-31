"""
SkillObjectTool: Wrap SkillBase instances for progressive disclosure

This module provides skill-aware wrapping that enables two-stage tool exposure:
1. Stage 1: Expose skill as a single entry tool (name + description)
2. Stage 2: Expand to reveal all internal method tools
"""

import inspect
from typing import Any, Dict, List

from ..native.function import NativeFunctionTool


class SkillObjectTool:
    """
    Skill object tool wrapper
    
    Wraps a SkillBase instance and provides two modes:
    - Entry mode: single tool representing the skill itself
    - Expanded mode: all public methods as individual tools
    
    This enables progressive disclosure: LLM first sees skills,
    then can "activate" a skill to access its internal tools.
    """
    
    def __init__(self, skill: Any):
        """
        Args:
            skill: SkillBase instance
        """
        self.skill = skill
        self._method_tools = self._discover_methods()
    
    def _discover_methods(self) -> Dict[str, NativeFunctionTool]:
        """
        Discover all public callable methods from the skill object.
        
        Returns:
            Dict mapping method name to NativeFunctionTool
        """
        methods = {}
        
        for name in dir(self.skill):
            # Skip private and magic methods
            if name.startswith('_'):
                continue
            
            # Get attribute
            attr = getattr(self.skill, name)
            
            # Only include callable methods (bound methods)
            if callable(attr) and inspect.ismethod(attr):
                # Wrap as NativeFunctionTool
                methods[name] = NativeFunctionTool(attr)
        
        return methods
    
    @property
    def name(self) -> str:
        """Skill name"""
        return self.skill.name
    
    @property
    def description(self) -> str:
        """Skill description"""
        return self.skill.description
    
    @property
    def method_names(self) -> List[str]:
        """Get list of available method names"""
        return list(self._method_tools.keys())
    
    def to_skill_entry_tool(self) -> Dict[str, Any]:
        """
        Convert to skill entry tool (Stage 1)
        
        Returns a single tool definition representing the skill itself.
        When LLM calls this, it signals intent to use the skill.
        
        Returns:
            OpenAI tool definition for skill entry
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.description}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "What you want to do with this skill"
                        }
                    },
                    "required": []
                }
            }
        }
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Convert to expanded method tools (Stage 2)
        
        Returns all internal methods as individual tool definitions.
        
        Returns:
            List of OpenAI tool definitions (one per method)
        """
        return [
            tool.to_openai_tool()
            for tool in self._method_tools.values()
        ]
    
    def get_method_tool(self, method_name: str) -> NativeFunctionTool:
        """
        Get NativeFunctionTool for a specific method
        
        Args:
            method_name: Name of the method
        
        Returns:
            NativeFunctionTool instance
        
        Raises:
            ValueError: If method not found
        """
        tool = self._method_tools.get(method_name)
        if not tool:
            raise ValueError(
                f"Method '{method_name}' not found in skill '{self.name}'. "
                f"Available methods: {', '.join(self.method_names)}"
            )
        return tool
    
    def __repr__(self) -> str:
        """Detailed representation"""
        method_count = len(self._method_tools)
        lines = [f"SkillObjectTool(name='{self.name}', methods={method_count})"]
        
        if self.description:
            desc = self.description[:60] + "..." if len(self.description) > 60 else self.description
            lines.append(f"  Description: {desc}")
        
        if self._method_tools:
            lines.append("  Methods:")
            for name, tool in self._method_tools.items():
                tool_desc = tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
                lines.append(f"    - {name}: {tool_desc}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return f"{self.name} skill ({len(self._method_tools)} methods)"
