"""
SkillObjectTool: Wrap SkillBase instances for progressive disclosure

This module provides skill-aware wrapping that enables two-stage tool exposure:
1. Stage 1: Expose skill as a single entry tool (name + description)
2. Stage 2: Expand to reveal all internal method tools
"""

import inspect
from typing import Any, Dict, List, Optional

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
    
    def to_skill_entry_tool(self, include_method_param: bool = False) -> Dict[str, Any]:
        """
        Convert to skill entry tool (Stage 1 or Stage 2)
        
        Stage 1 (include_method_param=False): Activation mode, returns method summary
        Stage 2 (include_method_param=True): Selection mode, allows method selection with enum constraint
        
        Args:
            include_method_param: If True, include method parameter with enum constraint
        
        Returns:
            OpenAI tool definition for skill entry
        """
        properties = {
            "instruction": {
                "type": "string",
                "description": "Your goal or the user's question"
            }
        }
        
        if include_method_param:
            # Stage 2: Add method parameter with enum constraint
            properties["method"] = {
                "type": "string",
                "enum": list(self.method_names),
                "description": "Select method(s) from available methods based on your analysis of the task requirements"
            }
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.description}\n\n{'Select specific method(s) to use.' if include_method_param else 'Call this skill to see available methods and their parameters.'}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": []
                }
            }
        }
    
    def generate_method_summary(self, instruction: Optional[str] = None) -> str:
        """
        Generate method summary for LLM to choose from (Stage 2).
        
        Args:
            instruction: Optional instruction to help filter relevant methods
            
        Returns:
            Formatted method summary string
        """
        import inspect
        
        method_summaries = []
        
        for name, tool in self._method_tools.items():
            # Build method signature
            try:
                sig = inspect.signature(tool.func)
                params = []
                for pname, param in sig.parameters.items():
                    # Get type annotation
                    if param.annotation != inspect.Parameter.empty:
                        ptype = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)
                    else:
                        ptype = "Any"
                    params.append(f"{pname}: {ptype}")
                
                signature = f"{name}({', '.join(params)})"
                
                # Add return type
                if sig.return_annotation != inspect.Signature.empty:
                    rtype = sig.return_annotation.__name__ if hasattr(sig.return_annotation, '__name__') else str(sig.return_annotation)
                    signature += f" -> {rtype}"
                
            except Exception:
                # Fallback to simple name if signature parsing fails
                signature = f"{name}(...)"
            
            # Combine: signature + description
            desc = tool.description or "No description"
            method_summaries.append(f"  • {signature}\n    {desc}")
        
        summary = f"Skill '{self.name}' activated.\n\n"
        summary += f"Available methods ({len(self._method_tools)}):\n"
        summary += "\n".join(method_summaries)
        summary += f"\n\nNext step: Call '{self.name}' again with the 'method' parameter to select which method(s) you need.\n"
        summary += f"You can select multiple methods by making multiple tool calls if needed for comparison or comprehensive analysis."
        
        return summary
    
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
