"""
NativeObjectTool: Wrap Python objects with multiple methods as tools

Automatically discovers all public methods from an object and wraps them as individual tools.
Maintains object state across method calls, enabling stateful tool interactions.
"""

import inspect
from typing import Any, Dict, List

from .function import NativeFunctionTool


class NativeObjectTool:
    """
    Native Python object tool
    
    Wraps a Python object and exposes all public methods as individual tools.
    Each method becomes a separate tool that the LLM can call.
    
    Key features:
    - Automatically discovers public methods (not starting with '_')
    - Maintains object state across calls (stateful)
    - Supports both sync and async methods
    - Reuses NativeFunctionTool for method wrapping
    
    Example:
        class Calculator:
            def __init__(self):
                self.history = []
            
            def add(self, a: int, b: int) -> int:
                '''Add two numbers'''
                result = a + b
                self.history.append(f"{a} + {b} = {result}")
                return result
            
            def get_history(self) -> list:
                '''Get calculation history'''
                return self.history
        
        calc = Calculator()
        tool = NativeObjectTool(calc)
        
        # LLM can call methods and state is preserved
        await tool.call_method("add", {"a": 10, "b": 20})
        await tool.call_method("add", {"a": 5, "b": 3})
        history = await tool.call_method("get_history", {})
        # history = ["10 + 20 = 30", "5 + 3 = 8"]
    """
    
    def __init__(self, obj: Any):
        """
        Args:
            obj: Python object instance with methods to expose as tools
        """
        self.obj = obj
        self._method_tools = self._discover_methods()
    
    def _discover_methods(self) -> Dict[str, NativeFunctionTool]:
        """
        Discover all public callable methods from the object.
        
        Returns:
            Dict mapping method name to NativeFunctionTool
        """
        methods = {}
        
        for name in dir(self.obj):
            # Skip private and magic methods
            if name.startswith('_'):
                continue
            
            # Get attribute
            attr = getattr(self.obj, name)
            
            # Only include callable methods (bound methods)
            if callable(attr) and inspect.ismethod(attr):
                # Wrap as NativeFunctionTool (reuse existing logic!)
                methods[name] = NativeFunctionTool(attr)
        
        return methods
    
    @property
    def method_names(self) -> List[str]:
        """Get list of available method names"""
        return list(self._method_tools.keys())
    
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
                f"Method '{method_name}' not found. "
                f"Available methods: {', '.join(self.method_names)}"
            )
        return tool
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Convert all methods to OpenAI function calling format
        
        Returns:
            List of OpenAI tool definitions (one per method)
        """
        return [
            tool.to_openai_tool()
            for tool in self._method_tools.values()
        ]
    
    async def call_method(self, method_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a specific method on the object
        
        Args:
            method_name: Name of the method to call
            arguments: Method arguments
        
        Returns:
            Method execution result
        
        Raises:
            ValueError: If method not found
        """
        tool = self.get_method_tool(method_name)
        return await tool.call(arguments)
    
    def __repr__(self) -> str:
        """Detailed representation"""
        obj_class = self.obj.__class__.__name__
        method_count = len(self._method_tools)
        
        lines = [f"NativeObjectTool(class='{obj_class}', methods={method_count})"]
        
        if self._method_tools:
            lines.append("  Methods:")
            for name, tool in self._method_tools.items():
                desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
                lines.append(f"    - {name}: {desc}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        obj_class = self.obj.__class__.__name__
        return f"{obj_class} ({len(self._method_tools)} methods)"
