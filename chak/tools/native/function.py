"""
NativeFunctionTool: Wrap ordinary Python functions as tools

Supports automatic parsing of function signatures, type annotations, and docstrings,
implements the same interface as MCPTool, can be mixed.
"""

import asyncio
import inspect
import typing
from typing import Any, Callable, Dict, get_args, get_origin

try:
    from docstring_parser import parse as parse_docstring
except ImportError:
    # If docstring_parser is not installed, use simple parsing
    parse_docstring = None


class NativeFunctionTool:
    """
    Native Python function tool
    
    Automatically extracts from function:
    - Function name (name)
    - Function description (from docstring)
    - Parameter definitions (from type annotations and docstring)
    
    Example:
        def get_weather(city: str, unit: str = "celsius") -> str:
            '''
            Get city weather information
            
            Args:
                city: City name
                unit: Temperature unit
            '''
            return f"{city} sunny"
        
        tool = NativeFunctionTool(get_weather)
        result = await tool.call({"city": "Beijing", "unit": "celsius"})
    """
    
    def __init__(self, func: Callable):
        """
        Args:
            func: Python function (can be sync or async)
        """
        self.func = func
        self._name = func.__name__
        
        # Automatically parse function information
        self._description, self._input_schema = self._parse_function()
    
    def _parse_function(self) -> tuple:
        """Parse information from function signature and docstring"""
        # 1. Parse docstring to get description
        if parse_docstring and self.func.__doc__:
            doc = parse_docstring(self.func.__doc__)
            description = doc.short_description or doc.long_description or self._name
            param_docs = {p.arg_name: (p.description or "").strip() for p in doc.params}
        else:
            # Simple parsing: use first line as description
            if self.func.__doc__:
                description = self.func.__doc__.strip().split('\n')[0]
            else:
                description = self._name
            param_docs = {}
        
        # 2. Get function signature
        sig = inspect.signature(self.func)

        # 2b. Resolve string annotations back to real types.
        #
        # If the caller's module uses ``from __future__ import annotations``
        # (PEP 563), every annotation on the function is stored as a *string*
        # (e.g. ``'int'``) instead of the class object (``int``). The naive
        # ``sig.parameters[name].annotation`` path then returns those raw
        # strings, and ``_python_type_to_json_type`` — which compares against
        # actual classes (``py_type == int``) — silently falls through to
        # ``"string"`` for every parameter. Result: chak advertises every
        # tool argument as a JSON string even when the user wrote ``int``.
        #
        # ``typing.get_type_hints`` is the standard way to force resolution:
        # it evaluates the string annotations in the function's ``__globals__``
        # + ``__locals__`` and gives us the real classes back. We wrap it in
        # try/except because it can raise if an annotation references a
        # name that isn't importable in that scope (e.g. TYPE_CHECKING-only
        # imports, forward refs to locals). In that case we keep going with
        # whatever ``sig`` provides — better a partial schema than a crash.
        try:
            resolved_hints = typing.get_type_hints(self.func)
        except Exception:
            resolved_hints = {}

        # 3. Build JSON Schema
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Prefer the PEP-563-safe resolved hint. Fall back to whatever
            # ``sig.parameters`` reports (which may still be a string, or
            # ``inspect.Parameter.empty`` for un-annotated params — both
            # handled downstream by ``_python_type_to_json_type``).
            annotation = resolved_hints.get(param_name, param.annotation)
            param_schema = self._python_type_to_json_type(annotation)
            
            # Get parameter description from docstring
            param_desc = param_docs.get(param_name, "")
            
            # If schema is a dict (Pydantic), use it directly; otherwise wrap in type
            if isinstance(param_schema, dict):
                # Pydantic schema, merge with description
                properties[param_name] = {**param_schema}
                if param_desc:
                    properties[param_name]["description"] = param_desc
            else:
                # Simple type
                properties[param_name] = {
                    "type": param_schema,
                    "description": param_desc
                }
            
            # Determine if required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        input_schema = {
            "type": "object",
            "properties": properties,
        }
        
        if required:
            input_schema["required"] = required
        
        return description, input_schema
    
    def _python_type_to_json_type(self, py_type):
        """Python type → JSON Schema type or full schema for Pydantic models"""
        if py_type == inspect.Parameter.empty:
            return "string"
        
        # Handle Pydantic BaseModel - return full schema
        try:
            from pydantic import BaseModel
            if isinstance(py_type, type) and issubclass(py_type, BaseModel):
                return py_type.model_json_schema()
        except (ImportError, TypeError):
            pass
        
        # Handle basic types
        if py_type == str:
            return "string"
        elif py_type == int:
            return "integer"
        elif py_type == float:
            return "number"
        elif py_type == bool:
            return "boolean"
        
        # Handle generic types (e.g., List[str])
        origin = get_origin(py_type)
        if origin is list:
            args = get_args(py_type)
            if args:
                item_type = self._python_type_to_json_type(args[0])
                return "array"
            return "array"
        elif origin is dict:
            return "object"
        
        # Default to string
        return "string"
    
    @property
    def name(self) -> str:
        """Tool name"""
        return self._name
    
    @property
    def description(self) -> str:
        """Tool description"""
        return self._description
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """Input parameters JSON Schema"""
        return self._input_schema
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling format
        
        Returns:
            OpenAI tool definition dict
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        }
    
    async def call(self, arguments: Dict[str, Any], retry: bool = True, executor=None) -> Any:
        """
        Call function
        
        Args:
            arguments: Function arguments (dict)
            retry: Enable retry (meaningless for native functions, kept for interface consistency)
            executor: Executor instance (ThreadPoolExecutor/ProcessPoolExecutor) or None (use asyncio.to_thread)
        
        Returns:
            Function execution result
        """
        # Convert arguments if Pydantic models are expected
        sig = inspect.signature(self.func)
        # See ``_parse_function`` for the same rationale — PEP 563 turns
        # annotations into strings, so ``param.annotation`` on its own is
        # unreliable for ``issubclass`` checks. Resolve first, fall back
        # to raw signature annotations when resolution can't complete.
        try:
            resolved_hints = typing.get_type_hints(self.func)
        except Exception:
            resolved_hints = {}
        converted_args = {}

        try:
            from pydantic import BaseModel
            for param_name, value in arguments.items():
                param = sig.parameters.get(param_name)
                annotation = resolved_hints.get(
                    param_name,
                    param.annotation if param else inspect.Parameter.empty,
                )
                if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    # Parameter is Pydantic, convert dict to model instance
                    converted_args[param_name] = annotation.model_validate(value)
                else:
                    converted_args[param_name] = value
        except (ImportError, TypeError):
            converted_args = arguments
        
        # Detect if sync or async function
        if asyncio.iscoroutinefunction(self.func):
            # Async function: call directly (no executor needed)
            result = await self.func(**converted_args)
        else:
            # Sync function: needs to run in executor to avoid blocking event loop
            if executor is None:
                # Default: use asyncio's default thread pool
                result = await asyncio.to_thread(self.func, **converted_args)
            else:
                # Use custom executor (ThreadPoolExecutor or ProcessPoolExecutor)
                import functools
                loop = asyncio.get_event_loop()
                func_with_args = functools.partial(self.func, **converted_args)
                result = await loop.run_in_executor(executor, func_with_args)
        
        # Serialize result if it's a Pydantic model
        try:
            from pydantic import BaseModel
            if isinstance(result, BaseModel):
                return result.model_dump()
        except ImportError:
            pass
        
        return result
    
    def __repr__(self) -> str:
        """Detailed representation"""
        lines = [f"NativeFunctionTool(name='{self.name}')"]
        
        # Add description
        if self.description:
            desc = self.description[:80] + "..." if len(self.description) > 80 else self.description
            lines.append(f"  Description: {desc}")
        
        # Add parameters
        schema = self.input_schema
        if schema and isinstance(schema, dict):
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            if properties:
                lines.append("  Parameters:")
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    required_mark = " (required)" if param_name in required else " (optional)"
                    
                    param_line = f"    - {param_name}: {param_type}{required_mark}"
                    if param_desc:
                        param_line += f" - {param_desc[:60]}..." if len(param_desc) > 60 else f" - {param_desc}"
                    lines.append(param_line)
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return f"{self.name} - {self.description}"
