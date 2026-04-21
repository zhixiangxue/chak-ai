"""
chak.tools - 工具集成模块

支持的工具类型：
- MCP 工具：通过 MCP 协议调用的工具
- 原生工具：普通 Python 函数或对象
- Skill 工具：Claude Agent Skills
- std 内置工具：chak 标准库（Bash、Python、FileSystem、Web、Search、Http）
"""

from typing import Callable, List, Union

# 导出子模块
from . import mcp
from . import native
from . import skills
from . import std

# 导出常用类
from .mcp import Server, MCPTool
from .native import NativeFunctionTool, NativeObjectTool
from .skills import SkillBase, SkillObjectTool, ClaudeSkill
from .std import Bash, Python, FileSystem, Web, Search, Http
from .manager import ToolManager

# Type alias for any supported tool — use this instead of repeating the full union.
ToolLike = Union[
    MCPTool,
    NativeFunctionTool,
    NativeObjectTool,
    SkillObjectTool,
    ClaudeSkill,
    Bash,
    Python,
    Callable,
    object,
]

__all__ = [
    "mcp",
    "native",
    "skills",
    "Server",
    "MCPTool",
    "NativeFunctionTool",
    "NativeObjectTool",
    "SkillBase",
    "SkillObjectTool",
    "ClaudeSkill",
    "Bash",
    "Python",
    "FileSystem",
    "Web",
    "Search",
    "Http",
    "ToolManager",
    "ToolLike",
    "wrap_tools",
]


def wrap_tools(tools: List[ToolLike]) -> List:
    """
    自动包装工具列表
    
    支持四种工具类型：
    - MCPTool → 保持不变
    - NativeFunctionTool → 保持不变
    - NativeObjectTool → 保持不变
    - SkillObjectTool → 保持不变
    - SkillBase instance → 包装成 SkillObjectTool
    - Callable → 包装成 NativeFunctionTool
    - Object (with methods) → 包装成 NativeObjectTool
    
    Args:
        tools: 工具列表，可以是：
            - MCPTool: MCP 协议工具
            - NativeFunctionTool: 已包装的函数工具
            - NativeObjectTool: 已包装的对象工具
            - SkillObjectTool: 已包装的 Skill 工具
            - SkillBase instance: Skill 实例（自动包装）
            - Callable: 普通函数（自动包装）
            - Object: 带公有方法的对象（自动包装）
    
    Returns:
        包装后的工具列表
    
    Example:
        # 普通函数
        def my_func(a: int) -> int:
            return a + 1
        
        # Skill
        class FileSkill(SkillBase):
            name = "file_helper"
            description = "Handle file operations"
            
            def read_file(self, path: str) -> str:
                return "content"
        
        # 对象
        class Calculator:
            def add(self, a: int, b: int) -> int:
                return a + b
        
        calc = Calculator()
        file_skill = FileSkill()
        
        # MCP 工具
        from chak.tools.mcp import Server
        server = Server(url="...")
        mcp_tools = await server.tools()
        
        # 混用：函数 + Skill + 对象 + MCP 工具
        wrapped = wrap_tools([my_func, file_skill, calc, *mcp_tools])
    """
    wrapped = []
    for tool in tools:
        if isinstance(tool, (MCPTool, NativeFunctionTool, NativeObjectTool, SkillObjectTool, ClaudeSkill, Bash, Python)):
            # 已经是工具对象，直接使用
            wrapped.append(tool)
        elif isinstance(tool, SkillBase):
            # 是 Skill 实例，包装成 SkillObjectTool
            wrapped.append(SkillObjectTool(tool))
        elif callable(tool):
            # 是普通函数，包装成 NativeFunctionTool
            wrapped.append(NativeFunctionTool(tool))
        elif hasattr(tool, '__dict__'):
            # 是对象（有属性字典），包装成 NativeObjectTool
            wrapped.append(NativeObjectTool(tool))
        else:
            raise TypeError(
                f"Tool must be MCPTool, NativeFunctionTool, NativeObjectTool, SkillObjectTool, "
                f"SkillBase instance, callable function, or object with methods, got {type(tool)}"
            )
    
    return wrapped
