"""
chak MCP (Model Context Protocol) integration module

Provides three usage patterns:

1. Single server (minimal)
   # Get multiple tools
   tools = await Server(url="...").tools(["maps_geo", "maps_weather"])
   
   # Get single tool
   tool = await Server(url="...").tool("maps_geo")

2. Combine multiple servers (Python native)
   tools = await Server(url="...").tools(["maps_*"]) + \
           await Server(command="python").tools(["add"])

3. Syntactic sugar (recommended for multiple servers)
   tools = await chak.tools(
       Server(url="...").tools(["maps_*"]),
       Server(command="python").tools(["add"])
   )
"""

from typing import List

from .server import Server
from .tool import Tool

__all__ = ["Server", "Tool", "tools"]


async def tools(*tool_lists: List[Tool]) -> List[Tool]:
    """
    语法糖：优雅地组合多个工具列表
    
    这是一个简单的辅助函数，用于优雅地组合来自多个服务器的工具。
    本质上就是合并多个 List[Tool]。
    
    Args:
        *tool_lists: 可变数量的工具列表
    
    Returns:
        List[Tool]: 合并后的工具列表
    
    Example:
        # 优雅的声明式写法
        tools = await chak.tools(
            Server(url="...").tools(["maps_*"]),
            Server(command="python").tools(["add"])
        )
        
        # 等价于（但更简洁）：
        tools = await Server(url="...").tools(["maps_*"]) + \
                await Server(command="python").tools(["add"])
    """
    result = []
    for tool_list in tool_lists:
        result.extend(tool_list)
    return result
