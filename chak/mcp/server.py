"""
Server 类：服务器配置

简单的配置构建器，用于声明 MCP 服务器连接信息。
"""

import fnmatch
import json
from typing import Any, Dict, List, Optional, Union

from .tool import Tool


class Server:
    """
    服务器配置
    
    极简的配置对象，支持三种传输类型：
    - stdio: 本地进程（command + args）
    - SSE: 长连接HTTP（url + headers）
    - Streamable HTTP: 无状态HTTP（url + headers）
    
    Example:
        # 方式 1：直接构造
        server = Server(
            url="https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        # 方式 2：从 JSON 字符串
        config_json = '{"url": "...", "headers": {...}}'
        server = Server.from_config(config_json)
        
        # 方式 3：从 Dict
        config_dict = {"url": "...", "headers": {...}}
        server = Server.from_config(config_dict)
        
        # stdio 服务器
        server = Server(
            command="python",
            args=["calculator.py"]
        )
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Args:
            url: HTTP(S) URL（用于 SSE 或 Streamable HTTP）
            command: 命令行程序（用于 stdio）
            args: 命令行参数（用于 stdio）
            headers: HTTP 请求头（用于 SSE 或 Streamable HTTP）
            **kwargs: 其他配置（如 type, env, baseUrl 等）
        """
        self._config: Dict[str, Any] = {
            "url": url,
            "command": command,
            "args": args,
            "headers": headers,
            **kwargs
        }
        
        # 删除 None 值
        self._config = {k: v for k, v in self._config.items() if v is not None}
    
    @classmethod
    def from_config(cls, config: Union[str, Dict[str, Any]]) -> "Server":
        """
        从 JSON 字符串或 Dict 创建 Server
        
        这是为了方便开发者直接使用从 MCP 托管平台复制的配置。
        
        Args:
            config: JSON 字符串或 Dict 配置
        
        Returns:
            Server 实例
        
        Example:
            # 从 JSON 字符串（从托管平台复制）
            config_json = '''
            {
                "type": "sse",
                "baseUrl": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
                "headers": {
                    "Authorization": "Bearer ${DASHSCOPE_API_KEY}"
                }
            }
            '''
            server = Server.from_config(config_json)
            
            # 从 Dict
            config_dict = {
                "url": "https://...",
                "headers": {"Authorization": "Bearer xxx"}
            }
            server = Server.from_config(config_dict)
        """
        if isinstance(config, str):
            # JSON 字符串
            config_dict: Dict[str, Any] = json.loads(config)
        else:
            config_dict = config
        
        # 从 Dict 创建
        return cls(**config_dict)
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取配置字典
        
        Returns:
            配置字典的副本
        """
        return self._config.copy()
    
    async def tools(self, patterns: Optional[List[str]] = None) -> List[Tool]:
        """
        Get tools from this server
        
        Args:
            patterns: Tool name patterns (supports wildcards)
                - Exact match: ["maps_geo", "maps_weather"]
                - Wildcard: ["maps_*"]
                - All: ["*"] or None (default)
        
        Returns:
            List[Tool]: List of tools
        
        Example:
            # Get all tools (default)
            tools = await Server(url="...").tools()
            
            # Get specific tools
            tools = await Server(url="...").tools(["maps_geo", "maps_weather"])
            
            # Get all maps_* tools
            tools = await Server(url="...").tools(["maps_*"])
        """
        # Default to all tools
        if patterns is None:
            patterns = ["*"]
        
        # Lazy import to avoid circular dependency
        from .client import MCPClient
        
        # Create MCP client (pass Server instance)
        client = MCPClient(self)
        
        # 发现所有工具
        all_mcp_tools = await client.list_tools()
        
        # 筛选匹配的工具
        selected_mcp_tools = []
        for pattern in patterns:
            matched = _match_pattern(all_mcp_tools, pattern)
            selected_mcp_tools.extend(matched)
        
        # 去重（保持顺序）
        seen = set()
        unique_mcp_tools = []
        for mcp_tool in selected_mcp_tools:
            if mcp_tool.name not in seen:
                seen.add(mcp_tool.name)
                unique_mcp_tools.append(mcp_tool)
        
        # 创建 Tool 对象
        result = [
            Tool(mcp_tool, client)
            for mcp_tool in unique_mcp_tools
        ]
        
        return result
    
    async def tool(self, name: str) -> Optional[Tool]:
        """
        Get a single tool by exact name
        
        Args:
            name: Exact tool name
        
        Returns:
            Tool | None: The requested tool, or None if not found
        
        Example:
            # Get single tool
            tool = await Server(url="...").tool("maps_geo")
            if tool:
                result = await tool.call({"location": "San Francisco"})
            
            # Or check explicitly
            tool = await Server(url="...").tool("maps_geo")
            if tool is None:
                print("Tool not found")
        """
        tools = await self.tools([name])
        return tools[0] if tools else None
    
    def __repr__(self) -> str:
        if "command" in self._config:
            return f"Server(command='{self._config['command']}')"
        elif "url" in self._config:
            return f"Server(url='{self._config['url']}')"
        else:
            return f"Server({self._config})"


def _match_pattern(tools: List[Any], pattern: str) -> List[Any]:
    """
    根据模式匹配工具
    
    Args:
        tools: mcp.types.Tool 列表
        pattern: 匹配模式（支持通配符 *）
    
    Returns:
        匹配的工具列表
    """
    if pattern == "*":
        return tools
    
    return [
        tool for tool in tools
        if fnmatch.fnmatch(tool.name, pattern)
    ]
