"""
MCPClient 类：统一的 MCP 客户端

自动检测传输类型（stdio/SSE/Streamable HTTP），管理连接生命周期。
"""

import os
import re
from typing import TYPE_CHECKING, Any, Dict, List

# MCP Python SDK 导入
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    raise ImportError(
        "mcp-python SDK not found. Please install: pip install mcp"
    )

if TYPE_CHECKING:
    from .server import Server


class MCPClient:
    """
    统一的 MCP 客户端
    
    特性：
    - 自动检测传输类型（stdio/SSE/Streamable HTTP）
    - 每次调用都正确管理上下文（不复用连接，避免清理问题）
    - 线程安全
    
    注意：MCP SDK 使用上下文管理器，不适合长期连接复用。
    每次 list_tools() 或 call_tool() 都会创建新连接。
    """
    
    def __init__(self, server: "Server"):
        """
        Args:
            server: Server 实例，包含所有配置信息
        """
        self.server = server
        self.config = server.get_config()
        self.transport_type = self._detect_transport()
    
    def _detect_transport(self) -> str:
        """
        自动检测传输类型
        
        检测规则：
        - 有 "command" 字段 → stdio
        - type="sse" 或 URL 包含 "/sse" → SSE
        - type="streamable-http" 或其他 HTTP URL → Streamable HTTP
        
        Returns:
            "stdio" | "sse" | "streamable-http"
        """
        if "command" in self.config:
            return "stdio"
        
        if self.config.get("type") == "sse":
            return "sse"
        
        if self.config.get("type") == "streamable-http":
            return "streamable-http"
        
        # 根据 URL 推断
        url = self.config.get("url", "")
        if "/sse" in url.lower():
            return "sse"
        
        if url.startswith("http"):
            return "streamable-http"
        
        raise ValueError(
            f"Cannot detect transport type from config: {self.config}"
        )
    
    async def list_tools(self) -> List[Any]:
        """
        发现服务器提供的所有工具
        
        Returns:
            List[mcp.types.Tool]: 工具列表
        """
        if self.transport_type == "stdio":
            return await self._list_tools_stdio()
        elif self.transport_type == "sse":
            return await self._list_tools_sse()
        elif self.transport_type == "streamable-http":
            return await self._list_tools_http()
        else:
            raise NotImplementedError(f"Transport type {self.transport_type} not yet supported")
    
    async def _list_tools_stdio(self) -> List[Any]:
        """通过 stdio 列出工具"""
        command = self.config["command"]
        args = self.config.get("args", [])
        env = self.config.get("env")
        
        if env:
            env = self._expand_env_vars(env)
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools
    
    async def _list_tools_sse(self) -> List[Any]:
        """通过 SSE 列出工具"""
        url = self.config.get("url") or self.config.get("baseUrl")
        if not url:
            raise ValueError("SSE transport requires 'url' or 'baseUrl'")
        
        headers = self.config.get("headers", {})
        headers = self._expand_env_vars(headers)
        
        async with sse_client(url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools
    
    async def _list_tools_http(self) -> List[Any]:
        """通过 Streamable HTTP 列出工具"""
        url = self.config.get("url") or self.config.get("baseUrl")
        if not url:
            raise ValueError("HTTP transport requires 'url' or 'baseUrl'")
        
        headers = self.config.get("headers", {})
        headers = self._expand_env_vars(headers)
        
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """
        递归展开环境变量
        
        支持格式：${VAR_NAME}
        
        Example:
            "Bearer ${DASHSCOPE_API_KEY}" → "Bearer sk-xxx"
        """
        if isinstance(obj, str):
            # 替换 ${VAR_NAME}
            def replace_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            
            return re.sub(r'\$\{([^}]+)\}', replace_var, obj)
        
        elif isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        
        else:
            return obj
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
        
        Returns:
            工具执行结果
        """
        if self.transport_type == "stdio":
            return await self._call_tool_stdio(tool_name, arguments)
        elif self.transport_type == "sse":
            return await self._call_tool_sse(tool_name, arguments)
        elif self.transport_type == "streamable-http":
            return await self._call_tool_http(tool_name, arguments)
        else:
            raise NotImplementedError(f"Transport type {self.transport_type} not yet supported")
    
    async def _call_tool_stdio(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """通过 stdio 调用工具"""
        command = self.config["command"]
        args = self.config.get("args", [])
        env = self.config.get("env")
        
        if env:
            env = self._expand_env_vars(env)
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return result
    
    async def _call_tool_sse(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """通过 SSE 调用工具"""
        url = self.config.get("url") or self.config.get("baseUrl")
        if not url:
            raise ValueError("SSE transport requires 'url' or 'baseUrl'")
        
        headers = self.config.get("headers", {})
        headers = self._expand_env_vars(headers)
        
        async with sse_client(url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return result
    
    async def _call_tool_http(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """通过 Streamable HTTP 调用工具"""
        url = self.config.get("url") or self.config.get("baseUrl")
        if not url:
            raise ValueError("HTTP transport requires 'url' or 'baseUrl'")
        
        headers = self.config.get("headers", {})
        headers = self._expand_env_vars(headers)
        
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return result
    
    def __repr__(self) -> str:
        return f"MCPClient(transport={self.transport_type})"
