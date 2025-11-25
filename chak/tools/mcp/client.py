"""
MCPClient class: Unified MCP client

Automatically detects transport type (stdio/SSE/Streamable HTTP), manages connection lifecycle.
"""

import os
import re
from typing import TYPE_CHECKING, Any, Dict, List

# MCP Python SDK imports
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
    Unified MCP client
    
    Features:
    - Auto-detect transport type (stdio/SSE/Streamable HTTP)
    - Properly manage context for each call (no connection reuse, avoid cleanup issues)
    - Thread-safe
    
    Note: MCP SDK uses context managers, not suitable for long-lived connection reuse.
    Each list_tools() or call_tool() creates a new connection.
    """
    
    def __init__(self, server: "Server"):
        """
        Args:
            server: Server instance containing all configuration
        """
        self.server = server
        self.config = server.get_config()
        self.transport_type = self._detect_transport()
    
    def _detect_transport(self) -> str:
        """
        Auto-detect transport type
        
        Detection rules:
        - Has "command" field → stdio
        - type="sse" or URL contains "/sse" → SSE
        - type="streamable-http" or other HTTP URL → Streamable HTTP
        
        Returns:
            "stdio" | "sse" | "streamable-http"
        """
        if "command" in self.config:
            return "stdio"
        
        if self.config.get("type") == "sse":
            return "sse"
        
        if self.config.get("type") == "streamable-http":
            return "streamable-http"
        
        # Infer from URL
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
        Discover all tools provided by server
        
        Returns:
            List[mcp.types.Tool]: List of tools
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
        """List tools via stdio"""
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
        """List tools via SSE"""
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
        """List tools via Streamable HTTP"""
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
        Recursively expand environment variables
        
        Supported format: ${VAR_NAME}
        
        Example:
            "Bearer ${DASHSCOPE_API_KEY}" → "Bearer sk-xxx"
        """
        if isinstance(obj, str):
            # Replace ${VAR_NAME}
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
        Call tool
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
        
        Returns:
            Tool execution result
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
        """Call tool via stdio"""
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
        """Call tool via SSE"""
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
        """Call tool via Streamable HTTP"""
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
