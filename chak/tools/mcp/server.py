"""
Server class: Server configuration

Simple configuration builder for declaring MCP server connection information.
"""

import fnmatch
import json
from typing import Any, Dict, List, Optional, Union

from .tool import MCPTool


class Server:
    """
    Server configuration
    
    Minimal configuration object supporting three transport types:
    - stdio: Local process (command + args)
    - SSE: Long-lived HTTP connection (url + headers)
    - Streamable HTTP: Stateless HTTP (url + headers)
    
    Example:
        # Method 1: Direct construction
        server = Server(
            url="https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        # Method 2: From JSON string
        config_json = '{"url": "...", "headers": {...}}'
        server = Server.from_config(config_json)
        
        # Method 3: From Dict
        config_dict = {"url": "...", "headers": {...}}
        server = Server.from_config(config_dict)
        
        # stdio server
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
            url: HTTP(S) URL (for SSE or Streamable HTTP)
            command: Command line program (for stdio)
            args: Command line arguments (for stdio)
            headers: HTTP request headers (for SSE or Streamable HTTP)
            **kwargs: Other configurations (such as type, env, baseUrl, etc.)
        """
        self._config: Dict[str, Any] = {
            "url": url,
            "command": command,
            "args": args,
            "headers": headers,
            **kwargs
        }
        
        # Remove None values
        self._config = {k: v for k, v in self._config.items() if v is not None}
    
    @classmethod
    def from_config(cls, config: Union[str, Dict[str, Any]]) -> "Server":
        """
        Create Server from JSON string or Dict
        
        This is for convenience when using configurations copied from MCP hosting platforms.
        
        Args:
            config: JSON string or Dict configuration
        
        Returns:
            Server instance
        
        Example:
            # From JSON string (copied from hosting platform)
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
            
            # From Dict
            config_dict = {
                "url": "https://...",
                "headers": {"Authorization": "Bearer xxx"}
            }
            server = Server.from_config(config_dict)
        """
        if isinstance(config, str):
            # JSON string
            config_dict: Dict[str, Any] = json.loads(config)
        else:
            config_dict = config
        
        # Create from Dict
        return cls(**config_dict)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dict
        
        Returns:
            Copy of configuration dict
        """
        return self._config.copy()
    
    async def tools(self, patterns: Optional[List[str]] = None) -> List[MCPTool]:
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
        
        # Discover all tools
        all_mcp_tools = await client.list_tools()
        
        # Filter matching tools
        selected_mcp_tools = []
        for pattern in patterns:
            matched = _match_pattern(all_mcp_tools, pattern)
            selected_mcp_tools.extend(matched)
        
        # Deduplicate (maintain order)
        seen = set()
        unique_mcp_tools = []
        for mcp_tool in selected_mcp_tools:
            if mcp_tool.name not in seen:
                seen.add(mcp_tool.name)
                unique_mcp_tools.append(mcp_tool)
        
        # Create MCPTool objects
        result = [
            MCPTool(mcp_tool, client)
            for mcp_tool in unique_mcp_tools
        ]
        
        return result
    
    async def tool(self, name: str) -> Optional[MCPTool]:
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
    Match tools based on pattern
    
    Args:
        tools: List of mcp.types.Tool
        pattern: Match pattern (supports wildcard *)
    
    Returns:
        List of matched tools
    """
    if pattern == "*":
        return tools
    
    return [
        tool for tool in tools
        if fnmatch.fnmatch(tool.name, pattern)
    ]
