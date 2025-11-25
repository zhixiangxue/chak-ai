"""
MCPTool class: Single MCP tool object

Tool object used directly by developers, encapsulating mcp.types.Tool and related client.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .client import MCPClient


class MCPTool:
    """
    Single MCP tool
    
    Tool object used directly by developers, providing:
    - Tool metadata (name, description, input_schema)
    - Tool invocation methods (with auto-retry)
    - Access to underlying MCP Client (advanced usage)
    """
    
    def __init__(self, mcp_tool: Any, client: "MCPClient"):
        """
        Args:
            mcp_tool: mcp.types.Tool object (from mcp-python SDK)
            client: MCPClient instance
        """
        self._mcp_tool = mcp_tool
        self._client = client
    
    @property
    def name(self) -> str:
        """Tool name"""
        return self._mcp_tool.name
    
    @property
    def description(self) -> str:
        """Tool description"""
        return self._mcp_tool.description or ""
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """Input schema (JSON Schema)"""
        return self._mcp_tool.inputSchema
    
    @property
    def mcp_tool(self) -> Any:
        """
        Get underlying mcp.types.Tool object
        
        For passing to LLM (internal use)
        """
        return self._mcp_tool
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Convert to OpenAI tool format
        
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
    
    @property
    def client(self) -> "MCPClient":
        """
        Get underlying MCP Client
        
        Advanced usage: Direct access to MCP client
        
        Note: MCPClient methods automatically manage connections,
        no need to manually call get_session().
        
        Example:
            tool = tools[0]
            client = tool.client
            # Call tool directly
            result = await client.call_tool("maps_geo", {"address": "West Lake"})
            # List tools
            tools = await client.list_tools()
        """
        return self._client
    
    async def call(self, arguments: Dict[str, Any], retry: bool = True) -> Any:
        """
        Call tool
        
        Args:
            arguments: Tool arguments (dict)
            retry: Enable retry (default True)
        
        Returns:
            Tool execution result
        
        Raises:
            Exception: Tool call failed (after retries)
        
        Example:
            tool = next(t for t in tools if t.name == "maps_geo")
            result = await tool.call({"address": "Hangzhou West Lake"})
            print(result)
        """
        if retry:
            return await self._call_with_retry(arguments)
        else:
            return await self._client.call_tool(self.name, arguments)
    
    async def _call_with_retry(
        self, 
        arguments: Dict[str, Any], 
        max_attempts: int = 3
    ) -> Any:
        """
        Retry with exponential backoff
        
        Retry strategy:
        - 1st failure: wait 1 second
        - 2nd failure: wait 2 seconds
        - 3rd failure: raise exception
        """
        for attempt in range(max_attempts):
            try:
                result = await self._client.call_tool(self.name, arguments)
                return result
            except Exception as e:
                if attempt < max_attempts - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    continue
                # Last attempt also failed, raise exception
                raise Exception(
                    f"Tool '{self.name}' failed after {max_attempts} attempts: {str(e)}"
                ) from e
    
    def __repr__(self) -> str:
        """Detailed representation with input schema"""
        lines = [f"MCPTool(name='{self.name}')"]
        
        # Add description
        if self.description:
            desc = self.description[:80] + "..." if len(self.description) > 80 else self.description
            lines.append(f"  Description: {desc}")
        
        # Add input schema (parameters)
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
                    
                    # Format parameter line
                    param_line = f"    - {param_name}: {param_type}{required_mark}"
                    if param_desc:
                        param_line += f" - {param_desc[:60]}..." if len(param_desc) > 60 else f" - {param_desc}"
                    lines.append(param_line)
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return f"{self.name} - {self.description}"
