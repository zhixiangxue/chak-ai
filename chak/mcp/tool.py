"""
Tool 类：单个 MCP 工具对象

开发者直接使用的工具对象，封装了 mcp.types.Tool 和相关的客户端。
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .client import MCPClient


class Tool:
    """
    单个 MCP 工具
    
    这是开发者直接操作的工具对象，提供：
    - 工具元信息（name, description, input_schema）
    - 工具调用方法（自动重试）
    - 访问底层 MCP Client（高级用法）
    """
    
    def __init__(self, mcp_tool: Any, client: "MCPClient"):
        """
        Args:
            mcp_tool: mcp.types.Tool 对象（来自 mcp-python SDK）
            client: MCPClient 实例
        """
        self._mcp_tool = mcp_tool
        self._client = client
    
    @property
    def name(self) -> str:
        """工具名称"""
        return self._mcp_tool.name
    
    @property
    def description(self) -> str:
        """工具描述"""
        return self._mcp_tool.description or ""
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """输入模式（JSON Schema）"""
        return self._mcp_tool.inputSchema
    
    @property
    def mcp_tool(self) -> Any:
        """
        获取底层的 mcp.types.Tool 对象
        
        用于传给 LLM（内部使用）
        """
        return self._mcp_tool
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """
        转换为 OpenAI 工具格式
        
        Returns:
            OpenAI 工具定义字典
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
        获取底层的 MCP Client
        
        高级用法：直接使用 MCP 客户端
        
        注意：MCPClient 的方法会自动管理连接，
        无需手动调用 get_session()。
        
        Example:
            tool = tools[0]
            client = tool.client
            # 直接调用工具
            result = await client.call_tool("maps_geo", {"address": "西湖"})
            # 列出工具
            tools = await client.list_tools()
        """
        return self._client
    
    async def call(self, arguments: Dict[str, Any], retry: bool = True) -> Any:
        """
        调用工具
        
        Args:
            arguments: 工具参数（dict）
            retry: 是否启用重试（默认 True）
        
        Returns:
            工具执行结果
        
        Raises:
            Exception: 工具调用失败（重试后仍失败）
        
        Example:
            tool = next(t for t in tools if t.name == "maps_geo")
            result = await tool.call({"address": "杭州西湖"})
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
        带指数退避的重试
        
        重试策略：
        - 第 1 次失败：等待 1 秒
        - 第 2 次失败：等待 2 秒
        - 第 3 次失败：抛出异常
        """
        for attempt in range(max_attempts):
            try:
                result = await self._client.call_tool(self.name, arguments)
                return result
            except Exception as e:
                if attempt < max_attempts - 1:
                    # 指数退避
                    await asyncio.sleep(2 ** attempt)
                    continue
                # 最后一次尝试也失败了，抛出异常
                raise Exception(
                    f"Tool '{self.name}' failed after {max_attempts} attempts: {str(e)}"
                ) from e
    
    def __repr__(self) -> str:
        """Detailed representation with input schema"""
        lines = [f"Tool(name='{self.name}')"]
        
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
