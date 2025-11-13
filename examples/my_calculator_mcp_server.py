"""
Simple Calculator MCP Server using FastMCP

A local MCP server that provides basic calculator tools via stdio.
This demonstrates how to create your own MCP server that chak can connect to.

Usage:
    mcp run examples/my_calculator_mcp_server.py
"""

from mcp.server.fastmcp import FastMCP


# Create FastMCP server instance
mcp = FastMCP("Calculator")


@mcp.tool()
def add(a: float, b: float) -> str:
    """Add two numbers together"""
    result = a + b
    return f"Result: {a} + {b} = {result}"


@mcp.tool()
def subtract(a: float, b: float) -> str:
    """Subtract second number from first number"""
    result = a - b
    return f"Result: {a} - {b} = {result}"


@mcp.tool()
def multiply(a: float, b: float) -> str:
    """Multiply two numbers"""
    result = a * b
    return f"Result: {a} × {b} = {result}"


@mcp.tool()
def divide(a: float, b: float) -> str:
    """Divide first number by second number"""
    if b == 0:
        return "Error: Division by zero is not allowed"
    result = a / b
    return f"Result: {a} ÷ {b} = {result}"


@mcp.tool()
def power(base: float, exponent: float) -> str:
    """Raise base to the power of exponent"""
    result = base ** exponent
    return f"Result: {base} ^ {exponent} = {result}"
