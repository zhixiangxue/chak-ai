"""Skill-based tool calling demo: FileSkill

Demonstrates how to use SkillBase to group related file operations
as a single skill that the LLM can discover and use.
"""

import asyncio
import os
from dotenv import load_dotenv

from chak import Conversation
from chak.tools import SkillBase, wrap_tools

# Load environment variables
load_dotenv()


class FileSkill(SkillBase):
    """File operation skill for reading, analyzing, and summarizing files."""
    
    name = "file_helper"
    description = "Handle file reading, analysis and summarization tasks"
    
    def __init__(self):
        self.processed_files = []
    
    def read_file(self, path: str) -> str:
        """Read content from a file.
        
        Args:
            path: File path to read
            
        Returns:
            File content as string
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.processed_files.append(path)
            return f"Successfully read {len(content)} characters from {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def list_history(self) -> list:
        """Get list of all processed files.
        
        Returns:
            List of file paths that have been processed
        """
        return self.processed_files
    
    async def analyze_size(self, path: str) -> str:
        """Analyze file size and estimate reading time.
        
        Args:
            path: File path to analyze
            
        Returns:
            Size analysis result
        """
        import os
        try:
            size_bytes = os.path.getsize(path)
            size_kb = size_bytes / 1024
            read_time = size_kb / 100  # Assume 100KB/s reading speed
            
            return f"File: {path}\nSize: {size_kb:.2f} KB\nEstimated read time: {read_time:.2f} seconds"
        except Exception as e:
            return f"Error analyzing file: {str(e)}"


async def main():
    """Run skill-based file operations demo."""
    
    # Create skill instance
    file_skill = FileSkill()
    
    # Wrap as tools (all public methods become individual tools)
    tools = wrap_tools([file_skill])
    
    print(f"🔧 Loaded {len(tools)} tool(s) from FileSkill")
    print("Available tools:")
    for tool in tools:
        if hasattr(tool, '_method_tools'):
            for method_name, method_tool in tool._method_tools.items():
                print(f"  - {method_name}: {method_tool.description}")
    
    # Create conversation with tools
    conv = Conversation(
        provider_uri="qwen",
        model_uri="bailian/qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        tools=tools
    )
    
    # Test: ask LLM to use file operations
    print("\n" + "="*60)
    print("💬 User: Please analyze the size of README.md file")
    print("="*60)
    
    response = await conv.asend("Please analyze the size of README.md file in the current directory")
    print(f"\n🤖 Assistant: {response.content}")
    
    print("\n" + "="*60)
    print("💬 User: Show me the file processing history")
    print("="*60)
    
    response = await conv.asend("Show me the file processing history")
    print(f"\n🤖 Assistant: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
