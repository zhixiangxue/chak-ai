"""
Multimodal Chat Example - Document Support

This example demonstrates how to use chak with document attachments
(PDF, DOCX, XLSX, CSV, TXT, Link) to analyze and extract information from files.

Prerequisites:
    1. Install document processing dependencies (already in requirements.txt)
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here
    
Usage:
    python examples/multimodal_chat_documents.py
"""

import asyncio
import os

import dotenv

dotenv.load_dotenv()

import chak
from chak.attachment import PDF, DOC, Excel, CSV, TXT, Link

# ============================================================================
# 🔑🔑🔑 IMPORTANT: Set your API key here 🔑🔑🔑
# ============================================================================
api_key = os.getenv("BAILIAN_API_KEY", "Your API key here")
if not api_key:
    print("❌ Error: Please set BAILIAN_API_KEY environment variable")
    print("   Example: export BAILIAN_API_KEY=sk-your-key-here")
    exit(1)

# Document URLs
OSS_BASE_URL = "https://wcbpub.oss-cn-hangzhou.aliyuncs.com/xue/xxxx"
WEB_ARTICLE_URL = "https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents"


async def example1_pdf_analysis():
    """Example 1: PDF Document Analysis"""
    try:
        conv = chak.Conversation(
            "bailian/qwen-plus",
            api_key=api_key
        )
        
        print("="*70)
        print("Example 1: PDF Document Analysis")
        print("="*70)
        
        response = await conv.asend(
            "Summarize the core content of this PDF document in 3-5 sentences",
            attachments=[
                PDF(f"{OSS_BASE_URL}/Effective-harnesses-for-long-running-agents.pdf")
            ],
            timeout=120
        )
        print(f"\n📄 Response: {response.content}")
        print(f"\n📊 Stats: {conv.stats()}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")


async def example2_docx_analysis():
    """Example 2: Word Document Analysis"""
    try:
        conv = chak.Conversation(
            "bailian/qwen-plus",
            api_key=api_key
        )
        
        print("\n" + "="*70)
        print("Example 2: Word Document Analysis")
        print("="*70)
        
        response = await conv.asend(
            "What are the main challenges about long-running agents discussed in this document? List 3 key points",
            attachments=[
                DOC(f"{OSS_BASE_URL}/Effective-harnesses-for-long-running-agents.docx")
            ],
            timeout=120
        )
        print(f"\n📝 Response: {response.content}")
        print(f"\n📊 Stats: {conv.stats()}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")


async def example3_txt_analysis():
    """Example 3: Plain Text File Analysis"""
    try:
        conv = chak.Conversation(
            "bailian/qwen-turbo",
            api_key=api_key
        )
        
        print("\n" + "="*70)
        print("Example 3: Plain Text File Analysis")
        print("="*70)
        
        response = await conv.asend(
            "What does the 'harness' concept mentioned in the document mean? How does it help solve agent problems?",
            attachments=[
                TXT(f"{OSS_BASE_URL}/Effective-harnesses-for-long-running-agents.txt")
            ],
            timeout=120
        )
        print(f"\n📃 Response: {response.content}")
        print(f"\n📊 Stats: {conv.stats()}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")


async def example4_markdown_analysis():
    """Example 4: Markdown File Analysis"""
    try:
        conv = chak.Conversation(
            "bailian/qwen-plus",
            api_key=api_key
        )
        
        print("\n" + "="*70)
        print("Example 4: Markdown File Analysis")
        print("="*70)
        
        response = await conv.asend(
            "What are the best practices recommended by the author extracted from this Markdown document?",
            attachments=[
                TXT(f"{OSS_BASE_URL}/Effective-harnesses-for-long-running-agents.md")
            ],
            timeout=120
        )
        print(f"\n📋 Response: {response.content}")
        print(f"\n📊 Stats: {conv.stats()}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")


async def example5_csv_analysis():
    """Example 5: CSV Data Analysis"""
    try:
        conv = chak.Conversation(
            "bailian/qwen-plus",
            api_key=api_key
        )
        
        print("\n" + "="*70)
        print("Example 5: CSV Data Analysis")
        print("="*70)
        
        response = await conv.asend(
            "This CSV file contains customer data. Please help me find which customers are from Chile?",
            attachments=[
                CSV(f"{OSS_BASE_URL}/customers-100.csv")
            ],
            timeout=120
        )
        print(f"\n📊 Response: {response.content}")
        print(f"\n📊 Stats: {conv.stats()}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")


async def example6_excel_analysis():
    """Example 6: Excel Spreadsheet Analysis"""
    try:
        conv = chak.Conversation(
            "bailian/qwen-plus",
            api_key=api_key
        )
        
        print("\n" + "="*70)
        print("Example 6: Excel Spreadsheet Analysis")
        print("="*70)
        
        response = await conv.asend(
            "This Excel spreadsheet is product data. Please recommend 2-3 children's clothing products (if any), specify the product names and reasons",
            attachments=[
                Excel(f"{OSS_BASE_URL}/products-100.xlsx")
            ],
            timeout=120
        )
        print(f"\n📈 Response: {response.content}")
        print(f"\n📊 Stats: {conv.stats()}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")


async def example7_link_web_content():
    """Example 7: Web Link Content Analysis"""
    try:
        conv = chak.Conversation(
            "bailian/qwen-plus",
            api_key=api_key
        )
        
        print("\n" + "="*70)
        print("Example 7: Web Link Content Analysis")
        print("="*70)
        
        response = await conv.asend(
            "Please summarize the main points of this web article (3-5 sentences)",
            attachments=[
                Link(WEB_ARTICLE_URL)
            ],
            timeout=120
        )
        print(f"\n🔗 Response: {response.content}")
        print(f"\n📊 Stats: {conv.stats()}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")


async def example8_streaming_with_document():
    """Example 8: Streaming Response with Document"""
    try:
        conv = chak.Conversation(
            "bailian/qwen-plus",
            api_key=api_key
        )
        
        print("\n" + "="*70)
        print("Example 8: Streaming Response with Document")
        print("="*70)
        
        print("\n📝 Streaming response: ", end="")
        async for chunk in await conv.asend(
            "Please analyze this PDF document in detail, including: 1) Topic 2) Target audience 3) Core arguments",
            attachments=[
                PDF(f"{OSS_BASE_URL}/Effective-harnesses-for-long-running-agents.pdf")
            ],
            stream=True,
            timeout=120
        ):
            print(chunk.content, end="", flush=True)
        print()  # New line
        print(f"\n📊 Stats: {conv.stats()}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")


async def main():
    """Run all examples"""
    print("\n📚 Document Attachment Examples - Real Document Chat")
    print("="*70)
    
    # Run all examples
    await example1_pdf_analysis()
    await example2_docx_analysis()
    await example3_txt_analysis()
    await example4_markdown_analysis()
    await example5_csv_analysis()
    await example6_excel_analysis()
    await example7_link_web_content()
    await example8_streaming_with_document()
    
    print("\n" + "="*70)
    print("✨ All examples completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
