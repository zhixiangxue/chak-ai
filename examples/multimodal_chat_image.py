"""
Multimodal Chat Example - Image Support (Bailian Qwen-VL)

This example demonstrates how to use chak with Alibaba Cloud Bailian's
vision-enabled Qwen-VL model to analyze images.

Prerequisites:
    1. Get your API key from: https://bailian.console.aliyun.com
    2. Set environment variable: export BAILIAN_API_KEY=your_key_here
    
Usage:
    python examples/multimodal_chat_image.py
"""

import asyncio
import os

import dotenv

dotenv.load_dotenv()

import chak
from chak import Image, Audio, MimeType, HumanMessage

# ============================================================================
# 🔑🔑🔑 IMPORTANT: Set your API key here 🔑🔑🔑
# ============================================================================
api_key = os.getenv("BAILIAN_API_KEY", "Your API key here")
if not api_key:
    print("❌ Error: Please set BAILIAN_API_KEY environment variable")
    print("   Example: export BAILIAN_API_KEY=sk-your-key-here")
    exit(1)


async def example1_single_image():
    """Example 1: Single Image Analysis"""
    conv = chak.Conversation(
        "bailian/qwen-vl-max",
        api_key=api_key
    )
    
    print("="*70)
    print("Example 1: Single Image Analysis")
    print("="*70)
    
    response = await conv.asend(
        "How do you solve this problem?",
        attachments=[
            Image("https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg")
        ]
    )
    print(f"\nResponse: {response.content}")
    print(f"\nStats: {conv.stats()}")


async def example2_multiple_images():
    """Example 2: Multiple Images Comparison"""
    conv = chak.Conversation(
        "bailian/qwen-vl-max",
        api_key=api_key
    )
    
    print("\n" + "="*70)
    print("Example 2: Multiple Images Comparison")
    print("="*70)
    
    response = await conv.asend(
        "What company's product is in the image? What product is it?",
        attachments=[
            Image("https://p.ampmake.com/mall/cms/news/1de73257-4172-4b5b-95f8-2f5dc5563609.jpg?x-oss-process=image/resize,w_800"),
            Image("https://cms-assets.xboxservices.com/assets/bc/40/bc40fdf3-85a6-4c36-af92-dca2d36fc7e5.png?n=642227_Hero-Gallery-0_A1_857x676.png", MimeType.PNG)
        ]
    )
    print(f"\nResponse: {response.content}")
    print(f"\nStats: {conv.stats()}")


async def example3_streaming():
    """Example 3: Streaming with Image"""
    conv = chak.Conversation(
        "bailian/qwen-vl-max",
        api_key=api_key
    )
    
    print("\n" + "="*70)
    print("Example 3: Streaming with Image")
    print("="*70)
    
    print("\nStreaming response: ", end="")
    async for chunk in await conv.asend(
        "Describe this image in detail:",
        attachments=[
            Image("https://img3.chinadaily.com.cn/images/202504/24/68099b50a310f5429c93178d.jpeg")
        ],
        stream=True
    ):
        print(chunk.content, end="", flush=True)
    print()  # New line
    print(f"\nStats: {conv.stats()}")


async def example4_audio():
    """Example 4: Audio Input"""
    # conv = chak.Conversation(
    #     "bailian/qwen-audio-turbo",  # Note： Qwen-Audio不支持OpenAI兼容协议，仅支持DashScope协议。
    #     api_key=api_key
    # )
    
    # print("\n" + "="*70)
    # print("Example 4: Audio Input")
    # print("="*70)
    
    # response = await conv.asend(
    #     "What is being said in this audio?",
    #     attachments=[
    #         Audio("https://bytedancespeech.github.io/seedtts_tech_report/audios/ZeroShotICL_samples/EN/prompt/4245145269330795065.wav", MimeType.WAV)
    #     ]
    # )
    # print(f"\nResponse: {response.content}")
    # print(f"\nStats: {conv.stats()}")
    pass


async def example5_advanced_multimodal():
    """Example 5: Advanced - Direct Multimodal Message"""
    conv = chak.Conversation(
        "bailian/qwen-vl-max",
        api_key=api_key
    )
    
    print("\n" + "="*70)
    print("Example 5: Advanced - Direct Multimodal Message")
    print("="*70)
    
    response = await conv.asend(
        HumanMessage(content=[
            {"type": "text", "text": "What is the main color in this image?"},
            {"type": "image_url", "image_url": {"url": "https://p0.meituan.net/smartvenus/22f27c3cdd8f2701ef246055b9b98b53235950.png.webp"}}
        ])
    )
    print(f"\nResponse: {response.content}")
    print(f"\nStats: {conv.stats()}")


async def main():
    """Run all examples"""
    # Run each example independently
    await example1_single_image()
    await example2_multiple_images()
    await example3_streaming()
    await example4_audio()
    await example5_advanced_multimodal()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
