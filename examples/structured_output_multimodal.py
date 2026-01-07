"""
Multimodal Structured Output Example

Demonstrates using the 'returns' parameter with image attachments
to extract structured data from visual content.

NOTE: This feature requires a vision model that supports function calling.
Currently, some providers' vision models may not support function calling.
- ✅ Supported: OpenAI GPT-4 Vision, GPT-4o
- ❌ Not yet: Bailian qwen-vl-max (as of current version)

This example shows how to:
- Combine image input with structured output
- Extract scene descriptions from images
- Parse visual information into Pydantic models
"""

import asyncio
import os
import base64
from typing import List, Optional

from pydantic import BaseModel, Field
import dotenv

dotenv.load_dotenv()

import chak
from chak.attachment import Image


# Define output models for image analysis
class SceneDescription(BaseModel):
    """Scene description extracted from image"""
    main_subject: str = Field(description="The main subject or focal point of the image")
    setting: str = Field(description="The location or setting (e.g., outdoor, indoor, nature)")
    time_of_day: Optional[str] = Field(default=None, description="Time of day if identifiable (morning, afternoon, evening, night)")
    weather: Optional[str] = Field(default=None, description="Weather conditions if visible")
    colors: List[str] = Field(description="Dominant colors in the image")
    mood: str = Field(description="Overall mood or atmosphere of the scene")
    objects: List[str] = Field(description="Notable objects or elements in the scene")


class ImageAnalysis(BaseModel):
    """Comprehensive image analysis"""
    category: str = Field(description="Image category (landscape, portrait, architecture, etc.)")
    description: str = Field(description="Brief overall description of the image")
    scene: SceneDescription = Field(description="Detailed scene analysis")
    tags: List[str] = Field(description="Relevant tags for the image")


async def main():
    """Main function demonstrating multimodal structured output"""
    
    # Check API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY not set")
        print("Please set your Bailian API key in .env file or environment variable")
        print("Get your key at: https://dashscope.aliyuncs.com/")
        return
    
    # Create conversation with vision model
    # Note: Some vision models may not support function calling
    # Using qwen3-vl-plus which supports both vision and function calling
    conv = chak.Conversation(
        "bailian/qwen3-vl-plus",  # Standard model with function calling support
        api_key=api_key
    )
    
    print("=== Multimodal Structured Output Demo ===\n")
    
    # Analyze the image
    print("📸 Analyzing image: tmp/1.jpeg")
    print("-" * 50)
    
    # Check if image exists
    image_path = "../tmp/1.jpeg"
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found")
        print("Please make sure the image is in the current directory")
        return
    
    try:
        # Load image and convert to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        image_uri = f"data:image/jpeg;base64,{image_data}"
        
        analysis = await conv.asend(
            "Analyze this image in detail. Identify the scene, main subjects, colors, mood, and any notable elements.",
            attachments=[Image(image_uri)],
            returns=ImageAnalysis
        )
        
        print(f"✅ Result type: {type(analysis).__name__}\n")
        print(f"📂 Category: {analysis.category}")
        print(f"📝 Description: {analysis.description}\n")
        
        print("🎬 Scene Details:")
        print(f"   Main Subject: {analysis.scene.main_subject}")
        print(f"   Setting: {analysis.scene.setting}")
        print(f"   Time of Day: {analysis.scene.time_of_day or 'Unknown'}")
        print(f"   Weather: {analysis.scene.weather or 'Unknown'}")
        print(f"   Colors: {', '.join(analysis.scene.colors)}")
        print(f"   Mood: {analysis.scene.mood}")
        print(f"   Objects: {', '.join(analysis.scene.objects)}")
        print()
        
        print(f"🏷️  Tags: {', '.join(analysis.tags)}")
        print()
        
        # Show conversation stats
        print("📊 Conversation Statistics:")
        stats = conv.stats()
        print(f"   Total messages: {stats['total_messages']}")
        print(f"   Total tokens: {stats['total_tokens']}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
