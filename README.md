<div align="center">

<a href="https://youtube.com/watch?v=xOKQ7EQcggw"><img src="https://raw.githubusercontent.com/zhixiangxue/chak-ai/main/docs/assets/logo.png" alt="Demo Video" width="120"></a>

[![PyPI version](https://badge.fury.io/py/chakpy.svg)](https://badge.fury.io/py/chakpy)
[![Python Version](https://img.shields.io/pypi/pyversions/chakpy)](https://pypi.org/project/chakpy/)
[![License](https://img.shields.io/github/license/zhixiangxue/chak-ai)](https://github.com/zhixiangxue/chak-ai/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/chakpy)](https://pypi.org/project/chakpy/)
[![GitHub Stars](https://img.shields.io/github/stars/zhixiangxue/chak-ai?style=social)](https://github.com/zhixiangxue/chak-ai)

[English](README.md) | [中文](docs/README_CN.md)

**A multi-model LLM client with built-in context management and flexible tool calling.**

chak is not another liteLLM, one-api, or OpenRouter, but a client library that actively manages conversation context and tool calls for you. Just focus on building your application, let chak handle the complexity.

</div>

<div align="center">

![Demo Video](https://raw.githubusercontent.com/zhixiangxue/chak-ai/main/docs/assets/demo.gif)

</div>

---

## 🌵 What's New

- **2025-01-12 | v0.2.6** - Added event stream support for real-time tool call observability. Use `event=True` to observe tool execution in your UI. See [Tool Call Observability](#tool-call-observability)
- **2025-01-09 | v0.2.5** - Added configurable tool executor for CPU-intensive tasks. Use `tool_executor` parameter to control execution mode. See [Tool Calling](#tool-calling)
- **2025-01-07 | v0.2.3** - Conversation now supports structured outputs via `returns` parameter. See [Structured Output](#structured-output)
- **2024-12-02 | v0.2.2** - Conversation now supports multimodal inputs. See [Multimodal Support](#multimodal-support)

---


## Core Features

### 🌱 Minimalist API Design

No complex configurations, no learning curve. chak is designed to be intuitive:

```python
# Use as SDK - connect to any LLM with a simple URI
conv = chak.Conversation("openai/gpt-4o-mini", api_key="YOUR_KEY")
response = conv.send("Hello!")

# Or run as a local gateway - start in 2 lines
import chak
chak.serve('chak-config.yaml')
```

Whether you're building an application or running a gateway, chak keeps things simple.

### 🌳 Multimodal Conversations

Conversations support multimodal inputs - images, audio, video, and documents. Just pass attachments:

```python
from chak import Image, PDF, Audio

# Send image with question
response = await conv.asend(
    "What's in this image?",
    attachments=[Image("photo.jpg")]  # local path, URL, or base64
)

# Analyze documents
response = await conv.asend(
    "Summarize this document",
    attachments=[PDF("report.pdf")]
)

# Multiple attachments at once
response = await conv.asend(
    "Compare these images",
    attachments=[
        Image("https://example.com/img1.jpg"),
        Image("./local/img2.png")
    ]
)
```

Supports images, audio, video, PDF, Word, Excel, CSV, TXT, and web links. See [Multimodal Support](#multimodal-support) for details.

### 🪴 Pluggable Context Management

Chak handles context automatically with multiple strategies:

```python
# Context is managed automatically
conv = chak.Conversation(
    "openai/gpt-4o",
    context_strategy=chak.FIFOStrategy(keep_recent_turns=5)
)
```

- **Now**: Short-term memory strategies (FIFO, Summarization, LRU) - production ready
- **Planning**: Long-term memory (RAG, memory bank) - making conversations truly "memorable"

No one else automates context management at this level. chak's strategy pattern makes it fully pluggable and extensible.

### 🌻 Simple Tool Calling

Write tools your way - functions, objects, or MCP servers, chak handles the rest:

```python
# Functions
def get_weather(city: str) -> str:
    ...

# Objects
class ShoppingCart:
    def add_item(self, name: str, price: float): ...
    def get_total(self) -> float: ...

cart = ShoppingCart()

# MCP servers
from chak.tools.mcp import Server
mcp_tools = await Server(url="...").tools()

# Use them, that's all
conv = Conversation(
    "openai/gpt-4o",
    tools=[get_weather, cart, *mcp_tools]
)
```

<a id="tool-call-observability"></a>

**Real-time Observability**: Get instant visibility into tool execution with event streams:

```python
from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent

# Use event=True to observe tool calls in real-time
async for event in await conv.asend("Calculate 15 + 27", event=True):
    match event:
        case ToolCallStartEvent(tool_name=name, arguments=args):
            print(f"🔧 Calling: {name} with {args}")
        
        case ToolCallSuccessEvent(tool_name=name, result=res):
            print(f"✅ Result: {name} -> {res}")
        
        case ToolCallErrorEvent(tool_name=name, error=err):
            print(f"❌ Failed: {name} - {err}")
        
        case MessageChunk(content=text, is_final=final):
            print(text, end="", flush=True)
```

Perfect for building UIs that show live tool execution progress. See [examples/event_stream_chat_demo.py](examples/event_stream_chat_demo.py)

**Configurable Execution**: For CPU-intensive tools, use `tool_executor` to control how tools run:

```python
import chak

# Default: best for IO-bound tasks (API calls, DB queries)
conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[...],
    tool_executor=chak.ToolExecutor.ASYNCIO  # default
)

# For CPU-intensive tasks: use process pool for true parallelism
conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[heavy_compute, ...],
    tool_executor=chak.ToolExecutor.PROCESS  # bypasses GIL
)

# Can switch anytime
conv.set_tool_executor(chak.ToolExecutor.PROCESS)

# Or override for a single call
await conv.asend("Run heavy task", tool_executor=chak.ToolExecutor.PROCESS)
```

**Choose the right executor**:

| Scenario | ASYNCIO | THREAD | PROCESS | Recommended |
|----------|---------|--------|---------|-------------|
| **CPU-intensive (sync)** | ❌ GIL limited | ❌ GIL limited | ✅ True parallel | PROCESS |
| **IO-intensive (async)** | ✅ Native concurrency | - | - | Default |
| **IO-intensive (sync)** | ✅ Works well | ✅ Works well | ⚠️ Overkill | ASYNCIO |

See full example: [examples/tool_calling_parallel_demo.py](examples/tool_calling_parallel_demo.py)

- **Now**: Functions, objects, and MCP tools all work the same way
- **Now**: Configurable executor for optimal performance
- **Planning**: Smart tool selection based on context

### 🌺 Structured Output

Get structured data directly from LLM responses using Pydantic models:

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    age: int = Field(description="User's age")

# Get structured output automatically
user = await conv.asend(
    "Create a user: John Doe, john@example.com, 30 years old",
    returns=User
)

print(user.name)   # "John Doe"
print(user.email)  # "john@example.com"
print(user.age)    # 30
```

Works with multimodal inputs too - extract structured data from images, documents, and more.

---

## Integrated Providers (18+)

OpenAI, Google Gemini, Azure OpenAI, Anthropic Claude, Alibaba Bailian, Baidu Wenxin, Tencent Hunyuan, ByteDance Doubao, Zhipu GLM, Moonshot, DeepSeek, iFlytek Spark, MiniMax, Mistral, SiliconFlow, xAI Grok, Ollama, vLLM, and more.

---

## 🌖 Quick Start

###  Installation

```bash
# Basic installation (SDK only)
pip install chakpy

# With server support
pip install chakpy[server]

# Install all optional dependencies
pip install chakpy[all]
```

### Chat with global models in a few lines

```python
import chak

conv = chak.Conversation(
    "openai/gpt-4o-mini",
    api_key="YOUR_KEY"
)

resp = conv.send("Explain context management in one sentence")
print(resp.content)
```

chak handles: connection initialization, message alignment, retry logic, context management, model format conversion... You just need to `send` messages.

---

## 🌒 Enable Automatic Context Management

Three built-in strategies:

- FIFO: Keep the last N turns, automatically drops older ones.
- Summarization: When context reaches a threshold, early history is summarized; recent turns stay in full.
- LRU: Built on Summarization, keeps hot topics and prunes cold ones.

Quick start:

```python
from chak import Conversation, FIFOStrategy

conv = Conversation(
    "bailian/qwen-flash",
    api_key="YOUR_KEY",
    context_strategy=FIFOStrategy(keep_recent_turns=3)
)
```

See full examples (parameters, how it works, tips):

- FIFO: [examples/strategy_chat_fifo.py](examples/strategy_chat_fifo.py)
- Summarization: [examples/strategy_chat_summarization.py](examples/strategy_chat_summarization.py)
- LRU: [examples/strategy_chat_lru.py](examples/strategy_chat_lru.py)

---

<a id="tool-calling"></a>

## 🌓 Tool Calling

Write tools the way you like - functions, objects, or MCP servers. chak handles the rest.

Just pass what you have, and it works.

### Pass Functions

Just pass regular Python functions:

```python
from datetime import datetime

def get_current_time() -> str:
    """Get current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(a: int, b: int, operation: str = "add") -> int:
    """Perform calculation on two numbers"""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    # ...

conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[get_current_time, calculate]
)

response = await conv.asend("What time is it? Then calculate 50 times 20")
```

**Type Safety with Pydantic**: Functions support [Pydantic](https://docs.pydantic.dev/) models for parameters and return values. Automatic validation and serialization included:

```python
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    age: int = Field(description="User's age")

class UserOutput(BaseModel):
    id: int
    name: str
    status: str = "active"

def create_user(user: UserInput) -> UserOutput:
    """Create a new user"""
    return UserOutput(id=123, name=user.name, status="active")

conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[create_user]
)

response = await conv.asend("Create a user: John Doe, john@example.com, 30 years old")
```

See full example: [tool_calling_chat_functions_pydantic.py](examples/tool_calling_chat_functions_pydantic.py)

### Pass Objects

Pass Python objects, their methods become tools. Object state persists across calls:

```python
class ShoppingCart:
    def __init__(self):
        self.items = []
        self.discount = 0
    
    def add_item(self, name: str, price: float, quantity: int = 1):
        """Add item to cart"""
        self.items.append({"name": name, "price": price, "quantity": quantity})
    
    def apply_discount(self, percent: float):
        """Apply discount percentage"""
        self.discount = percent
    
    def get_total(self) -> float:
        """Calculate total price"""
        subtotal = sum(item["price"] * item["quantity"] for item in self.items)
        return subtotal * (1 - self.discount / 100)

cart = ShoppingCart()

conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[cart]  # Pass object directly!
)

# LLM modifies cart state through natural language!
response = await conv.asend(
    "Add 2 iPhones at $999 each, then apply 10% discount and tell me the total"
)

print(cart.items)     # [{'name': 'iPhone', 'price': 999, 'quantity': 2}]
print(cart.discount)  # 10
print(cart.get_total())  # 1798.2
```

The LLM modifies object state through method calls.

**Pydantic + Stateful Objects**: Combine type safety with state persistence:

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    quantity: int = 1

class Order(BaseModel):
    order_id: int
    products: list[Product]
    total: float

class OrderManager:
    def __init__(self):
        self.orders = []  # State persists!
    
    def create_order(self, product: Product) -> Order:
        """Create order with type-safe Product"""
        order = Order(
            order_id=len(self.orders) + 1,
            products=[product],
            total=product.price * product.quantity
        )
        self.orders.append(order)
        return order
    
    def get_stats(self) -> dict:
        """Get statistics from accumulated state"""
        return {"total_orders": len(self.orders)}

manager = OrderManager()
conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[manager]  # Type-safe + stateful!
)

await conv.asend("Create an order: Laptop, $1200, quantity 1")
await conv.asend("Create another order: Mouse, $25, quantity 2")
response = await conv.asend("Show me the order statistics")

print(len(manager.orders))  # 2 - state persisted!
```

See full example: [tool_calling_chat_objects_pydantic.py](examples/tool_calling_chat_objects_pydantic.py)

### Pass MCP Tools

chak integrates the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/):

```python
import asyncio
from chak import Conversation
from chak.tools.mcp import Server

async def main():
    # Connect to MCP server and load tools
    tools = await Server(
        url="https://your-mcp-server.com/sse",
        headers={"Authorization": "Bearer YOUR_TOKEN"}
    ).tools()
    
    # Create conversation with tools
    conv = Conversation(
        "openai/gpt-4o",
        api_key="YOUR_KEY",
        tools=tools
    )
    
    # Model automatically calls tools when needed
    response = await conv.asend("What's the weather in San Francisco?")
    print(response.content)

asyncio.run(main())
```

Supports three transport types:

- **SSE** (Server-Sent Events): Cloud-hosted MCP services
- **stdio**: Local MCP servers
- **HTTP**: HTTP-based MCP services

### Mix Everything

Functions, objects, and MCP tools work together:

```python
def send_email(to: str, subject: str): ...

class OrderWorkflow:
    def add_items(self, items): ...
    def submit_order(self): ...

mcp_tools = await Server(url="...").tools()  # External tools

conv = Conversation(
    "openai/gpt-4o",
    tools=[
        send_email,           # Native function
        OrderWorkflow(),      # Native object (stateful!)
        *mcp_tools           # MCP tools
    ]
)
```

### Examples

See complete examples:

- **Native Functions**: [examples/tool_calling_chat_functions.py](examples/tool_calling_chat_functions.py)
- **Functions with Pydantic**: [examples/tool_calling_chat_functions_pydantic.py](examples/tool_calling_chat_functions_pydantic.py)
- **Stateful Objects**: [examples/tool_calling_chat_objects_stateful.py](examples/tool_calling_chat_objects_stateful.py)
- **Objects with Pydantic**: [examples/tool_calling_chat_objects_pydantic.py](examples/tool_calling_chat_objects_pydantic.py)
- **Event Stream (Observability)**: [examples/event_stream_chat_demo.py](examples/event_stream_chat_demo.py)
- **MCP (SSE)**: [examples/tool_calling_chat_mcp_sse.py](examples/tool_calling_chat_mcp_sse.py)
- **MCP (stdio)**: [examples/tool_calling_chat_mcp_stdio.py](examples/tool_calling_chat_mcp_stdio.py)
- **MCP (HTTP)**: [examples/tool_calling_chat_mcp_http.py](examples/tool_calling_chat_mcp_http.py)


---

<a id="structured-output"></a>

## 🌙 Structured Output

chak's `Conversation` supports structured outputs through the `returns` parameter. Instead of parsing LLM text responses manually, you can specify a Pydantic model and get validated, type-safe data directly.

### Basic Usage

#### Simple Data Extraction

```python
from pydantic import BaseModel, Field
from chak import Conversation

class User(BaseModel):
    """User information"""
    name: str = Field(description="User's full name")
    email: str = Field(description="User's email address")
    age: int = Field(description="User's age")

conv = Conversation("openai/gpt-4o", api_key="YOUR_KEY")

# Extract structured data from natural language
user = await conv.asend(
    "Create a user profile for John Doe, email john@example.com, 30 years old",
    returns=User
)

print(user.name)   # "John Doe"
print(user.email)  # "john@example.com"
print(user.age)    # 30
```

#### Complex Nested Models

```python
from typing import List
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    industry: str
    address: Address
    employee_count: int

# Works with nested structures
company = await conv.asend(
    "Apple Inc is a technology company with 150,000 employees, located at One Apple Park Way, Cupertino, USA",
    returns=Company
)

print(company.name)              # "Apple Inc"
print(company.address.city)      # "Cupertino"
print(company.employee_count)    # 150000
```

### Multimodal Structured Output

Combine structured outputs with images, documents, and other attachments:

#### Extract Data from Images

```python
from chak import Image

class SceneDescription(BaseModel):
    """Scene description extracted from image"""
    main_subject: str = Field(description="The main subject or focal point")
    setting: str = Field(description="The location or setting")
    colors: List[str] = Field(description="Dominant colors in the image")
    mood: str = Field(description="Overall mood or atmosphere")

# Analyze image and get structured output
scene = await conv.asend(
    "Analyze this image and describe the scene",
    attachments=[Image("photo.jpg")],
    returns=SceneDescription
)

print(scene.main_subject)  # "Mount Fuji"
print(scene.colors)        # ["blue", "white", "pink"]
print(scene.mood)          # "peaceful and serene"
```

#### Extract Data from Documents

```python
from chak import PDF

class Invoice(BaseModel):
    """Invoice information extracted from document"""
    invoice_number: str
    date: str
    total_amount: float
    vendor_name: str
    items: List[str]

# Extract structured data from PDF
invoice = await conv.asend(
    "Extract invoice information from this document",
    attachments=[PDF("invoice.pdf")],
    returns=Invoice
)

print(invoice.invoice_number)  # "INV-2024-001"
print(invoice.total_amount)    # 1250.00
print(invoice.vendor_name)     # "Acme Corp"
```

### Complete Example

See full working examples:
- **Basic Structured Output**: [examples/structured_output_simple.py](examples/structured_output_simple.py)
- **Multimodal Structured Output**: [examples/structured_output_multimodal.py](examples/structured_output_multimodal.py)

### Notes

- **Pydantic Required**: The `returns` parameter must be a Pydantic `BaseModel` subclass
- **Function Calling Support**: Your LLM must support function calling (most modern models do)
- **Async Only**: Structured output currently works with `asend()` only, not `send()`
- **Validation**: All data is automatically validated against your Pydantic model schema
- **Provider Compatibility**: 
  - ✅ Supported: OpenAI, Anthropic, Google Gemini, most text models
  - ⚠️ Limited: Some vision models may not support function calling
  - Use text models with multimodal support (e.g., OpenAI gpt-4o, gpt-4-vision) for best results

---

<a id="multimodal-support"></a>

## 🌔 Multimodal Support

chak's `Conversation` supports multimodal inputs through the `attachments` parameter. You can send images, audio, video, documents (PDF, Word, Excel, CSV, TXT), and web links alongside your text messages.

### Supported File Types

| Type | Class | Supported Formats | Use Cases |
|------|-------|-------------------|------------|
| **Image** | `Image` | JPEG, PNG, GIF, WEBP | Image analysis, visual Q&A, OCR |
| **Audio** | `Audio` | WAV, MP3, OGG | Speech recognition, audio analysis |
| **Video** | `Video` | MP4, WEBM | Video understanding, frame extraction |
| **PDF** | `PDF` | PDF | Document analysis, extraction |
| **Word** | `DOC` | DOC, DOCX | Document reading, content extraction |
| **Excel** | `Excel` | XLS, XLSX | Data analysis, spreadsheet processing |
| **CSV** | `CSV` | CSV | Structured data analysis |
| **Text** | `TXT` | TXT, MD, etc. | Plain text/markdown analysis |
| **Link** | `Link` | HTTP/HTTPS URLs | Web content analysis |

### Input Format Flexibility

All attachment types support **three input formats**:

1. **Local file path**: `Image("./photo.jpg")`
2. **Remote URL**: `Image("https://example.com/photo.jpg")`
3. **Base64 data URI**: `Image("data:image/jpeg;base64,/9j/4AAQ...")`

### Basic Usage

#### Single Image

```python
from chak import Conversation, Image

conv = Conversation("openai/gpt-4o", api_key="YOUR_KEY")

# Using URL
response = await conv.asend(
    "What's in this image?",
    attachments=[Image("https://example.com/photo.jpg")]
)

# Using local path
response = await conv.asend(
    "Describe this image",
    attachments=[Image("./local/photo.png")]
)

# Using base64
response = await conv.asend(
    "Analyze this",
    attachments=[Image("data:image/jpeg;base64,/9j/4AAQSkZJRg...")]
)
```

#### Multiple Images

```python
from chak import Image, MimeType

# Compare multiple images
response = await conv.asend(
    "What are the differences between these images?",
    attachments=[
        Image("https://example.com/image1.jpg"),
        Image("./local/image2.png", MimeType.PNG),
        Image("data:image/webp;base64,...", MimeType.WEBP)
    ]
)
```

#### Audio Files

```python
from chak import Audio, MimeType

response = await conv.asend(
    "What is being said in this audio?",
    attachments=[Audio("https://example.com/speech.wav", MimeType.WAV)]
)
```

#### Documents

```python
from chak import PDF, DOC, Excel, CSV, TXT

# PDF analysis
response = await conv.asend(
    "Summarize this PDF document",
    attachments=[PDF("./report.pdf")],
    timeout=120  # Longer timeout for large files
)

# Word document
response = await conv.asend(
    "Extract key points from this document",
    attachments=[DOC("https://example.com/document.docx")]
)

# Excel spreadsheet
response = await conv.asend(
    "What's the total revenue in this spreadsheet?",
    attachments=[Excel("./sales_data.xlsx")]
)

# CSV data
response = await conv.asend(
    "Find all customers from California",
    attachments=[CSV("./customers.csv")]
)

# Plain text or markdown
response = await conv.asend(
    "Summarize this article",
    attachments=[TXT("https://example.com/article.md")]
)
```

#### Web Links

```python
from chak import Link

# Analyze web content
response = await conv.asend(
    "What are the main points in this article?",
    attachments=[Link("https://example.com/article")]
)
```

### Streaming with Attachments

Multimodal inputs work seamlessly with streaming:

```python
from chak import Image

print("Response: ", end="")
async for chunk in await conv.asend(
    "Describe this image in detail",
    attachments=[Image("photo.jpg")],
    stream=True
):
    print(chunk.content, end="", flush=True)
```

### Advanced: Direct Multimodal Message

For fine-grained control, construct multimodal messages directly:

```python
from chak import HumanMessage

response = await conv.asend(
    HumanMessage(content=[
        {"type": "text", "text": "What colors are in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ])
)
```

### Complete Examples

See full working examples:

- **Images**: [examples/multimodal_chat_image.py](examples/multimodal_chat_image.py)
  - Single image analysis
  - Multiple image comparison
  - Streaming with images
  - Audio input (when supported)
  - Advanced multimodal messages

- **Documents**: [examples/multimodal_chat_documents.py](examples/multimodal_chat_documents.py)
  - PDF document analysis
  - Word document processing
  - Plain text and markdown files
  - CSV data analysis
  - Excel spreadsheet processing
  - Web link content analysis
  - Streaming with documents

### Notes

- **Model Support**: Not all LLM providers support all modalities. Check your provider's documentation:
  - Vision models: OpenAI GPT-4o, Anthropic Claude 3, Google Gemini, Bailian Qwen-VL
  - Audio models: Some Qwen variants, Whisper-based models
  - Document support varies by provider

- **File Size**: Large files may require longer timeouts. Use `timeout` parameter:
  ```python
  response = await conv.asend(
      "Analyze this large PDF",
      attachments=[PDF("large.pdf")],
      timeout=180  # 3 minutes
  )
  ```

- **Custom Readers**: Built-in readers cover most use cases. For specialized needs, you can provide custom reader functions to document attachment types (PDF, DOC, Excel, etc.).

- **Async Required**: Multimodal support works with both `send()` and `asend()`, but async is recommended for better performance with large files.

---

## 🌗 Practical Utilities

### View Conversation Statistics

```python
stats = conv.stats()
print(stats)
# {
#     'total_messages': 10,
#     'by_type': {'user': 5, 'assistant': 4, 'context': 1},
#     'total_tokens': '12.5K',
#     'input_tokens': '8.2K',
#     'output_tokens': '4.3K'
# }
```

### Debug Mode

Set environment variables to see internal execution details:

```bash
export CHAK_LOG_LEVEL=DEBUG
python your_script.py
```

chak will output detailed logs for:
- **Context strategies**: trigger points, retention intervals, summary previews, token counts
- **Tool calls**: tool invocation, request/response details, execution results

---

## Local Server Mode (Optional)

Start a local gateway service with 2 lines of code:

### 1. Create Configuration File

```yaml
# chak-config.yaml
api_keys:
  # Simple format - use default base_url
  openai: ${OPENAI_API_KEY}           # Read from environment variable (recommended)
  bailian: "sk-your-api-key-here"    # Plain text (for development/testing)
  
  # Custom base_url (requires quotes)
  "ollama@http://localhost:11434": "ollama"
  "vllm@http://192.168.1.100:8000": "dummy-key"

server:
  host: "0.0.0.0"
  port: 8000
```

### 2. Start Server

```python
import chak

chak.serve('chak-config.yaml')
```

That's it! The server starts and you'll see:

```
======================================================================

  ✨ Chak AI Gateway
  A simple, yet handy, LLM gateway

======================================================================

  🚀 Server running at:     http://localhost:8000
  🎮 Playground:            http://localhost:8000/playground
  📡 WebSocket endpoint:    ws://localhost:8000/ws/conversation

  ⭐ Star on GitHub:        https://github.com/zhixiangxue/chak-ai

======================================================================
```

### 3. Use Playground for Quick Model Conversations

Open `http://localhost:8000/playground`, select a provider and model, start chatting immediately. Experience real-time interaction with global LLMs.

### 4. Call from Any Language

The service provides a WebSocket API, callable from JavaScript, Go, Java, Rust, or any language:

```javascript
// JavaScript example
const ws = new WebSocket('ws://localhost:8000/ws/conversation');

// Initialize session
ws.send(JSON.stringify({
  type: 'init',
  model_uri: 'openai/gpt-4o-mini'
}));

// Send message
ws.send(JSON.stringify({
  type: 'send',
  message: 'Hello!',
  stream: true
}));
```

This way chak becomes your local LLM gateway, centrally managing all provider API keys, callable from any language.

---

## Supported LLM Providers

| Provider | Registration | URI Example |
|----------|-------------|-------------|
| OpenAI | https://platform.openai.com | `openai/gpt-4o` |
| Anthropic | https://console.anthropic.com | `anthropic/claude-3-5-sonnet` |
| Google Gemini | https://ai.google.dev | `google/gemini-1.5-pro` |
| DeepSeek | https://platform.deepseek.com | `deepseek/deepseek-chat` |
| Alibaba Bailian | https://bailian.console.aliyun.com | `bailian/qwen-max` |
| Zhipu GLM | https://open.bigmodel.cn | `zhipu/glm-4` |
| Moonshot | https://platform.moonshot.cn | `moonshot/moonshot-v1-8k` |
| Baidu Wenxin | https://console.bce.baidu.com/qianfan | `baidu/ernie-bot-4` |
| Tencent Hunyuan | https://cloud.tencent.com/product/hunyuan | `tencent/hunyuan-standard` |
| ByteDance Doubao | https://console.volcengine.com/ark | `volcengine/doubao-pro` |
| iFlytek Spark | https://xinghuo.xfyun.cn | `iflytek/spark-v3.5` |
| MiniMax | https://platform.minimaxi.com | `minimax/abab-5.5` |
| Mistral | https://console.mistral.ai | `mistral/mistral-large` |
| xAI Grok | https://console.x.ai | `xai/grok-beta` |
| SiliconFlow | https://siliconflow.cn | `siliconflow/qwen-7b` |
| Azure OpenAI | https://azure.microsoft.com/en-us/products/ai-services/openai-service | `azure/gpt-4o` |
| Ollama | https://ollama.com | `ollama/llama3.1` |
| vLLM | https://github.com/vllm-project/vllm | `vllm/custom-model` |

**Notes:**
- URI format: `provider/model`
- Custom base_url: Use complete format `provider@base_url:model`
- Local deployments (Ollama, vLLM) require custom base_url configuration

---

##  MCP Server Resources

Explore thousands of ready-to-use MCP servers:

| Platform | Description | URL |
|----------|-------------|-----|
| **Mcp.so** | 8,000+ servers, supports STDIO & SSE, with API playground | https://mcp.so |
| **Smithery** | 4,500+ servers, beginner-friendly, one-click config for Cursor | https://smithery.ai |
| **Alibaba Bailian** | Enterprise-grade MCP marketplace with cloud-hosted services | https://bailian.console.aliyun.com/?tab=mcp#/mcp-market |
| **ModelScope** | Largest Chinese MCP community by Alibaba Cloud | https://modelscope.cn/mcp |
| **Awesome MCP** | 200+ curated servers organized by category (GitHub) | https://github.com/punkpeye/awesome-mcp-servers |
| **ByteDance Volcengine** | Enterprise-level stable and secure MCP services | https://www.volcengine.com/mcp-marketplace |
| **iFlytek Spark** | MCP servers for Spark AI platform | https://mcp.xfyun.cn |
| **Baidu SAI** | Explore massive available MCP servers | https://sai.baidu.com/mcp |
| **PulseMCP** | 3,290+ servers with weekly updates and tutorials | https://www.pulsemcp.com |
| **mcp.run** | 200+ templates with one-click web deployment | https://www.mcp.run |



## 🌕 Is chak for You?

If you:
- Need to connect to multiple model platforms
- Want simple, automatic context management
- Want the simplest tool calling experience - just pass functions or objects or mcp tools
- Want to focus on building applications, not wrestling with context and tools

Then chak is made for you.

<div align="right"><a href="https://youtube.com/watch?v=xOKQ7EQcggw"><img src="https://raw.githubusercontent.com/zhixiangxue/chak-ai/main/docs/assets/logo.png" alt="Demo Video" width="120"></a></div>