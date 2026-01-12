<div align="center">

<a href="https://youtube.com/watch?v=xOKQ7EQcggw"><img src="https://raw.githubusercontent.com/zhixiangxue/chak-ai/main/docs/assets/logo.png" alt="Demo Video" width="120"></a>

[![PyPI version](https://badge.fury.io/py/chakpy.svg)](https://badge.fury.io/py/chakpy)
[![Python Version](https://img.shields.io/pypi/pyversions/chakpy)](https://pypi.org/project/chakpy/)
[![License](https://img.shields.io/github/license/zhixiangxue/chak-ai)](https://github.com/zhixiangxue/chak-ai/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/chakpy)](https://pypi.org/project/chakpy/)
[![GitHub Stars](https://img.shields.io/github/stars/zhixiangxue/chak-ai?style=social)](https://github.com/zhixiangxue/chak-ai)

[English](README.md) | [中文](docs/README_CN.md)

**一个内置上下文管理和灵活工具调用的多模型 LLM 客户端。**

chak 不是另一个 liteLLM、one-api 或 OpenRouter，而是一个为你主动管理对话上下文和工具调用的客户端库。你只需专注于构建应用，让 chak 处理复杂性。

</div>

<div align="center">

![Demo Video](https://raw.githubusercontent.com/zhixiangxue/chak-ai/main/docs/assets/demo.gif)

</div>

---

## 🌵 最近更新

- **2025-01-12 | v0.2.6** - 新增事件流支持，实现工具调用的实时可观测性。使用 `event=True` 在 UI 中观察工具执行。详见 [工具调用可观测性](#tool-call-observability)
- **2025-01-09 | v0.2.5** - 新增可配置工具执行器，支持 CPU 密集型任务。使用 `tool_executor` 参数控制执行模式。详见 [工具调用](#tool-calling)
- **2025-01-07 | v0.2.3** - Conversation 现已支持通过 `returns` 参数输出结构化数据。详见 [结构化输出](#structured-output)
- **2024-12-02 | v0.2.2** - Conversation 现已支持多模态对话。详见 [多模态支持](#multimodal-support)

---

## 核心特性

### 🌱 极简 API 设计

没有复杂的配置，没有学习曲线。chak 设计得直观易懂：

```python
# 作为 SDK 使用 - 通过简单 URI 连接任何 LLM
conv = chak.Conversation("openai/gpt-4o-mini", api_key="YOUR_KEY")
response = conv.send("Hello!")

# 或作为本地网关运行 - 2 行代码启动
import chak
chak.serve('chak-config.yaml')
```

无论你是构建应用还是运行网关,chak 都保持简单。

### 🌳 多模态对话

对话支持多模态输入——图片、音频、视频和文档。只需传递附件（就像我们使用聊天工具时一样自然）：

```python
from chak import Image, PDF, Audio

# 发送图片并提问
response = await conv.asend(
    "这张图片里有什么？",
    attachments=[Image("photo.jpg")]  # 本地路径、URL 或 base64
)

# 分析文档
response = await conv.asend(
    "总结这份文档",
    attachments=[PDF("report.pdf")]
)

# 一次发送多个附件
response = await conv.asend(
    "比较这些图片",
    attachments=[
        Image("https://example.com/img1.jpg"),
        Image("./local/img2.png")
    ]
)
```

支持图片、音频、视频、PDF、Word、Excel、CSV、TXT 和网页链接。详见 [多模态支持](#multimodal-support)。

### 🪴 可插拔的上下文管理

Chak 自动处理上下文，提供多种策略：

```python
# 上下文自动管理
conv = chak.Conversation(
    "openai/gpt-4o",
    context_strategy=chak.FIFOStrategy(keep_recent_turns=5)
)
```

- **当前**: 短期记忆策略（FIFO、摘要、LRU）- 生产就绪
- **规划中**: 长期记忆（RAG、记忆库）- 让对话真正"有记忆"

没有其他库在这个级别自动化上下文管理。chak 的策略模式使其完全可插拔和可扩展。

### 🌻 简单的工具调用

用你喜欢的方式编写工具——函数、对象或 MCP 服务器，chak 处理其余部分：

```python
# 函数
def get_weather(city: str) -> str:
    ...

# 对象
class ShoppingCart:
    def add_item(self, name: str, price: float): ...
    def get_total(self) -> float: ...

cart = ShoppingCart()

# MCP 服务器
from chak.tools.mcp import Server
mcp_tools = await Server(url="...").tools()

# 使用它们，就这样简单
conv = Conversation(
    "openai/gpt-4o",
    tools=[get_weather, cart, *mcp_tools]
)
```

<a id="tool-call-observability"></a>

**实时可观测性**：通过事件流实时获取工具执行情况：

```python
from chak.message import MessageChunk, ToolCallStartEvent, ToolCallSuccessEvent, ToolCallErrorEvent

# 使用 event=True 实时观察工具调用
async for event in await conv.asend("计算 15 + 27", event=True):
    match event:
        case ToolCallStartEvent(tool_name=name, arguments=args):
            print(f"🔧 正在调用: {name}，参数 {args}")
        
        case ToolCallSuccessEvent(tool_name=name, result=res):
            print(f"✅ 结果: {name} -> {res}")
        
        case ToolCallErrorEvent(tool_name=name, error=err):
            print(f"❌ 失败: {name} - {err}")
        
        case MessageChunk(content=text, is_final=final):
            print(text, end="", flush=True)
```

非常适合构建展示工具实时执行进度的 UI。详见 [examples/event_stream_chat_demo.py](../examples/event_stream_chat_demo.py)

**可配置执行**：对于 CPU 密集型工具，使用 `tool_executor` 控制工具执行方式：

```python
import chak

# 默认：适合 IO 密集型任务（API 调用、数据库查询）
conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[...],
    tool_executor=chak.ToolExecutor.ASYNCIO  # 默认值
)

# CPU 密集型任务：使用进程池实现真正的并行
conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[heavy_compute, ...],
    tool_executor=chak.ToolExecutor.PROCESS  # 绕过 GIL
)

# 随时切换
conv.set_tool_executor(chak.ToolExecutor.PROCESS)

# 或在单次调用时覆盖
await conv.asend("运行重型任务", tool_executor=chak.ToolExecutor.PROCESS)
```

**选择合适的执行器**：

| 场景 | ASYNCIO | THREAD | PROCESS | 推荐 |
|------|---------|--------|---------|------|
| **CPU 密集型（同步）** | ❌ GIL 限制 | ❌ GIL 限制 | ✅ 真正并行 | PROCESS |
| **IO 密集型（异步）** | ✅ 天然并发 | - | - | 默认值 |
| **IO 密集型（同步）** | ✅ 运行良好 | ✅ 运行良好 | ⚠️ 过度使用 | ASYNCIO |

完整示例：[examples/tool_calling_parallel_demo.py](../examples/tool_calling_parallel_demo.py)

- **当前**：函数、对象和 MCP 工具都以相同方式工作
- **当前**：可配置执行器以获得最佳性能
- **规划中**：基于上下文的智能工具选择

### 🌺 结构化输出

使用 Pydantic 模型直接从 LLM 响应中获取结构化数据：

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(description="用户全名")
    email: str = Field(description="用户邮箱")
    age: int = Field(description="用户年龄")

# 自动获取结构化输出
user = await conv.asend(
    "创建用户：张三，zhangsan@example.com，30岁",
    returns=User
)

print(user.name)   # "张三"
print(user.email)  # "zhangsan@example.com"
print(user.age)    # 30
```

也支持多模态输入——从图片、文档等提取结构化数据。

---

## 集成提供商 (18+)

OpenAI、Google Gemini、Azure OpenAI、Anthropic Claude、阿里巴巴百炼、百度文心、腾讯混元、字节跳动豆包、智谱 GLM、Moonshot、DeepSeek、科大讯飞星火、MiniMax、Mistral、SiliconFlow、xAI Grok、Ollama、vLLM 等。

---

## 🌖 快速开始

### 安装

```bash
# 基础安装（仅 SDK）
pip install chakpy

# 带服务器支持
pip install chakpy[server]

# 安装所有可选依赖
pip install chakpy[all]
```

### 几行代码与全球模型聊天

```python
import chak

conv = chak.Conversation(
    "openai/gpt-4o-mini",
    api_key="YOUR_KEY"
)

resp = conv.send("用一句话解释上下文管理")
print(resp.content)
```

chak 处理：连接初始化、消息对齐、重试逻辑、上下文管理、模型格式转换……你只需 `send` 消息。

---

## 🌒 启用自动上下文管理

三种内置策略：

- FIFO：保留最近 N 轮对话，自动丢弃较早的。
- 摘要：当上下文达到阈值时，早期历史被摘要；近期对话保持完整。
- LRU：基于摘要构建，保留热门话题并修剪冷门内容。

快速开始：

```python
from chak import Conversation, FIFOStrategy

conv = Conversation(
    "bailian/qwen-flash",
    api_key="YOUR_KEY",
    context_strategy=FIFOStrategy(keep_recent_turns=3)
)
```

查看完整示例(参数、工作原理、技巧):

- FIFO: [examples/strategy_chat_fifo.py](examples/strategy_chat_fifo.py)
- 摘要: [examples/strategy_chat_summarization.py](examples/strategy_chat_summarization.py)
- LRU: [examples/strategy_chat_lru.py](examples/strategy_chat_lru.py)

---

<a id="tool-calling"></a>

## 🌓 工具调用

用你喜欢的方式编写工具——函数、对象或 MCP 服务器。chak 处理其余部分。

### 传递函数

只需传递常规 Python 函数：

```python
from datetime import datetime

def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(a: int, b: int, operation: str = "add") -> int:
    """对两个数字执行计算"""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    # ...

conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[get_current_time, calculate]
)

response = await conv.asend("现在几点了？然后计算 50 乘以 20")
```

### 传递对象

传递 Python 对象，它们的方法成为工具。对象状态在调用间持久化：

```python
class ShoppingCart:
    def __init__(self):
        self.items = []
        self.discount = 0
    
    def add_item(self, name: str, price: float, quantity: int = 1):
        """添加商品到购物车"""
        self.items.append({"name": name, "price": price, "quantity": quantity})
    
    def apply_discount(self, percent: float):
        """应用折扣百分比"""
        self.discount = percent
    
    def get_total(self) -> float:
        """计算总价"""
        subtotal = sum(item["price"] * item["quantity"] for item in self.items)
        return subtotal * (1 - self.discount / 100)

cart = ShoppingCart()

conv = chak.Conversation(
    "openai/gpt-4o",
    tools=[cart]  # 直接传递对象！
)

# LLM 通过自然语言修改购物车状态！
response = await conv.asend(
    "添加 2 部 iPhone，每部 999 美元，然后应用 10% 折扣并告诉我总价"
)

print(cart.items)     # [{'name': 'iPhone', 'price': 999, 'quantity': 2}]
print(cart.discount)  # 10
print(cart.get_total())  # 1798.2
```

Chak 帮你维护对象的状态。

### 传递 MCP 工具

chak 集成了 https://modelcontextprotocol.io/：

```python
import asyncio
from chak import Conversation
from chak.tools.mcp import Server

async def main():
    # 连接到 MCP 服务器并加载工具
    tools = await Server(
        url="https://your-mcp-server.com/sse",
        headers={"Authorization": "Bearer YOUR_TOKEN"}
    ).tools()
    
    # 创建带工具的对话
    conv = Conversation(
        "openai/gpt-4o",
        api_key="YOUR_KEY",
        tools=tools
    )
    
    # 模型在需要时自动调用工具
    response = await conv.asend("旧金山天气怎么样？")
    print(response.content)

asyncio.run(main())
```

支持三种传输类型：

- **SSE**（服务器发送事件）：云托管的 MCP 服务
- **stdio**: 本地 MCP 服务器
- **HTTP**: 基于 HTTP 的 MCP 服务

### 混合使用所有内容

函数、对象和 MCP 工具协同工作：

```python
def send_email(to: str, subject: str): ...

class OrderWorkflow:
    def add_items(self, items): ...
    def submit_order(self): ...

mcp_tools = await Server(url="...").tools()  # 外部工具

conv = Conversation(
    "openai/gpt-4o",
    tools=[
        send_email,           # 原生函数
        OrderWorkflow(),      # 原生对象（有状态！）
        *mcp_tools           # MCP 工具
    ]
)
```

### 示例

查看完整示例：

- **原生函数**: examples/tool_calling_chat_functions.py
- **有状态对象**: examples/tool_calling_chat_objects_stateful.py
- **事件流（可观测性）**: examples/event_stream_chat_demo.py
- **MCP (SSE)**: examples/tool_calling_chat_mcp_sse.py
- **MCP (stdio)**: examples/tool_calling_chat_mcp_stdio.py
- **MCP (HTTP)**: examples/tool_calling_chat_mcp_http.py

---

<a id="structured-output"></a>

## 🌙 结构化输出

chak 的 `Conversation` 通过 `returns` 参数支持结构化输出。无需手动解析 LLM 文本响应，你可以指定一个 Pydantic 模型，直接获取验证过的、类型安全的数据。

### 基本用法

#### 简单数据提取

```python
from pydantic import BaseModel, Field
from chak import Conversation

class User(BaseModel):
    """用户信息"""
    name: str = Field(description="用户全名")
    email: str = Field(description="用户邮箱地址")
    age: int = Field(description="用户年龄")

conv = Conversation("openai/gpt-4o", api_key="YOUR_KEY")

# 从自然语言提取结构化数据
user = await conv.asend(
    "为张三创建用户资料，邮箱 zhangsan@example.com，30 岁",
    returns=User
)

print(user.name)   # "张三"
print(user.email)  # "zhangsan@example.com"
print(user.age)    # 30
```

#### 复杂嵌套模型

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

# 支持嵌套结构
company = await conv.asend(
    "苹果公司是一家科技公司，有 15 万名员工，位于美国库比蒂诺苹果园区",
    returns=Company
)

print(company.name)              # "苹果公司" 或 "Apple Inc"
print(company.address.city)      # "库比蒂诺" 或 "Cupertino"
print(company.employee_count)    # 150000
```

### 多模态结构化输出

结合结构化输出与图片、文档和其他附件：

#### 从图片提取数据

```python
from chak import Image

class SceneDescription(BaseModel):
    """从图片提取的场景描述"""
    main_subject: str = Field(description="主要主体或焦点")
    setting: str = Field(description="位置或场景")
    colors: List[str] = Field(description="图片中的主要颜色")
    mood: str = Field(description="整体气氛或心情")

# 分析图片并获取结构化输出
scene = await conv.asend(
    "分析这张图片并描述场景",
    attachments=[Image("photo.jpg")],
    returns=SceneDescription
)

print(scene.main_subject)  # "富士山"
print(scene.colors)        # ["蓝色", "白色", "粉色"]
print(scene.mood)          # "宁静和平">
```

#### 从文档提取数据

```python
from chak import PDF

class Invoice(BaseModel):
    """从文档提取的发票信息"""
    invoice_number: str
    date: str
    total_amount: float
    vendor_name: str
    items: List[str]

# 从 PDF 提取结构化数据
invoice = await conv.asend(
    "从这份文档提取发票信息",
    attachments=[PDF("invoice.pdf")],
    returns=Invoice
)

print(invoice.invoice_number)  # "INV-2024-001"
print(invoice.total_amount)    # 1250.00
print(invoice.vendor_name)     # "Acme 公司"
```

### 完整示例

查看完整可运行示例：
- **基础结构化输出**: [examples/structured_output_simple.py](examples/structured_output_simple.py)
- **多模态结构化输出**: [examples/structured_output_multimodal.py](examples/structured_output_multimodal.py)

### 注意事项

- **需要 Pydantic**：`returns` 参数必须是 Pydantic `BaseModel` 子类
- **Function Calling 支持**：你的 LLM 必须支持 function calling（大多数现代模型都支持）
- **仅异步**：结构化输出目前仅适用于 `asend()`，不适用于 `send()`
- **自动验证**：所有数据都会根据你的 Pydantic 模型 schema 自动验证
- **提供商兼容性**：
  - ✅ 支持：OpenAI、Anthropic、Google Gemini、大多数文本模型
  - ⚠️ 限制：某些视觉模型可能不支持 function calling
  - 建议使用支持多模态的文本模型（如 OpenAI gpt-4o、gpt-4-vision）以获得最佳效果

---

<a id="multimodal-support"></a>

## 🌔 多模态支持

chak 的 `Conversation` 通过 `attachments` 参数支持多模态输入。你可以在发送文本消息的同时发送图片、音频、视频、文档（PDF、Word、Excel、CSV、TXT）和网页链接。

### 支持的文件类型

| 类型 | 类名 | 支持格式 | 使用场景 |
|------|-------|-------------------|------------|
| **图片** | `Image` | JPEG, PNG, GIF, WEBP | 图像分析、视觉问答、OCR |
| **音频** | `Audio` | WAV, MP3, OGG | 语音识别、音频分析 |
| **视频** | `Video` | MP4, WEBM | 视频理解、帧提取 |
| **PDF** | `PDF` | PDF | 文档分析、内容提取 |
| **Word** | `DOC` | DOC, DOCX | 文档阅读、内容提取 |
| **Excel** | `Excel` | XLS, XLSX | 数据分析、电子表格处理 |
| **CSV** | `CSV` | CSV | 结构化数据分析 |
| **文本** | `TXT` | TXT, MD 等 | 纯文本/Markdown 分析 |
| **链接** | `Link` | HTTP/HTTPS URLs | 网页内容分析 |

### 输入格式灵活性

所有附件类型支持 **三种输入格式**：

1. **本地文件路径**: `Image("./photo.jpg")`
2. **远程 URL**: `Image("https://example.com/photo.jpg")`
3. **Base64 数据 URI**: `Image("data:image/jpeg;base64,/9j/4AAQ...")`

### 基础用法

#### 单张图片

```python
from chak import Conversation, Image

conv = Conversation("openai/gpt-4o", api_key="YOUR_KEY")

# 使用 URL
response = await conv.asend(
    "这张图片里有什么？",
    attachments=[Image("https://example.com/photo.jpg")]
)

# 使用本地路径
response = await conv.asend(
    "描述这张图片",
    attachments=[Image("./local/photo.png")]
)

# 使用 base64
response = await conv.asend(
    "分析这个",
    attachments=[Image("data:image/jpeg;base64,/9j/4AAQSkZJRg...")]
)
```

#### 多张图片

```python
from chak import Image, MimeType

# 比较多张图片
response = await conv.asend(
    "这些图片之间有什么区别？",
    attachments=[
        Image("https://example.com/image1.jpg"),
        Image("./local/image2.png", MimeType.PNG),
        Image("data:image/webp;base64,...", MimeType.WEBP)
    ]
)
```

#### 音频文件

```python
from chak import Audio, MimeType

response = await conv.asend(
    "这段音频在说什么？",
    attachments=[Audio("https://example.com/speech.wav", MimeType.WAV)]
)
```

#### 文档

```python
from chak import PDF, DOC, Excel, CSV, TXT

# PDF 分析
response = await conv.asend(
    "总结这份 PDF 文档",
    attachments=[PDF("./report.pdf")],
    timeout=120  # 大文件需要更长超时时间
)

# Word 文档
response = await conv.asend(
    "从这份文档中提取要点",
    attachments=[DOC("https://example.com/document.docx")]
)

# Excel 电子表格
response = await conv.asend(
    "这份电子表格中的总收入是多少？",
    attachments=[Excel("./sales_data.xlsx")]
)

# CSV 数据
response = await conv.asend(
    "找出所有来自加州的客户",
    attachments=[CSV("./customers.csv")]
)

# 纯文本或 Markdown
response = await conv.asend(
    "总结这篇文章",
    attachments=[TXT("https://example.com/article.md")]
)
```

#### 网页链接

```python
from chak import Link

# 分析网页内容
response = await conv.asend(
    "这篇文章的主要观点是什么？",
    attachments=[Link("https://example.com/article")]
)
```

### 流式响应与附件

多模态输入与流式响应无缝配合：

```python
from chak import Image

print("响应: ", end="")
async for chunk in await conv.asend(
    "详细描述这张图片",
    attachments=[Image("photo.jpg")],
    stream=True
):
    print(chunk.content, end="", flush=True)
```

### 高级用法：直接构造多模态消息

如需精细控制，可以直接构造多模态消息：

```python
from chak import HumanMessage

response = await conv.asend(
    HumanMessage(content=[
        {"type": "text", "text": "这张图片中有什么颜色？"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ])
)
```

### 完整示例

查看完整可运行示例：

- **图片**: [examples/multimodal_chat_image.py](examples/multimodal_chat_image.py)
  - 单张图片分析
  - 多张图片比较
  - 流式响应与图片
  - 音频输入（如果支持）
  - 高级多模态消息

- **文档**: [examples/multimodal_chat_documents.py](examples/multimodal_chat_documents.py)
  - PDF 文档分析
  - Word 文档处理
  - 纯文本和 Markdown 文件
  - CSV 数据分析
  - Excel 电子表格处理
  - 网页链接内容分析
  - 流式响应与文档

### 注意事项

- **模型支持**：并非所有 LLM 提供商都支持所有模态。请查看提供商文档：
  - 视觉模型：OpenAI GPT-4o、Anthropic Claude 3、Google Gemini、百炼 Qwen-VL
  - 音频模型：部分 Qwen 变体、基于 Whisper 的模型
  - 文档支持因提供商而异

- **文件大小**：大文件可能需要更长的超时时间。使用 `timeout` 参数：
  ```python
  response = await conv.asend(
      "分析这份大型 PDF",
      attachments=[PDF("large.pdf")],
      timeout=180  # 3 分钟
  )
  ```

- **自定义读取器**：内置读取器可满足大多数需求。对于特殊需求，你可以为文档附件类型（PDF、DOC、Excel 等）提供自定义读取器函数。

- **建议使用异步**：多模态支持在 `send()` 和 `asend()` 中都可用，但建议使用异步以获得更好的大文件处理性能。

---

## 🌗 实用工具

### 查看对话统计信息

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

### 调试模式

设置环境变量查看内部执行详情：

```bash
export CHAK_LOG_LEVEL=DEBUG
python your_script.py
```

chak 将输出详细日志：
- **上下文策略**: 触发点、保留间隔、摘要预览、令牌计数
- **工具调用**: 工具调用、请求/响应详情、执行结果

---

## 本地服务器模式（可选）

用 2 行代码启动本地网关服务：

### 1. 创建配置文件

```yaml
# chak-config.yaml
api_keys:
  # 简单格式 - 使用默认 base_url
  openai: ${OPENAI_API_KEY}           # 从环境变量读取（推荐）
  bailian: "sk-your-api-key-here"    # 明文（用于开发/测试）
  
  # 自定义 base_url（需要引号）
  "ollama@http://localhost:11434": "ollama"
  "vllm@http://192.168.1.100:8000": "dummy-key"

server:
  host: "0.0.0.0"
  port: 8000
```

### 2. 启动服务器

```python
import chak

chak.serve('chak-config.yaml')
```

就这样！服务器启动后你会看到：

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

### 3. 使用 Playground 快速与模型对话

打开 `http://localhost:8000/playground`，选择提供商和模型，立即开始聊天。体验与全球 LLM 的实时交互。

### 4. 从任何语言调用

该服务提供 WebSocket API，可从 JavaScript、Go、Java、Rust 或任何语言调用：

```javascript
// JavaScript 示例
const ws = new WebSocket('ws://localhost:8000/ws/conversation');

// 初始化会话
ws.send(JSON.stringify({
  type: 'init',
  model_uri: 'openai/gpt-4o-mini'
}));

// 发送消息
ws.send(JSON.stringify({
  type: 'send',
  message: 'Hello!',
  stream: true
}));
```

这样 chak 就成为你的本地 LLM 网关，集中管理所有提供商 API 密钥，可从任何语言调用。

---

## 支持的 LLM 提供商

| 提供商 | 注册地址 | URI 示例 |
|----------|-------------|-------------|
| OpenAI | https://platform.openai.com | `openai/gpt-4o` |
| Anthropic | https://console.anthropic.com | `anthropic/claude-3-5-sonnet` |
| Google Gemini | https://ai.google.dev | `google/gemini-1.5-pro` |
| DeepSeek | https://platform.deepseek.com | `deepseek/deepseek-chat` |
| 阿里巴巴百炼 | https://bailian.console.aliyun.com | `bailian/qwen-max` |
| 智谱 GLM | https://open.bigmodel.cn | `zhipu/glm-4` |
| Moonshot | https://platform.moonshot.cn | `moonshot/moonshot-v1-8k` |
| 百度文心 | https://console.bce.baidu.com/qianfan | `baidu/ernie-bot-4` |
| 腾讯混元 | https://cloud.tencent.com/product/hunyuan | `tencent/hunyuan-standard` |
| 字节跳动豆包 | https://console.volcengine.com/ark | `volcengine/doubao-pro` |
| 科大讯飞星火 | https://xinghuo.xfyun.cn | `iflytek/spark-v3.5` |
| MiniMax | https://platform.minimaxi.com | `minimax/abab-5.5` |
| Mistral | https://console.mistral.ai | `mistral/mistral-large` |
| xAI Grok | https://console.x.ai | `xai/grok-beta` |
| SiliconFlow | https://siliconflow.cn | `siliconflow/qwen-7b` |
| Azure OpenAI | https://azure.microsoft.com/en-us/products/ai-services/openai-service | `azure/gpt-4o` |
| Ollama | https://ollama.com | `ollama/llama3.1` |
| vLLM | https://github.com/vllm-project/vllm | `vllm/custom-model` |

**注意：**
- URI 格式：`provider/model`
- 自定义 base_url：使用完整格式 `provider@base_url:model`
- 本地部署（Ollama、vLLM）需要自定义 base_url 配置

---

## MCP 服务器资源

探索数千个现成的 MCP 服务器：

| 平台 | 描述 | 网址 |
|----------|-------------|-----|
| **Mcp.so** | 8,000+ 服务器，支持 STDIO 和 SSE，带 API 游乐场 | https://mcp.so |
| **Smithery** | 4,500+ 服务器，新手友好，Cursor 一键配置 | https://smithery.ai |
| **阿里巴巴百炼** | 企业级 MCP 市场，提供云托管服务 | https://bailian.console.aliyun.com/?tab=mcp#/mcp-market |
| **ModelScope** | 阿里巴巴云运营的最大中文 MCP 社区 | https://modelscope.cn/mcp |
| **Awesome MCP** | 200+ 精选服务器，按类别组织（GitHub） | https://github.com/punkpeye/awesome-mcp-servers |
| **字节跳动火山引擎** | 企业级稳定安全的 MCP 服务 | https://www.volcengine.com/mcp-marketplace |
| **科大讯飞星火** | 星火 AI 平台的 MCP 服务器 | https://mcp.xfyun.cn |
| **百度 SAI** | 探索海量可用 MCP 服务器 | https://sai.baidu.com/mcp |
| **PulseMCP** | 3,290+ 服务器，每周更新和教程 | https://www.pulsemcp.com |
| **mcp.run** | 200+ 模板，支持一键网页部署 | https://www.mcp.run |

## 🌕 chak 适合你吗？

如果你：
- 需要连接到多个模型平台
- 想要简单、自动的上下文管理
- 想要最简单的工具调用体验——只需传递函数、对象或 MCP 工具
- 想要专注于构建应用，而不是纠结于上下文和工具

那么 chak 就是为你而生的。

<div align="right"><a href="https://youtube.com/watch?v=xOKQ7EQcggw"><img src="https://raw.githubusercontent.com/zhixiangxue/chak-ai/main/docs/assets/logo.png" alt="Demo Video" width="120"></a></div>