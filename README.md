# chak

<div align="center">

[![Demo Video](https://raw.githubusercontent.com/zhixiangxue/chak-ai/main/docs/assets/logo.png)](https://youtube.com/watch?v=xOKQ7EQcggw)

[English](README.md) | [中文](docs/README_CN.md)

A multi-model LLM client with built-in context management and MCP tool integration.

chak is not another liteLLM, one-api, or OpenRouter, but a client library that actively manages conversation context and tool calls for you. Just focus on building your application, let chak handle the complexity.

</div>

---

## Core Features

**1. Minimalist API Design**

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

**2. Pluggable Context Management**

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

**3. Seamless Tool Calling (MCP Protocol)**

Extreme simplicity - just point to an MCP server:

```python
from chak import Conversation
from chak.mcp import Server

# Load tools from MCP server
tools = await Server(url="...").tools()

# That's it! Tool calling just works
conv = Conversation("openai/gpt-4o", tools=tools)
response = await conv.asend("What's the weather in San Francisco?")
```

- **Now**: Full async support with both streaming and non-streaming modes
- **Planning**: Smart tool selection - intelligently filter relevant tools based on context

---

## Integrated Providers (18+)

OpenAI, Google Gemini, Azure OpenAI, Anthropic Claude, Alibaba Bailian, Baidu Wenxin, Tencent Hunyuan, ByteDance Doubao, Zhipu GLM, Moonshot, DeepSeek, iFlytek Spark, MiniMax, Mistral, SiliconFlow, xAI Grok, Ollama, vLLM, and more.

---

## Quick Start

### Installation

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

## Enable Automatic Context Management

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

## MCP Tool Calling

chak integrates the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for seamless tool calling.

Quick start:

```python
import asyncio
from chak import Conversation
from chak.mcp import Server

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

See full examples (parameters, how it works, tips):

- SSE: [examples/mcp_chat_sse.py](examples/mcp_chat_sse.py)
- stdio: [examples/mcp_chat_stdio.py](examples/mcp_chat_stdio.py)
- HTTP: [examples/mcp_chat_http.py](examples/mcp_chat_http.py)


---

## Practical Utilities

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
- **MCP tool calls**: tool invocation, request/response details, execution results

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

## MCP Server Resources

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



## Is chak for You?

If you:
- Need to connect to multiple model platforms
- Want simple, automatic context management
- Need seamless MCP tool integration with minimal code
- Want to focus on building applications, not wrestling with context and tools

Then chak is made for you.