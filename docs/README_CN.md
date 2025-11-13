# chak

<div align="center">

[![Demo Video](https://raw.githubusercontent.com/zhixiangxue/chak-ai/main/docs/assets/logo.png)](https://youtube.com/watch?v=xOKQ7EQcggw)

[English](README.md) | [中文](docs/README_CN.md)

一个极简的多模型LLM客户端，支持上下文管理和工具调用。

chak不是另一个liteLLM、one-api或OpenRouter，而是一个为您主动管理对话上下文和工具调用的客户端库。你只需专注于对话，让chak处理上下文工程。

</div>

---

## 核心特性

**1. 极简API设计**

没有复杂的配置，没有学习曲线。chak设计直观：

```python
# 作为SDK使用 - 通过简单的URI连接任何LLM
conv = chak.Conversation("openai/gpt-4o-mini", api_key="YOUR_KEY")
response = conv.send("Hello!")

# 或作为本地网关运行 - 2行代码启动
import chak
chak.serve('chak-config.yaml')
```

无论您是构建应用程序还是运行网关，chak都保持简单。

**2. 可插拔的上下文管理**

Chak通过多种策略自动处理上下文：

- **当前**：短期记忆策略（FIFO、摘要、LRU）- 已可用于生产
- **规划中**：长期记忆（RAG、记忆库）- 使对话真正"有记忆"

没有其他工具能在这一级别自动化上下文管理。chak的策略模式使其完全可插拔和可扩展。

**3. 无缝工具调用（MCP协议）**

极其简单 - 只需指向一个MCP服务器：

```python
from chak import Conversation
from chak.mcp import Server

# 从MCP服务器加载工具
tools = await Server(url="...").tools()

# 就这样！工具调用即可工作
conv = Conversation("openai/gpt-4o", tools=tools)
response = await conv.asend("What's the weather in San Francisco?")
```

- **当前**：完整的异步支持，包括流式和非流式模式
- **规划中**：智能工具选择 - 根据上下文智能筛选相关工具

---

## 集成提供商（18+）

OpenAI、Google Gemini、Azure OpenAI、Anthropic Claude、阿里巴巴百炼、百度文心、腾讯混元、字节跳动豆包、智谱GLM、月之暗面、深度求索、讯飞星火、MiniMax、Mistral、SiliconFlow、xAI Grok、Ollama、vLLM等。

---

## 快速开始

### 安装

```bash
# 基础安装（仅SDK）
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

chak处理：连接初始化、消息对齐、重试逻辑、上下文管理、模型格式转换...您只需要`send`消息。

---

## 启用自动上下文管理

三种内置策略：

- FIFO：保留最近N轮对话，自动丢弃较早的。
- 摘要：当上下文达到阈值时，早期历史被摘要；最近几轮保持完整。
- LRU：基于摘要构建，保留热门话题并修剪冷门话题。

快速开始：

```python
from chak import Conversation, FIFOStrategy

conv = Conversation(
    "bailian/qwen-flash",
    api_key="YOUR_KEY",
    context_strategy=FIFOStrategy(keep_recent_turns=3)
)
```

查看完整示例（参数、工作原理、技巧）：

- FIFO: examples/strategy_chat_fifo.py
- 摘要: examples/strategy_chat_summarization.py
- LRU: examples/strategy_chat_lru.py

---

## MCP工具调用

chak集成了https://modelcontextprotocol.io/以实现无缝工具调用。

快速开始：

```python
import asyncio
from chak import Conversation
from chak.mcp import Server

async def main():
    # 连接到MCP服务器并加载工具
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

- **SSE**（服务器发送事件）：云托管的MCP服务
- **stdio**：本地MCP服务器
- **HTTP**：基于HTTP的MCP服务

查看完整示例（参数、工作原理、技巧）：

- SSE: examples/mcp_chat_sse.py
- stdio: examples/mcp_chat_stdio.py
- HTTP: examples/mcp_chat_http.py

---

## 实用工具

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

chak将输出详细日志：
- **上下文策略**：触发点、保留间隔、摘要预览、令牌计数
- **MCP工具调用**：工具调用、请求/响应详情、执行结果

---

## 本地服务器模式（可选）

用2行代码启动本地网关服务：

### 1. 创建配置文件

```yaml
# chak-config.yaml
api_keys:
  # 简单格式 - 使用默认base_url
  openai: ${OPENAI_API_KEY}           # 从环境变量读取（推荐）
  bailian: "sk-your-api-key-here"    # 纯文本（用于开发/测试）
  
  # 自定义base_url（需要引号）
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

就这样！服务器启动后您将看到：

```
======================================================================

  ✨✨ Chak AI 网关
  一个简单却方便的LLM网关

======================================================================

  🚀🚀🚀 服务器运行在:     http://localhost:8000
  🎮🎮🎮  playground:            http://localhost:8000/playground
  📡📡 WebSocket端点:    ws://localhost:8000/ws/conversation

  ⭐⭐ GitHub上点赞:        https://github.com/zhixiangxue/chak-ai

======================================================================
```

### 3. 使用Playground快速进行模型对话

打开`http://localhost:8000/playground`，选择提供商和模型，立即开始聊天。体验与全球LLM的实时交互。

### 4. 从任何语言调用

该服务提供WebSocket API，可从JavaScript、Go、Java、Rust或任何语言调用：

```javascript
// JavaScript示例
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

这样chak就成为您的本地LLM网关，集中管理所有提供商API密钥，可从任何语言调用。

---

## 支持的LLM提供商

| 提供商 | 注册 | URI示例 |
|----------|-------------|-------------|
| OpenAI | https://platform.openai.com | `openai/gpt-4o` |
| Anthropic | https://console.anthropic.com | `anthropic/claude-3-5-sonnet` |
| Google Gemini | https://ai.google.dev | `google/gemini-1.5-pro` |
| DeepSeek | https://platform.deepseek.com | `deepseek/deepseek-chat` |
| 阿里巴巴百炼 | https://bailian.console.aliyun.com | `bailian/qwen-max` |
| 智谱GLM | https://open.bigmodel.cn | `zhipu/glm-4` |
| 月之暗面 | https://platform.moonshot.cn | `moonshot/moonshot-v1-8k` |
| 百度文心 | https://console.bce.baidu.com/qianfan | `baidu/ernie-bot-4` |
| 腾讯混元 | https://cloud.tencent.com/product/hunyuan | `tencent/hunyuan-standard` |
| 字节跳动豆包 | https://console.volcengine.com/ark | `volcengine/doubao-pro` |
| 讯飞星火 | https://xinghuo.xfyun.cn | `iflytek/spark-v3.5` |
| MiniMax | https://platform.minimaxi.com | `minimax/abab-5.5` |
| Mistral | https://console.mistral.ai | `mistral/mistral-large` |
| xAI Grok | https://console.x.ai | `xai/grok-beta` |
| SiliconFlow | https://siliconflow.cn | `siliconflow/qwen-7b` |
| Azure OpenAI | https://azure.microsoft.com/en-us/products/ai-services/openai-service | `azure/gpt-4o` |
| Ollama | https://ollama.com | `ollama/llama3.1` |
| vLLM | https://github.com/vllm-project/vllm | `vllm/custom-model` |

**注意：**
- URI格式：`provider/model`
- 自定义base_url：使用完整格式`provider@base_url:model`
- 本地部署（Ollama、vLLM）需要自定义base_url配置

---

## MCP服务器资源

探索数千个即用型MCP服务器：

| 平台 | 描述 | 网址 |
|----------|-------------|-----|
| **Mcp.so** | 8,000+服务器，支持STDIO和SSE，带API playground | https://mcp.so |
| **Smithery** | 4,500+服务器，对新手友好，Cursor一键配置 | https://smithery.ai |
| **阿里巴巴百炼** | 企业级MCP市场，提供云托管服务 | https://bailian.console.aliyun.com/?tab=mcp#/mcp-market |
| **ModelScope** | 阿里云运营的最大中文MCP社区 | https://modelscope.cn/mcp |
| **Awesome MCP** | 200+精选服务器，按类别组织（GitHub） | https://github.com/punkpeye/awesome-mcp-servers |
| **字节跳动火山引擎** | 企业级稳定安全的MCP服务 | https://www.volcengine.com/mcp-marketplace |
| **讯飞星火** | 星火AI平台的MCP服务器 | https://mcp.xfyun.cn |
| **百度SAI** | 探索海量可用MCP服务器 | https://sai.baidu.com/mcp |
| **PulseMCP** | 3,290+服务器，每周更新和教程 | https://www.pulsemcp.com |
| **mcp.run** | 200+模板，支持一键Web部署 | https://www.mcp.run |

## chak适合您吗？

如果您：
- 需要连接多个模型平台
- 想要简单、自动的上下文管理
- 需要以最少代码无缝集成MCP工具
- 希望专注于构建应用程序，而不是纠结于上下文和工具

那么chak就是为您打造的。