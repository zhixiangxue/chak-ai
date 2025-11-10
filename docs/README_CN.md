# chak

[English](../README.md) | [中文](README_CN.md)

一个多模型 LLM 客户端，内置上下文管理能力。

chak 不是另一个 one-api 或 OpenRouter，而是一个会主动帮你管理对话上下文的客户端库。你只需要专注对话本身，上下文工程交给 chak。

---

## 核心特性

**1. 内置上下文管理**

chak 的核心能力是上下文管理。提供多种策略（FIFO、Summarization、LRU）自动帮你处理对话历史，既保持完整记录，又节省 token 开销。你只管对话，上下文交给 chak。

**2. 简洁的 URI 调用**

一行代码连接全球主流模型，无需记忆复杂的 SDK 配置：

```python
# 简洁形式（推荐）
conv = chak.Conversation("openai/gpt-4o-mini", api_key="YOUR_KEY")

# 完整形式（自定义 base_url）
conv = chak.Conversation("deepseek@https://api.deepseek.com:deepseek-chat", api_key="YOUR_KEY")
```

**3. 短期记忆 → 长期记忆**

- 现在：短期记忆管理（FIFO 截断、Summarization 归纳、LRU主动遗忘），开箱可用
- 未来：长期记忆能力（RAG、记忆库），让对话真正"记得住"，计划中

---

## 已集成供应商（18+）

OpenAI、Google Gemini、Azure OpenAI、Anthropic Claude、阿里百炼、百度文心、腾讯混元、字节火山、智谱 GLM、Moonshot、DeepSeek、科大讯飞、MiniMax、Mistral、SiliconFlow、xAI Grok、Ollama、vLLM 等。

---

## 快速开始

### 安装

```bash
pip install chak
```

### 几行代码即可和全球模型对话

```python
import chak

conv = chak.Conversation(
    "openai/gpt-4o-mini",
    api_key="YOUR_KEY"
)

resp = conv.send("用一句话解释什么是上下文管理")
print(resp.content)
```

chak 帮你处理了：连接初始化、消息对齐、异常重试、上下文管理、模型格式转换……你只需要 `send` 消息就行了。

---

## 开启上下文自动管理

### 策略 A：`FIFOStrategy` - 保留最近 N 轮

适合快节奏对话，像滚动窗口一样保持对话新鲜：

```python
from chak import Conversation, FIFOStrategy

conv = Conversation(
    "deepseek/deepseek-chat",
    api_key="YOUR_KEY",
    context_strategy=FIFOStrategy(
        keep_recent_turns=3,       # 只保留最近 3 轮对话
        max_input_tokens=120_000   # 上下文窗口大小
    )
)
```

**参数说明：**
- `keep_recent_turns`：保留最近几轮？一轮 = 从一个用户消息到下一个用户消息之间的所有内容。
- `max_input_tokens`：给策略一个"胃容量"上限，超过这个数就往前挪,确保不会爆掉模型的上下文窗口。

工作方式：策略在保留区间之前插入一个截断 Marker，实际发送时只发送 Marker 之后的内容。原始对话？一条不少，全在 `conversation.messages` 里。

### 策略 B：`SummarizationStrategy` - 智能归纳历史

适合长对话，像一个贴心的总结助手：

```python
from chak import Conversation, SummarizationStrategy

conv = Conversation(
    "openai/gpt-5",
    api_key="YOUR_KEY",
    context_strategy=SummarizationStrategy(
        max_input_tokens=128_000,            # 上下文窗口大小
        summarize_threshold=0.75,            # 触发归纳的阈值
        prefer_recent_turns=2,               # 保留最近几轮
        summarizer_model_uri="openai/gpt-4o-mini",  # 总结模型
        summarizer_api_key="YOUR_KEY"
    )
)
```

**参数说明：**
- `max_input_tokens`：你的模型上下文窗口有多大？策略会参考这个值来决定何时触发。
- `summarize_threshold`：到达窗口的多少比例时触发归纳？0.75 = 75%，给后续对话留点余地。
- `prefer_recent_turns`：最近几轮不要动，保持对话的"现场感"。
- `summarizer_model_uri` / `summarizer_api_key`：用哪个模型来做归纳？可以和主对话用同一个，也可以用更便宜的。

**工作方式：**

当对话积累到一定长度时，chak 会自动触发归纳。把早期对话浓缩成几条要点，插入一个标记到消息链中。后续发送时，只发送这个标记及之后的内容。这样既保留了完整历史，又大幅减少了实际发送的 token 数，可以让你一直对话下去，而无需担心上下文窗口的大小。

原始对话依然完整保存在 `conversation.messages`，你随时可以查看、导出、分析。

### 策略 C：`LRUStrategy` - 智能遗忘冷话题

适合话题跳跃的长对话，自动淡化不再讨论的话题，保留热点内容：

```python
from chak import Conversation, LRUStrategy

conv = Conversation(
    "deepseek/deepseek-chat",
    api_key="YOUR_KEY",
    context_strategy=LRUStrategy(
        max_input_tokens=128_000,            # 上下文窗口大小
        summarize_threshold=0.75,            # 触发归纳的阈值
        prefer_recent_turns=2,               # 保留最近几轮
        summarizer_model_uri="deepseek/deepseek-chat", # 总结模型
        summarizer_api_key="YOUR_KEY"
    )
)
```

**参数说明：**
- 参数与 `SummarizationStrategy` 完全相同，使用方式也一致
- 内部增强：基于 Summarization 策略，额外分析最近 5 个摘要标记
- 智能遗忘：检测哪些话题不再被讨论，自动淡化冷话题，强化热点内容

**工作方式：**

1. 首先像 `SummarizationStrategy` 一样工作，生成摘要标记
2. 当摘要标记积累到一定数量时，LRU 增强机制启动
3. 分析最近 5 个标记，识别"热话题"（持续被讨论的）和"冷话题"（不再提及的）
4. 创建 LRU 标记，只保留热话题内容，淡化冷话题
5. 原始摘要标记和完整历史依然保留，随时可查看

**适用场景：**
- 话题经常切换的对话（如：Python → Java → 机器学习）
- 长时间对话中只关心当前讨论的主题
- 希望模型"遗忘"早期不相关的话题，聚焦当前任务

---

## 实用工具

### 查看对话统计

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

### 调试策略行为

设置环境变量查看策略内部运作：

```bash
export CHAK_LOG_LEVEL=DEBUG
python your_script.py
```

chak 会输出详细的策略执行日志：触发点、保留区间、摘要预览等。

---

## 本地服务模式（可选）

2 行代码即可启动一个本地网关服务：

### 1. 创建配置文件

```yaml
# chak-config.yaml
api_keys:
  # 简单格式 - 使用默认 base_url
  openai: ${OPENAI_API_KEY}           # 从环境变量读取（推荐）
  bailian: "sk-your-api-key-here"    # 明文配置（开发测试用）
  
  # 自定义 base_url（需加引号）
  "ollama@http://localhost:11434": "ollama"
  "vllm@http://192.168.1.100:8000": "dummy-key"

server:
  host: "0.0.0.0"
  port: 8000
```

### 2. 启动服务

```python
import chak

chak.serve('chak-config.yaml')
```

就这样！服务就启动了，你会看到：

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

### 3. 使用 Playground 快速和模型对话

打开 `http://localhost:8000/playground`，选择供应商和模型，立即开始对话。实时体验和全球LLM进行交互。

### 4. 用任意语言调用

服务提供 WebSocket API，你可以用 JavaScript、Go、Java、Rust 等任何语言调用：

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
  message: '你好！',
  stream: true
}));
```

这样 chak 就成了你的本地 LLM 网关，统一管理所有厂商的 API key，任意语言都能调用。

---

## 支持的 LLM 厂商

| 厂商 | 注册地址 | URI 示例 |
|------|---------|----------|
| OpenAI | https://platform.openai.com | `openai/gpt-4o` |
| Anthropic | https://console.anthropic.com | `anthropic/claude-3-5-sonnet` |
| Google Gemini | https://ai.google.dev | `google/gemini-1.5-pro` |
| DeepSeek | https://platform.deepseek.com | `deepseek/deepseek-chat` |
| 阿里百炼 | https://bailian.console.aliyun.com | `bailian/qwen-max` |
| 智谱 GLM | https://open.bigmodel.cn | `zhipu/glm-4` |
| Moonshot | https://platform.moonshot.cn | `moonshot/moonshot-v1-8k` |
| 百度文心 | https://console.bce.baidu.com/qianfan | `baidu/ernie-bot-4` |
| 腾讯混元 | https://cloud.tencent.com/product/hunyuan | `tencent/hunyuan-standard` |
| 字节豆包 | https://console.volcengine.com/ark | `volcengine/doubao-pro` |
| 科大讯飞 | https://xinghuo.xfyun.cn | `iflytek/spark-v3.5` |
| MiniMax | https://platform.minimaxi.com | `minimax/abab-5.5` |
| Mistral | https://console.mistral.ai | `mistral/mistral-large` |
| xAI Grok | https://console.x.ai | `xai/grok-beta` |
| SiliconFlow | https://siliconflow.cn | `siliconflow/qwen-7b` |
| Azure OpenAI | https://azure.microsoft.com/en-us/products/ai-services/openai-service | `azure/gpt-4o` |
| Ollama | https://ollama.com | `ollama/llama3.1` |
| vLLM | https://github.com/vllm-project/vllm | `vllm/custom-model` |

**说明：**
- URI 格式：`provider/model`
- 自定义 base_url：使用完整格式 `provider@base_url:model`
- 本地部署（Ollama、vLLM）需配置自定义 base_url

## 适合你吗？

如果你：
- 需要连接多个模型平台
- 想要"开箱即用"的上下文管理，而不是自己造轮子

那 chak 就是为你准备的。
