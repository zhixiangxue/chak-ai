"""
模型信息配置工具

维护主流LLM模型的核心配置信息，包括：
- 上下文长度
- 最大输入/输出tokens
- 最长思维链tokens
- 定价信息

用于：
- 计算上下文使用比例
- 估算对话成本
- 模型能力参考
"""

from typing import Dict, Any, Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """模型信息"""
    name: str  # 模型名称
    version: str  # 版本
    mode: str  # 模式（如 chat, completion, reasoning）
    context_length: int  # 上下文长度（tokens）
    max_input: int  # 最大输入tokens
    max_reasoning: Optional[int]  # 最长思维链tokens（如果支持）
    max_output: int  # 最大输出tokens
    input_cost: float  # 输入成本（每百万tokens）
    output_cost: float  # 输出成本（每百万tokens）
    currency: str  # 计价币种


# 模型信息映射表
MODEL_INFO_MAP: Dict[str, ModelInfo] = {
    # OpenAI Models
    "gpt-4o": ModelInfo(
        name="GPT-4o",
        version="2024-11",
        mode="chat",
        context_length=128_000,
        max_input=128_000,
        max_reasoning=None,
        max_output=16_384,
        input_cost=2.5,  # $2.50 per 1M tokens
        output_cost=10.0,  # $10.00 per 1M tokens
        currency="USD"
    ),
    "gpt-4o-mini": ModelInfo(
        name="GPT-4o-mini",
        version="2024-07",
        mode="chat",
        context_length=128_000,
        max_input=128_000,
        max_reasoning=None,
        max_output=16_384,
        input_cost=0.15,  # $0.15 per 1M tokens
        output_cost=0.6,  # $0.60 per 1M tokens
        currency="USD"
    ),
    "o1": ModelInfo(
        name="o1",
        version="2024-12",
        mode="reasoning",
        context_length=200_000,
        max_input=200_000,
        max_reasoning=100_000,
        max_output=100_000,
        input_cost=15.0,  # $15 per 1M tokens
        output_cost=60.0,  # $60 per 1M tokens
        currency="USD"
    ),
    "o1-mini": ModelInfo(
        name="o1-mini",
        version="2024-09",
        mode="reasoning",
        context_length=128_000,
        max_input=128_000,
        max_reasoning=65_536,
        max_output=65_536,
        input_cost=3.0,  # $3 per 1M tokens
        output_cost=12.0,  # $12 per 1M tokens
        currency="USD"
    ),
    
    # Alibaba Cloud Models
    "qwen-plus": ModelInfo(
        name="Qwen-Plus",
        version="latest",
        mode="chat",
        context_length=131_072,
        max_input=131_072,
        max_reasoning=None,
        max_output=8192,
        input_cost=0.4,  # ¥0.0004 per 1k tokens = ¥0.4 per 1M tokens
        output_cost=1.2,  # ¥0.0012 per 1k tokens = ¥1.2 per 1M tokens
        currency="CNY"
    ),
    "qwen-max": ModelInfo(
        name="Qwen-Max",
        version="latest",
        mode="chat",
        context_length=32_768,
        max_input=30_000,
        max_reasoning=None,
        max_output=8192,
        input_cost=4.0,  # ¥0.004 per 1k tokens = ¥4 per 1M tokens
        output_cost=12.0,  # ¥0.012 per 1k tokens = ¥12 per 1M tokens
        currency="CNY"
    ),
    "qwen-turbo": ModelInfo(
        name="Qwen-Turbo",
        version="latest",
        mode="chat",
        context_length=131_072,
        max_input=131_072,
        max_reasoning=None,
        max_output=8192,
        input_cost=0.2,  # ¥0.0002 per 1k tokens = ¥0.2 per 1M tokens
        output_cost=0.6,  # ¥0.0006 per 1k tokens = ¥0.6 per 1M tokens
        currency="CNY"
    ),
    
    # Google Models
    "gemini-2.0-flash-exp": ModelInfo(
        name="Gemini 2.0 Flash",
        version="Experimental",
        mode="chat",
        context_length=1_048_576,  # 1M tokens
        max_input=1_048_576,
        max_reasoning=None,
        max_output=8192,
        input_cost=0.0,  # Free tier
        output_cost=0.0,  # Free tier
        currency="USD"
    ),
    "gemini-1.5-pro": ModelInfo(
        name="Gemini 1.5 Pro",
        version="latest",
        mode="chat",
        context_length=2_097_152,  # 2M tokens
        max_input=2_097_152,
        max_reasoning=None,
        max_output=8192,
        input_cost=1.25,  # $1.25 per 1M tokens (prompt <= 128k)
        output_cost=5.0,  # $5.00 per 1M tokens
        currency="USD"
    ),
    "gemini-1.5-flash": ModelInfo(
        name="Gemini 1.5 Flash",
        version="latest",
        mode="chat",
        context_length=1_048_576,  # 1M tokens
        max_input=1_048_576,
        max_reasoning=None,
        max_output=8192,
        input_cost=0.075,  # $0.075 per 1M tokens (prompt <= 128k)
        output_cost=0.3,  # $0.30 per 1M tokens
        currency="USD"
    ),
}


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """
    获取模型信息。
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型信息对象，如果模型不存在则返回 None
    """
    return MODEL_INFO_MAP.get(model_name)


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> Optional[Dict[str, Any]]:
    """
    计算对话成本。
    
    Args:
        model_name: 模型名称
        input_tokens: 输入tokens数
        output_tokens: 输出tokens数
        
    Returns:
        包含成本信息的字典，如果模型不存在则返回 None
        {
            'input_cost': float,
            'output_cost': float,
            'total_cost': float,
            'currency': str,
            'formatted': str  # 格式化的成本字符串
        }
    """
    model_info = get_model_info(model_name)
    if not model_info:
        return None
    
    # 计算成本（成本是per 1M tokens）
    input_cost = (input_tokens / 1_000_000) * model_info.input_cost
    output_cost = (output_tokens / 1_000_000) * model_info.output_cost
    total_cost = input_cost + output_cost
    
    # 格式化成本字符串
    currency_symbol = "¥" if model_info.currency == "CNY" else "$"
    formatted = f"{currency_symbol}{total_cost:.4f}"
    
    return {
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'currency': model_info.currency,
        'formatted': formatted
    }


def get_context_usage_ratio(model_name: str, current_tokens: int) -> Optional[float]:
    """
    计算当前上下文使用比例。
    
    Args:
        model_name: 模型名称
        current_tokens: 当前使用的tokens数
        
    Returns:
        使用比例（0-1之间的浮点数），如果模型不存在则返回 None
    """
    model_info = get_model_info(model_name)
    if not model_info:
        return None
    
    return current_tokens / model_info.context_length


def format_context_usage(model_name: str, current_tokens: int) -> Optional[str]:
    """
    格式化上下文使用信息（类似Qoder的显示方式）。
    
    Args:
        model_name: 模型名称
        current_tokens: 当前使用的tokens数
        
    Returns:
        格式化的使用信息字符串，例如 "12.5K / 128K (9.8%)"
    """
    model_info = get_model_info(model_name)
    if not model_info:
        return None
    
    ratio = get_context_usage_ratio(model_name, current_tokens)
    if ratio is None:
        return None
    
    # 格式化当前tokens
    if current_tokens >= 1000:
        current_str = f"{current_tokens / 1000:.1f}K"
    else:
        current_str = str(current_tokens)
    
    # 格式化总tokens
    total_tokens = model_info.context_length
    if total_tokens >= 1_000_000:
        total_str = f"{total_tokens / 1_000_000:.1f}M"
    elif total_tokens >= 1000:
        total_str = f"{total_tokens / 1000:.0f}K"
    else:
        total_str = str(total_tokens)
    
    # 格式化百分比
    percentage = ratio * 100
    
    return f"{current_str} / {total_str} ({percentage:.1f}%)"
