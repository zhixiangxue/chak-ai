# src/chak/utils/uri.py
"""
Simple URI utilities for building and parsing model URIs.

Supports two URI formats:
1. Simple format: provider/model
   - Uses default base_url from provider config
   - All parameters passed via constructor
   - Example: "deepseek/deepseek-chat", "openai/gpt-4"

2. Full format: provider@base_url:model?params
   - Full control over base_url and parameters
   - Example: "deepseek@https://api.deepseek.com:deepseek-chat?temperature=0.7"
   - Use "~" for default base_url: "deepseek@~:deepseek-chat"

Examples:
    >>> build("openai", "gpt-4")
    'openai@~:gpt-4'
    
    >>> parse("deepseek/deepseek-chat")
    {
        'provider': 'deepseek',
        'base_url': None,
        'model': 'deepseek-chat',
        'params': {}
    }
    
    >>> parse("openai@https://api.openai.com/v1:gpt-4?temperature=0.7")
    {
        'provider': 'openai',
        'base_url': 'https://api.openai.com/v1', 
        'model': 'gpt-4',
        'params': {'temperature': '0.7'}
    }
"""
import re
from typing import Dict, Any, Optional
from urllib.parse import urlencode, parse_qs, unquote

from ..exceptions import URIError


def build(
        provider: str,
        model: str,
        base_url: Optional[str] = None,
        **params: Any
) -> str:
    """
    Build a model URI from components.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3")
        base_url: Custom base URL (can be full URL with path), or None for default (~)
        **params: Query parameters

    Returns:
        Formatted URI string

    Examples:
        >>> build("openai", "gpt-4")
        'openai@~:gpt-4'

        >>> build("dashscope", "qwen-turbo", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        'dashscope@https://dashscope.aliyuncs.com/compatible-mode/v1:qwen-turbo?temperature=0.7'

        >>> build("bailian", "qwen-plus", "localhost:3000")
        'bailian@localhost:3000:qwen-plus?temperature=0.7'
    """
    # Validate inputs
    if not provider or not isinstance(provider, str):
        raise URIError("Provider must be a non-empty string")
    if not model or not isinstance(model, str):
        raise URIError("Model must be a non-empty string")

    # 验证 provider 不包含特殊字符（model 可以包含冒号，如 ollama 的 qwen3:8b）
    if any(c in provider for c in '@:~?#'):
        raise URIError(f"Provider cannot contain special characters: {provider}")
    if any(c in model for c in '@~?#'):
        raise URIError(f"Model cannot contain special characters (@~?#): {model}")

    # 使用 ~ 作为默认 base_url 的占位符
    authority = "~" if base_url is None else base_url.rstrip('/')

    # 构建URI：provider@authority:model
    uri = f"{provider}@{authority}:{model}"

    # 添加查询参数
    if params:
        filtered_params = {k: v for k, v in params.items() if v is not None}
        if filtered_params:
            query_string = urlencode(filtered_params)
            uri = f"{uri}?{query_string}"

    return uri


def parse(uri: str) -> Dict[str, Any]:
    """
    Parse a model URI into its components.
    
    Supports two formats:
    1. Simple: "provider/model" (uses default base_url)
    2. Full: "provider@base_url:model?params" (full control)

    Args:
        uri: Model URI string
        
    Returns:
        Dictionary with keys: provider, base_url, model, params
        
    Raises:
        URIError: If URI format is invalid
    """
    if not uri or not isinstance(uri, str):
        raise URIError("URI must be a non-empty string")
    
    # Detect format based on presence of '@'
    if '@' in uri:
        # Full format: provider@base_url:model?params
        return _parse_full_format(uri)
    elif '/' in uri:
        # Simple format: provider/model
        return _parse_simple_format(uri)
    else:
        raise URIError(
            f"Invalid URI format: {uri}\n"
            f"Expected formats:\n"
            f"  - Simple: provider/model (e.g., 'deepseek/deepseek-chat')\n"
            f"  - Full: provider@base_url:model?params (e.g., 'deepseek@~:deepseek-chat')"
        )


def _parse_simple_format(uri: str) -> Dict[str, Any]:
    """
    Parse simple format URI: provider/model
    
    Args:
        uri: Simple format URI (e.g., "deepseek/deepseek-chat")
        
    Returns:
        Dictionary with provider, model, base_url=None, params={}
        
    Raises:
        URIError: If format is invalid
    """
    # Simple format should not have query parameters
    if '?' in uri:
        raise URIError(
            f"Simple format URI cannot contain query parameters: {uri}\n"
            f"Use full format for parameters: provider@base_url:model?params"
        )
    
    # Split by first '/'
    parts = uri.split('/', 1)
    if len(parts) != 2:
        raise URIError(f"Invalid simple format URI: {uri}\nExpected: provider/model")
    
    provider, model = parts
    
    # Validate provider and model
    if not provider or not model:
        raise URIError(f"Provider and model cannot be empty: {uri}")
    
    # Provider should not contain special characters
    if any(c in provider for c in '@:~?#/'):
        raise URIError(f"Invalid provider name: {provider}")
    
    return {
        'provider': provider,
        'base_url': None,  # Will use default from provider config
        'model': model,
        'params': {}
    }


def _parse_full_format(uri: str) -> Dict[str, Any]:
    """
    Parse full format URI: provider@base_url:model?params
    
    This is the original parsing logic for backward compatibility.
    
    Args:
        uri: Full format URI
        
    Returns:
        Dictionary with provider, base_url, model, params
    """
    if not uri or not isinstance(uri, str):
        raise URIError("URI must be a non-empty string")

    # 先检查是否有查询参数
    if '?' in uri:
        uri_part, query_string = uri.split('?', 1)
    else:
        uri_part = uri
        query_string = None

    # 使用更智能的解析方法
    # 格式：provider@base_url:model
    # 挑战：base_url 可能包含端口号（如 localhost:8080），model 也可能包含冒号（如 qwen3:8b）
    # 策略：找到 base_url 和 model 之间的分隔冒号
    #   - 如果 base_url 是完整 URL（以 http:// 或 https:// 开头），在 // 后面找第一个路径结束位置
    #   - 否则，base_url 可能是 host:port 格式，端口后面紧跟的冒号才是分隔符

    # 首先分割provider和其余部分
    if '@' not in uri_part:
        raise URIError(f"Invalid URI format: missing '@' separator in {uri}")

    provider, rest = uri_part.split('@', 1)

    if ':' not in rest:
        raise URIError(f"Invalid URI format: missing ':' separator in {uri}")

    # 智能查找 base_url 和 model 的分隔冒号
    base_url_part = None
    model = None
    
    # 情况1：base_url 是完整的 HTTP(S) URL
    if rest.startswith('http://') or rest.startswith('https://'):
        # 找到协议后的第一个冒号（端口号或路径结束）
        protocol_end = rest.index('//') + 2
        rest_after_protocol = rest[protocol_end:]
        
        # 在协议后查找路径结束的标记
        # 路径可能以 /v1, /api 等结尾，后面跟冒号和模型名
        # 策略：找到不属于 URL 部分的第一个冒号
        # URL 中冒号后只能跟数字（端口）或 /（路径继续）
        split_index = -1
        for i, char in enumerate(rest_after_protocol):
            if char == ':':
                # 检查冒号后面是什么
                next_part = rest_after_protocol[i+1:i+10]  # 检查后面最多10个字符
                # 如果冒号后不是纯数字开头，也不是 / 开头，那就是模型名分隔符
                if next_part and not next_part[0].isdigit() and not next_part[0] == '/':
                    split_index = protocol_end + i
                    break
        
        if split_index > 0:
            base_url_part = rest[:split_index]
            model = rest[split_index + 1:]
        else:
            # 没找到，fallback 到最后一个冒号
            last_colon = rest.rfind(':')
            base_url_part = rest[:last_colon]
            model = rest[last_colon + 1:]
    
    # 情况2：base_url 是 ~ (默认)
    elif rest.startswith('~:'):
        base_url_part = '~'
        model = rest[2:]  # 跳过 ~:
    
    # 情况3：base_url 可能是 host:port 格式
    else:
        # 找第一个冒号（可能是端口）
        first_colon = rest.index(':')
        # 检查第一个冒号后面是否是数字（端口号）
        after_first_colon = rest[first_colon + 1:]
        
        # 如果冒号后面开始是数字，说明是端口号
        if after_first_colon and after_first_colon[0].isdigit():
            # 找端口号后的下一个冒号
            port_end = first_colon + 1
            while port_end < len(rest) and rest[port_end].isdigit():
                port_end += 1
            
            if port_end < len(rest) and rest[port_end] == ':':
                # 找到了端口后的冒号，这是分隔符
                base_url_part = rest[:port_end]
                model = rest[port_end + 1:]
            else:
                # 没有找到端口后的冒号，使用最后一个冒号
                last_colon = rest.rfind(':')
                base_url_part = rest[:last_colon]
                model = rest[last_colon + 1:]
        else:
            # 第一个冒号后不是数字，那就是模型分隔符
            base_url_part = rest[:first_colon]
            model = rest[first_colon + 1:]

    # 处理 base_url 部分：~ 表示默认/无 base_url
    base_url = None if base_url_part == "~" else base_url_part

    # 解析查询参数
    params = {}
    if query_string:
        try:
            parsed = parse_qs(query_string, keep_blank_values=False)
            for key, values in parsed.items():
                if len(values) == 1:
                    params[key] = values[0]
                else:
                    params[key] = values
        except Exception as e:
            raise URIError(f"Failed to parse query string: {e}")

    return {
        'provider': provider,
        'base_url': base_url,
        'model': model,
        'params': params
    }


def build_simple(provider: str, model: str) -> str:
    """
    Build a simple format URI: provider/model
    
    Args:
        provider: Provider name (e.g., "openai", "deepseek")
        model: Model name (e.g., "gpt-4", "deepseek-chat")
        
    Returns:
        Simple format URI string
        
    Examples:
        >>> build_simple("deepseek", "deepseek-chat")
        'deepseek/deepseek-chat'
    """
    # Validate inputs
    if not provider or not isinstance(provider, str):
        raise URIError("Provider must be a non-empty string")
    if not model or not isinstance(model, str):
        raise URIError("Model must be a non-empty string")
    
    # Validate provider doesn't contain special characters
    if any(c in provider for c in '@:~?#/'):
        raise URIError(f"Provider cannot contain special characters: {provider}")
    
    return f"{provider}/{model}"