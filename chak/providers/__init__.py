# src/chak/providers/__init__.py
"""
Provider factory - 统一入口，按类型分类管理
支持LLM、Vision、Video等不同类型的provider

Provider注册采用二维索引：(provider_name, category) -> provider_info
允许同一个provider名称在不同category下存在不同的实现。
"""

from typing import Dict, Type, Any, List, Optional, Tuple
from .llm.base import BaseProviderConfig, BaseMessageConverter, Provider
from .types import ProviderCategory

# 全局provider注册表（二维索引）
# 结构: (provider_name, category) -> {"class": Provider, "config": Config, "converter": Converter}
_PROVIDERS: Dict[Tuple[str, ProviderCategory], Dict[str, Any]] = {}


def register_provider(
        name: str,
        provider_class: Type[Provider],
        config_class: Type[BaseProviderConfig],
        converter_class: Type[BaseMessageConverter],
        category: ProviderCategory
):
    """注册provider，支持同名provider在不同category下注册
    
    Args:
        name: provider名称（如"openai"、"zhipu"）
        provider_class: Provider类
        config_class: 配置类
        converter_class: 转换器类
        category: 类别枚举（ProviderCategory.LLM等）
    """
    key = (name.lower(), category)
    _PROVIDERS[key] = {
        "class": provider_class,
        "config": config_class,
        "converter": converter_class,
    }


def create_provider(
        provider_name: str,
        config_dict: Dict[str, Any],
        category: ProviderCategory
) -> Provider:
    """
    创建provider实例（工厂方法）
    
    Args:
        provider_name: provider名称（如"openai"、"zhipu"）
        config_dict: 配置字典
        category: provider类别（如ProviderCategory.LLM）
        
    Returns:
        Provider实例
        
    Raises:
        ValueError: 当指定的provider不存在时
    """
    provider_name = provider_name.lower()
    key = (provider_name, category)

    if key not in _PROVIDERS:
        available_categories = [cat for name, cat in _PROVIDERS.keys() if name == provider_name]
        if available_categories:
            raise ValueError(
                f"Provider '{provider_name}' not found for category '{category}'. "
                f"Available categories for '{provider_name}': {available_categories}"
            )
        else:
            raise ValueError(
                f"Provider '{provider_name}' not found. "
                f"Available providers: {get_available_providers()}"
            )

    info = _PROVIDERS[key]
    
    # Create config instance
    config = info["config"](**config_dict)

    # Create converter instance
    converter = info["converter"]()

    # Create provider instance
    return info["class"](config, converter)


def get_available_providers(category: Optional[ProviderCategory] = None) -> List[str]:
    """获取可用的provider列表，可按类别筛选
    
    Args:
        category: 类别筛选，None表示返回所有provider（去重）
        
    Returns:
        provider名称列表（已去重）
    """
    if category is None:
        # 返回所有唯一的provider名称
        return sorted(list(set(name for name, _ in _PROVIDERS.keys())))
    return sorted(list(set(name for name, cat in _PROVIDERS.keys() if cat == category)))


def get_provider_info(provider_name: str, category: Optional[ProviderCategory] = None) -> Dict[str, Any]:
    """获取provider的详细信息
    
    Args:
        provider_name: provider名称
        category: 可选的类别筛选，如果provider在多个category下存在则必须指定
        
    Returns:
        包含categories信息的字典
        
    Raises:
        ValueError: 当provider不存在或需要指定category时
    """
    provider_name = provider_name.lower()
    
    # 查找该provider在哪些category下存在
    categories = [cat for name, cat in _PROVIDERS.keys() if name == provider_name]
    
    if not categories:
        raise ValueError(
            f"Provider '{provider_name}' not found. "
            f"Available providers: {get_available_providers()}"
        )
    
    # 如果指定了category，验证是否存在
    if category is not None:
        if category not in categories:
            raise ValueError(
                f"Provider '{provider_name}' not found for category '{category}'. "
                f"Available categories: {categories}"
            )
        return {
            "name": provider_name,
            "category": category
        }
    
    # 如果没有指定category，返回所有相关信息
    return {
        "name": provider_name,
        "categories": categories
    }


# 初始化时注册所有providers
from .llm import register_llm_providers

# 注册LLM providers
register_llm_providers(register_provider)

# 未来扩展示例:
# from .vision import register_vision_providers
# register_vision_providers(register_provider)

__all__ = ['create_provider', 'get_available_providers', 'get_provider_info', 'register_provider']