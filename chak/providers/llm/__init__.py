"""LLM providers - 文本生成模型"""

from ..types import ProviderCategory
from .bailian import BailianProvider, BailianConfig, BailianMessageConverter
from .moonshot import MoonshotProvider, MoonshotConfig, MoonshotMessageConverter
from .siliconflow import SiliconFlowProvider, SiliconFlowConfig, SiliconFlowMessageConverter
from .volcengine import VolcEngineProvider, VolcEngineConfig, VolcEngineMessageConverter
from .tencent import TencentProvider, TencentConfig, TencentMessageConverter
from .baidu import BaiduProvider, BaiduConfig, BaiduMessageConverter
from .deepseek import DeepSeekProvider, DeepSeekConfig, DeepSeekMessageConverter
from .zhipu import ZhipuProvider, ZhipuConfig, ZhipuMessageConverter
from .minimax import MiniMaxProvider, MiniMaxConfig, MiniMaxMessageConverter
from .iflytek import IFlyTekProvider, IFlyTekConfig, IFlyTekMessageConverter
from .xai import XAIProvider, XAIConfig, XAIMessageConverter
from .openai import OpenAIProvider, OpenAIConfig, OpenAIMessageConverter
from .anthropic import AnthropicProvider, AnthropicConfig, AnthropicMessageConverter
from .mistral import MistralProvider, MistralConfig, MistralMessageConverter
from .google import GoogleProvider, GoogleConfig, GoogleMessageConverter
from .ollama import OllamaProvider, OllamaConfig, OllamaMessageConverter
from .vllm import VLLMProvider, VLLMConfig, VLLMMessageConverter
from .azure import AzureProvider, AzureConfig, AzureMessageConverter


def register_llm_providers(register_func):
    """注册所有LLM providers到全局注册表"""
    register_func("bailian", BailianProvider, BailianConfig, BailianMessageConverter, category=ProviderCategory.LLM)
    register_func("moonshot", MoonshotProvider, MoonshotConfig, MoonshotMessageConverter, category=ProviderCategory.LLM)
    register_func("siliconflow", SiliconFlowProvider, SiliconFlowConfig, SiliconFlowMessageConverter, category=ProviderCategory.LLM)
    register_func("volcengine", VolcEngineProvider, VolcEngineConfig, VolcEngineMessageConverter, category=ProviderCategory.LLM)
    register_func("tencent", TencentProvider, TencentConfig, TencentMessageConverter, category=ProviderCategory.LLM)
    register_func("baidu", BaiduProvider, BaiduConfig, BaiduMessageConverter, category=ProviderCategory.LLM)
    register_func("deepseek", DeepSeekProvider, DeepSeekConfig, DeepSeekMessageConverter, category=ProviderCategory.LLM)
    register_func("zhipu", ZhipuProvider, ZhipuConfig, ZhipuMessageConverter, category=ProviderCategory.LLM)
    register_func("minimax", MiniMaxProvider, MiniMaxConfig, MiniMaxMessageConverter, category=ProviderCategory.LLM)
    register_func("iflytek", IFlyTekProvider, IFlyTekConfig, IFlyTekMessageConverter, category=ProviderCategory.LLM)
    register_func("xai", XAIProvider, XAIConfig, XAIMessageConverter, category=ProviderCategory.LLM)
    register_func("openai", OpenAIProvider, OpenAIConfig, OpenAIMessageConverter, category=ProviderCategory.LLM)
    register_func("anthropic", AnthropicProvider, AnthropicConfig, AnthropicMessageConverter, category=ProviderCategory.LLM)
    register_func("mistral", MistralProvider, MistralConfig, MistralMessageConverter, category=ProviderCategory.LLM)
    register_func("google", GoogleProvider, GoogleConfig, GoogleMessageConverter, category=ProviderCategory.LLM)
    register_func("ollama", OllamaProvider, OllamaConfig, OllamaMessageConverter, category=ProviderCategory.LLM)
    register_func("vllm", VLLMProvider, VLLMConfig, VLLMMessageConverter, category=ProviderCategory.LLM)
    register_func("azure", AzureProvider, AzureConfig, AzureMessageConverter, category=ProviderCategory.LLM)


__all__ = ['register_llm_providers']
