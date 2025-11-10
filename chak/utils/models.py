"""
Model configuration for all supported providers.

This module maintains the mapping between display names and actual model IDs
for each LLM provider. Used across the codebase for model selection.
"""

PROVIDER_MODELS = {
    "anthropic": {
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307"
    },
    "azure": {
        "gpt-4o": "gpt-4o",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-3.5-turbo": "gpt-3.5-turbo"
    },
    "baidu": {
        "ernie-bot": "ERNIE-Bot",
        "ernie-bot-turbo": "ERNIE-Bot-Turbo",
        "ernie-bot-4": "ERNIE-Bot-4"
    },
    "bailian": {
        "qwen-flash": "qwen-flash",
        "qwen-plus": "qwen-plus",
        "qwen-max": "qwen-max",
        "qwen-turbo": "qwen-turbo",
        "baichuan-7b": "baichuan-7b"
    },
    "deepseek": {
        "deepseek-chat": "deepseek-chat",
        "deepseek-coder": "deepseek-coder"
    },
    "google": {
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-pro": "gemini-pro"
    },
    "iflytek": {
        "spark-v3.5": "spark-v3.5",
        "spark-v3.0": "spark-v3.0",
        "spark-lite": "spark-lite"
    },
    "minimax": {
        "abab-5.5": "abab-5.5-chat",
        "abab-5.0": "abab-5.0-chat"
    },
    "mistral": {
        "mistral-large": "mistral-large-latest",
        "mixtral-8x7b": "mixtral-8x7b-instruct",
        "mistral-7b": "mistral-7b-instruct"
    },
    "moonshot": {
        "moonshot-v1-8k": "moonshot-v1-8k",
        "moonshot-v1-32k": "moonshot-v1-32k",
        "moonshot-v1-128k": "moonshot-v1-128k"
    },
    "ollama": {
        "qwen3-8b": "qwen3:8b",
        "llama3.1": "llama3.1:latest",
        "phi3": "phi3:latest"
    },
    "openai": {
        "gpt-4o": "gpt-4o",
        "gpt-4-turbo": "gpt-4-turbo-preview",
        "gpt-4": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo"
    },
    "siliconflow": {
        "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
        "llama-7b": "meta-llama/Llama-3.3-7B-Instruct"
    },
    "tencent": {
        "hunyuan-standard": "hunyuan-standard",
        "hunyuan-lite": "hunyuan-lite"
    },
    "vllm": {
        "custom-model": "your/model/path"
    },
    "volcengine": {
        "doubao-pro": "doubao-pro",
        "doubao-lite": "doubao-lite"
    },
    "xai": {
        "grok-beta": "grok-beta",
        "grok-vision": "grok-vision-preview"
    },
    "zhipu": {
        "glm-4": "glm-4",
        "glm-4-air": "glm-4-air",
        "glm-3-turbo": "glm-3-turbo"
    }
}


def get_models(provider: str) -> dict:
    """
    Get available models for a provider.
    
    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        
    Returns:
        Dictionary of {display_name: model_id}
    """
    return PROVIDER_MODELS.get(provider, {})


def get_default_model(provider: str) -> str:
    """
    Get default model for a provider (first in the list).
    
    Args:
        provider: Provider name
        
    Returns:
        Default model ID, or empty string if provider not found
    """
    models = get_models(provider)
    if models:
        return list(models.values())[0]
    return ""
