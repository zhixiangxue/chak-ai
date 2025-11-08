"""Provider类型定义"""

from enum import Enum


class ProviderCategory(str, Enum):
    """Provider类别枚举
    
    定义不同类型的AI服务提供者类别。
    继承str以便于序列化和字符串比较。
    """
    
    LLM = "llm"                    # 大语言模型（文本生成）
    VISION = "vision"              # 视觉模型（图像生成、图像理解）
    VIDEO = "video"                # 视频模型（视频生成、视频理解）
    AUDIO = "audio"                # 音频模型（语音识别、语音合成）
    EMBEDDING = "embedding"        # 向量化模型（文本嵌入）
    
    def __str__(self) -> str:
        """返回枚举值的字符串形式"""
        return self.value
