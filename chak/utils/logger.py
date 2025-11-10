"""
统一的日志模块

基于 loguru 的简单易用的日志系统，可以在任何地方直接 import 使用。

使用方式：
    from chak.utils.logger import logger
    
    logger.info("这是一条信息")
    logger.debug("调试信息")
    logger.warning("警告信息")
    logger.error("错误信息")

配置方式：
    通过环境变量配置：
    - CHAK_LOG_LEVEL: 日志级别（DEBUG/INFO/WARNING/ERROR）默认 INFO
    - CHAK_LOG_TO_FILE: 是否输出到文件（true/false）默认 false
    - CHAK_LOG_FILE: 日志文件路径，默认 logs/chak.log
"""

import os
import sys
from pathlib import Path

from loguru import logger as _logger

# 移除 loguru 的默认 handler
_logger.remove()

# ===== 配置参数（从环境变量读取） =====
LOG_LEVEL = os.getenv("CHAK_LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("CHAK_LOG_TO_FILE", "false").lower() == "true"
LOG_FILE = os.getenv("CHAK_LOG_FILE", "logs/chak.log")

# ===== 日志格式 =====
# 简洁格式：时间 | 级别 | 消息
SIMPLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

# 详细格式：时间 | 级别 | 位置 | 消息
DETAILED_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# 根据日志级别选择格式（DEBUG 使用详细格式，其他使用简洁格式）
LOG_FORMAT = DETAILED_FORMAT if LOG_LEVEL == "DEBUG" else SIMPLE_FORMAT


# ===== 配置 logger =====
def _setup_logger():
    """配置 logger（模块导入时自动执行）"""
    
    # 1. 控制台输出（始终开启）
    _logger.add(
        sys.stdout,
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 2. 文件输出（可选）
    if LOG_TO_FILE:
        # 确保日志目录存在
        log_file_path = Path(LOG_FILE)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        _logger.add(
            LOG_FILE,
            format=DETAILED_FORMAT,  # 文件始终使用详细格式
            level=LOG_LEVEL,
            rotation="10 MB",  # 单文件最大 10MB
            retention="30 days",  # 保留 30 天
            compression="zip",  # 压缩旧日志
            backtrace=True,
            diagnose=True,
            encoding="utf-8"
        )
        _logger.info(f"日志文件输出已启用: {LOG_FILE}")


# 初始化 logger
_setup_logger()

# 导出 logger（用户直接使用这个）
logger = _logger

__all__ = ["logger"]
