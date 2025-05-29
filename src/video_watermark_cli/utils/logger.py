import logging
from logging import StreamHandler, Formatter, FileHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from video_watermark_cli.config import LOG_PATH, LOG_LEVEL

def setup_logger(name: str, log_level=None, max_bytes=10*1024*1024, backup_count=5) -> logging.Logger:
    """
    初始化标准化日志记录器
    Args:
        name: 模块名称 (通常用 __name__)
        log_level: 日志级别 (默认使用 LOG_LEVEL 配置)
        max_bytes: 日志文件最大大小 (默认 10 MB)
        backup_count: 备份文件数量 (默认 5)
    Returns:
        配置好的日志记录器
    """
    # 使用传入的日志级别或默认配置
    level = log_level if log_level is not None else LOG_LEVEL

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 确保日志目录存在
    Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

    # 统一日志格式
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s'
    )

    # 控制台处理器
    console_handler = StreamHandler()
    console_handler.setFormatter(formatter)

    # 文件输出（自动轮转）
    file_handler = RotatingFileHandler(
        os.path.join(LOG_PATH, f"{name}.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 避免重复添加处理器
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger