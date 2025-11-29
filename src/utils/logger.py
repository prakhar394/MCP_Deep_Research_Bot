from loguru import logger
import sys
import os

_level = os.getenv("LOG_LEVEL", "INFO")

logger.remove()
logger.add(
    sys.stdout,
    level=_level,
    format="<green>{time}</green> | <level>{level}</level> | <cyan>{module}</cyan> | {message}",
)


def get_logger(name: str):
    return logger.bind(module=name)
