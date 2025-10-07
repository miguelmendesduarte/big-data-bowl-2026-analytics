"""Logging configuration."""

import sys

from loguru import logger

from .settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure logging settings.

    Args:
        settings (Settings): Application settings.
    """
    logger.remove()
    logger.add(sink=sys.stdout, level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)

    if settings.LOG_FILE:
        logger.add(
            settings.LOG_FILE,
            level=settings.LOG_LEVEL,
            format=settings.LOG_FORMAT,
            rotation="1 MB",
            retention="7 days",
        )
