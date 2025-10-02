"""Global application settings."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.absolute()


class LogLevel(StrEnum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Directories
    DATA_DIR: Path = BASE_DIR / "data"
    PROMPTS_DIR: Path = BASE_DIR / "prompts"

    # Logging
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO, description="Logging level for the application."
    )
    LOG_FORMAT: str = Field(
        default="{time:DD/MM/YYYY HH:mm:ss} | {level} | {file}:{line} | {message}",
        description="Format for log messages.",
    )
    LOG_FILE: Path | None = Field(
        default=None,
        description="File path for logging. If None, logs won't be written to a file.",
    )


@lru_cache()
def get_settings(refresh_cache: bool = False) -> Settings:
    """Get application settings.

    Args:
        refresh_cache (bool, optional): Whether to clear the cache and reload settings.
            Defaults to False.

    Returns:
        Settings: Application settings instance.
    """
    if refresh_cache:
        get_settings.cache_clear()
    return Settings()
