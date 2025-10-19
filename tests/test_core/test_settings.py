"""Tests for the settings module."""

import pytest

from src.core.settings import LogLevel, get_settings


@pytest.mark.parametrize(
    "log_level, expected",
    [
        (LogLevel.DEBUG, "DEBUG"),
        (LogLevel.INFO, "INFO"),
        (LogLevel.WARNING, "WARNING"),
        (LogLevel.ERROR, "ERROR"),
        (LogLevel.CRITICAL, "CRITICAL"),
    ],
)
def test_log_level_vaues(log_level: LogLevel, expected: str) -> None:
    """Test the string representation of LogLevel enum.

    Args:
        log_level (LogLevel): The LogLevel enum to test.
        expected (str): Expected string representation.
    """
    assert log_level.value == expected  # nosec B101


def test_default_settings() -> None:
    """Test the default settings values."""
    settings = get_settings()

    assert settings.LOG_LEVEL == LogLevel.INFO  # nosec B101
    assert settings.LOG_FILE is None  # nosec B101
    assert settings.LOG_FORMAT == (
        "{time:DD/MM/YYYY HH:mm:ss} | {level} | {file}:{line} | {message}"
    )  # nosec B101


def test_get_settings_cache() -> None:
    """Test that get_settings uses caching."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2, "get_settings should return the same instance."  # nosec B101


def test_get_settings_refresh_cache() -> None:
    """Test that get_settings can refresh the cache."""
    settings1 = get_settings()
    settings2 = get_settings(refresh_cache=True)

    assert settings1 is not settings2, (
        "get_settings should return a new instance when refreshed."
    )  # nosec B101
