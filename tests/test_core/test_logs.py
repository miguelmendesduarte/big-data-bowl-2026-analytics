"""Tests for the logging module."""

import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from src.core.logs import configure_logging
from src.core.settings import LogLevel, Settings


@pytest.fixture
def mock_logger(mocker: MockerFixture) -> Generator[MagicMock, None, None]:
    """Fixture to patch and return the mock logger."""
    return mocker.patch("src.core.logs.logger")


def test_configure_logging_stdout_only(mock_logger: MagicMock) -> None:
    """Test logging configuration with stdout only.

    Args:
        mock_logger (MockerFixture): The pytest-mock fixture.
    """
    settings = Settings(LOG_LEVEL=LogLevel.INFO, LOG_FILE=None)

    configure_logging(settings)

    mock_logger.remove.assert_called_once()
    mock_logger.add.assert_called_once_with(
        sink=sys.stdout,
        level=LogLevel.INFO,
        format=settings.LOG_FORMAT,
    )


def test_configure_logging_with_file(mock_logger: MagicMock) -> None:
    """Test logging configuration with a log file.

    Args:
        mock_logger (MockerFixture): The pytest-mock fixture.
    """
    log_file = Path("/tmp/test.log")  # nosec B108
    settings = Settings(LOG_LEVEL=LogLevel.DEBUG, LOG_FILE=log_file)

    configure_logging(settings)

    assert mock_logger.add.call_count == 2  # nosec B101

    file_call = mock_logger.add.call_args_list[1]
    assert file_call[0][0] == log_file  # nosec B101
    assert file_call[1]["rotation"] == "1 MB"  # nosec B101
    assert file_call[1]["retention"] == "7 days"  # nosec B101


@pytest.mark.parametrize(
    "level",
    [
        LogLevel.DEBUG,
        LogLevel.INFO,
        LogLevel.WARNING,
        LogLevel.ERROR,
        LogLevel.CRITICAL,
    ],
)
def test_logging_levels(mock_logger: MagicMock, level: LogLevel) -> None:
    """Test that logging respects different log levels.

    Args:
        mock_logger (MockerFixture): The pytest-mock fixture.
        level (LogLevel): The log level to test.
    """
    settings = Settings(LOG_LEVEL=level, LOG_FILE=None)

    configure_logging(settings)

    mock_logger.remove.assert_called_once()
    mock_logger.add.assert_called_once_with(
        sink=sys.stdout,
        level=level,
        format=settings.LOG_FORMAT,
    )
