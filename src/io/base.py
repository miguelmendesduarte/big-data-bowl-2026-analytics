"""Base classes for file I/O operations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseReader(ABC):
    """Base class for file readers."""

    def __init__(self, limit: int | None = None) -> None:
        """Initialize the BaseReader.

        Args:
            limit (int | None, optional): Maximum number of rows to read.
                Defaults to None.
        """
        self.limit = limit

    @abstractmethod
    def read(self, path: Path) -> Any:
        """Read data from a file.

        Args:
            path (Path): Path to the file to read.

        Returns:
            Any: Data read from the file.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        raise NotImplementedError()


class BaseWriter(ABC):
    """Base class for file writers."""

    @abstractmethod
    def write(self, data: Any, path: Path) -> None:
        """Write data to a file.

        Args:
            data (Any): Data to write.
            path (Path): Path to the file to write.
        """
        raise NotImplementedError()
