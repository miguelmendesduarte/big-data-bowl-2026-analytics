"""Dataset readers and writers."""

from pathlib import Path

import pandas as pd
from loguru import logger

from .base import BaseReader, BaseWriter


class CSVReader(BaseReader):
    """Reader for CSV files."""

    def read(self, path: Path) -> pd.DataFrame:
        """Read data from CSV file.

        Args:
            path (Path): Path to CSV file.

        Returns:
            pd.DataFrame: Data read from CSV file.
        """
        logger.debug(f"Reading data from {path}")
        data = pd.read_csv(path, low_memory=False)

        if self.limit is not None:
            data = data.head(self.limit)

        return data


class CSVWriter(BaseWriter):
    """Writer to CSV files."""

    def write(self, data: pd.DataFrame, path: Path) -> None:
        """Write data to CSV file.

        Args:
            data (pd.DataFrame): Data to write.
            path (Path): Path to CSV file.
        """
        logger.debug(f"Writing data to {path}")

        parent_dir = path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        data.to_csv(path, index=False)
