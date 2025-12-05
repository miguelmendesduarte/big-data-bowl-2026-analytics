"""Global application settings."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

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
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    CLEANED_DATA_DIR: Path = DATA_DIR / "cleaned"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    TRAINING_DATA_DIR: Path = PROCESSED_DATA_DIR / "training"
    ASSETS_DIR: Path = BASE_DIR / "assets"
    LOGOS_DIR: Path = ASSETS_DIR / "logos"

    # Files
    RAW_PLAYS_FILE: Path = RAW_DATA_DIR / "supplementary_data.csv"
    CLEANED_PLAYS_FILE: Path = CLEANED_DATA_DIR / "plays.csv"
    PLAYERS_FILE: Path = CLEANED_DATA_DIR / "players.csv"
    TRAIN_DATA_FILE: Path = TRAINING_DATA_DIR / "train.csv"
    TEST_DATA_FILE: Path = TRAINING_DATA_DIR / "test.csv"

    # Templates
    TRACKING_DATA_BEFORE_THROW_TEMPLATE: ClassVar[str] = "input_2023_w{week:02d}.csv"
    TRACKING_DATA_AFTER_THROW_TEMPLATE: ClassVar[str] = "output_2023_w{week:02d}.csv"
    LOGO_FILE_TEMPLATE: ClassVar[str] = "{team}.png"

    # Variables
    NUM_WEEKS: int = Field(default=18, description="Number of weeks in the season.")
    NUM_TRAIN_WEEKS: int = Field(default=9, description="Number of weeks for training.")
    DB_POSITIONS: list[str] = ["CB", "DB", "FS", "SS"]
    FIGURE_SIZE: tuple[int, int] = Field(
        default=(12, 6), description="Size of the output figure in inches."
    )
    FRAME_RATE: int = Field(
        default=10, description="Frame rate for animations in frames per second."
    )
    BITRATE: int = Field(default=1800, description="Bitrate for video exports in kbps.")
    FIELD_LENGTH: float = Field(
        default=120, description="Length of the field in yards."
    )
    FIELD_WIDTH: float = Field(default=53.3, description="Width of the field in yards.")
    FIELD_COLOR: str = Field(
        default="#3B7A57", description="Hex color code for the field."
    )
    LINE_COLOR: str = Field(
        default="#FFFFFF", description="Hex color code for the field lines."
    )
    BALL_COLOR: str = Field(
        default="#8B5B29", description="Hex color code for the ball."
    )
    LINE_OF_SCRIMMAGE_COLOR: str = Field(
        default="#0000FF", description="Hex color code for the line of scrimmage."
    )
    FIRST_DOWN_LINE_COLOR: str = Field(
        default="#FFFF00", description="Hex color code for the first down line."
    )

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

    def get_tracking_data_path(
        self,
        week: int,
        data_stage: Literal["raw", "cleaned", "processed"] = "raw",
        throw_stage: Literal["before", "after"] = "before",
    ) -> Path:
        """Get the path to a tracking data file.

        Args:
            week (int): Week number.
            data_stage (Literal): "raw", "cleaned", or "processed".
                Default is "raw".
            throw_stage (Literal): "before" or "after".
                Default is "before".

        Raises:
            ValueError: If the week is out of range.
            FileNotFoundError: If the file does not exist.

        Returns:
            Path: Path to the tracking data file.
        """
        if not (1 <= week <= self.NUM_WEEKS):
            raise ValueError(f"Week must be between 1 and {self.NUM_WEEKS}.")

        base_dir_map = {
            "raw": self.RAW_DATA_DIR,
            "cleaned": self.CLEANED_DATA_DIR,
            "processed": self.PROCESSED_DATA_DIR,
        }

        template_map = {
            "before": self.TRACKING_DATA_BEFORE_THROW_TEMPLATE,
            "after": self.TRACKING_DATA_AFTER_THROW_TEMPLATE,
        }

        base_dir = base_dir_map[data_stage]
        template = template_map[throw_stage]
        full_path = base_dir / template.format(week=week)

        if not full_path.exists():
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                full_path.touch(exist_ok=True)  # Creates the file if it doesn't exist
            except Exception as e:
                raise FileNotFoundError(
                    f"File could not be created: {full_path}."
                ) from e

        return full_path


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
