"""Module for creating a dataset of unique players from tracking data."""

from pathlib import Path

import pandas as pd
from loguru import logger

from ...core.settings import get_settings
from ...io.datasets import CSVReader, CSVWriter

settings = get_settings()

PLAYER_COLS = [
    "nfl_id",
    "player_name",
    "player_height",
    "player_weight",
    "player_birth_date",
    "player_position",
]


def load_weekly_players(week: int) -> pd.DataFrame:
    """Load player data from a single week's tracking data.

    Args:
        week (int): Week number to load data for.

    Returns:
        pd.DataFrame: Player data with specified columns, deduplicated by nfl_id.

    Raises:
        ValueError: If required columns are missing or nfl_id contains null values.
        FileNotFoundError: If the tracking data file does not exist.
    """
    input_path = settings.get_tracking_data_path(
        week=week, data_stage="raw", throw_stage="before"
    )
    logger.debug(f"Reading tracking data for week {week} from {input_path}")

    try:
        df = CSVReader().read(input_path)
    except FileNotFoundError as e:
        logger.warning(f"Tracking data file not found for week {week}: {input_path}")
        raise FileNotFoundError(f"Tracking data file not found: {input_path}") from e

    if df.empty:
        logger.warning(f"No data found in {input_path}")
        return pd.DataFrame(columns=PLAYER_COLS)

    missing_cols = [col for col in PLAYER_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in week {week} data: {missing_cols}"
        )

    if df["nfl_id"].isna().any():
        raise ValueError(f"Found null nfl_id values in week {week} data")

    players = df[PLAYER_COLS].drop_duplicates(subset=["nfl_id"])
    logger.info(f"Loaded {len(players)} unique players for week {week}")
    return players


def validate_players_data(players_df: pd.DataFrame) -> pd.DataFrame:
    """Validate the integrity of the players DataFrame.

    Args:
        players_df (pd.DataFrame): Players data to validate.

    Returns:
        pd.DataFrame: Validated players data.

    Raises:
        ValueError: If data contains invalid values (e.g., negative weights).
    """
    if players_df.empty:
        raise ValueError("No player data provided for validation")

    if players_df["player_weight"].le(0).any():
        raise ValueError("Invalid player_weight values (must be positive)")

    logger.debug("Player data validation passed")
    return players_df


def create_players_dataset(
    output_path: Path = settings.CLEANED_DATA_DIR / "players.csv",
) -> None:
    """Create a dataset of unique players from tracking data across all weeks.

    Args:
        output_path (Path): Path to save the players CSV file.
            Defaults to settings.CLEANED_DATA_DIR / "players.csv".

    Raises:
        ValueError: If no valid player data is found across all weeks.
        FileNotFoundError: If any required tracking data files are missing.
    """
    logger.info("Starting creation of players dataset")
    all_players = []

    for week in range(1, settings.NUM_WEEKS + 1):
        try:
            players = load_weekly_players(week)
            if not players.empty:
                all_players.append(players)
        except FileNotFoundError:
            continue  # Skip missing weeks, as warned in load_weekly_players

    if not all_players:
        raise ValueError("No valid player data found across all weeks")

    players_df = (
        pd.concat(all_players, ignore_index=True)
        .pipe(lambda df: df.drop_duplicates(subset=["nfl_id"], ignore_index=True))
        .pipe(validate_players_data)
        .sort_values(by="nfl_id", ascending=True)
    )

    logger.info(f"Created dataset with {len(players_df)} unique players")
    CSVWriter().write(players_df, output_path)


if __name__ == "__main__":
    create_players_dataset()
