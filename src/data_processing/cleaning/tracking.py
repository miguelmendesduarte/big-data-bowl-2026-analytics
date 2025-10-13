"""Module for cleaning and processing tracking data."""

from typing import Any

import pandas as pd

from ...core.settings import get_settings
from ...io.datasets import CSVReader, CSVWriter

settings = get_settings()

TRACKING_COLS_BEFORE_SNAP = [
    "game_id",
    "play_id",
    "nfl_id",
    "frame_id",
    "player_side",
    "player_role",
    "x",
    "y",
    "s",
    "a",
    "dir",
    "o",
]


def clean_tracking_data_before_snap() -> None:
    """Clean and process tracking data before the snap.

    We keep only players that are present in both before and after snap data.
    """
    for week in range(1, settings.NUM_WEEKS + 1):
        input_path_before = settings.get_tracking_data_path(
            week=week,
            data_stage="raw",
            throw_stage="before",
        )
        before_snap_df = CSVReader().read(input_path_before)

        before_snap_df = before_snap_df[TRACKING_COLS_BEFORE_SNAP]

        input_path_after = settings.get_tracking_data_path(
            week=week,
            data_stage="raw",
            throw_stage="after",
        )
        after_snap_df = CSVReader().read(input_path_after)

        merged_df = pd.merge(
            before_snap_df,
            after_snap_df[["game_id", "play_id", "nfl_id"]],
            on=["game_id", "play_id", "nfl_id"],
            how="inner",  # Only keep rows that exist in both before and after snap
        )
        merged_df = add_player_info(merged_df)
        merged_df = add_team(merged_df)
        merged_df = merged_df.drop_duplicates(
            subset=["game_id", "play_id", "nfl_id", "frame_id"]
        )

        output_path = settings.CLEANED_DATA_DIR / f"input_2023_w{week:02d}.csv"
        CSVWriter().write(merged_df, output_path)


def clean_tracking_data_after_snap() -> None:
    """Clean and process tracking data after the snap."""
    for week in range(1, settings.NUM_WEEKS + 1):
        input_path = settings.get_tracking_data_path(
            week=week,
            data_stage="raw",
            throw_stage="before",
        )
        before_snap_df = CSVReader().read(input_path)

        input_path = settings.get_tracking_data_path(
            week=week,
            data_stage="raw",
            throw_stage="after",
        )
        after_snap_df = CSVReader().read(input_path)

        before_snap_unique = before_snap_df[
            ["game_id", "play_id", "nfl_id", "player_side", "player_role"]
        ]

        merged_df = pd.merge(
            after_snap_df,
            before_snap_unique,
            on=["game_id", "play_id", "nfl_id"],
            how="left",
        )
        merged_df = add_player_info(merged_df)
        merged_df = add_team(merged_df)
        merged_df = merged_df.drop_duplicates(
            subset=["game_id", "play_id", "nfl_id", "frame_id"]
        )

        output_path = settings.CLEANED_DATA_DIR / f"output_2023_w{week:02d}.csv"
        CSVWriter().write(merged_df, output_path)


def add_player_info(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich tracking data with player information such as name and position.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.

    Returns:
        pd.DataFrame: Enriched DataFrame with player information.
    """
    players_df = pd.read_csv(settings.CLEANED_DATA_DIR / "players.csv")
    players_df = players_df[["nfl_id", "player_name", "player_position"]]

    enriched_df = pd.merge(
        tracking_df,
        players_df,
        on="nfl_id",
        how="left",  # Using left join to keep all rows from the tracking data
    )

    return enriched_df


def add_team(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Add team to the tracking data.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.

    Returns:
        pd.DataFrame: DataFrame enriched with team name.
    """
    plays_df = pd.read_csv(settings.CLEANED_DATA_DIR / "plays.csv")
    plays_df = plays_df[["game_id", "play_id", "possession_team", "defensive_team"]]

    tracking_with_teams = pd.merge(
        tracking_df,
        plays_df,
        on=["game_id", "play_id"],
        how="left",
    )

    def determine_team(row: pd.Series) -> Any:
        if row["player_side"] == "Defense":
            return row["defensive_team"]
        elif row["player_side"] == "Offense":
            return row["possession_team"]
        else:
            return ""

    tracking_with_teams["team"] = tracking_with_teams.apply(determine_team, axis=1)
    tracking_with_teams = tracking_with_teams.drop(
        columns=["possession_team", "defensive_team"]
    )

    return tracking_with_teams


if __name__ == "__main__":
    clean_tracking_data_before_snap()
    clean_tracking_data_after_snap()
