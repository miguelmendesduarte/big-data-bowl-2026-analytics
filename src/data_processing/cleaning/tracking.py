"""Module for cleaning tracking data."""

from typing import Any, Literal

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


def load_tracking_data(
    week: int,
    data_stage: Literal["raw", "cleaned", "processed"] = "raw",
    throw_stage: Literal["before", "after"] = "before",
) -> pd.DataFrame:
    """Load tracking data for a given week, data stage, and throw stage.

    Args:
        week (int): The week number for the tracking data to load.
        data_stage (Literal['raw', 'cleaned', 'processed'], optional): The stage of
            the data (raw, cleaned, or processed). Defaults to "raw".
        throw_stage (Literal['before', 'after'], optional): Whether to load "before" or
            "after" snap data.
            Defaults to "before".

    Returns:
        pd.DataFrame: The loaded tracking data as a DataFrame.
    """
    path = settings.get_tracking_data_path(
        week=week, data_stage=data_stage, throw_stage=throw_stage
    )
    return CSVReader().read(path)


def merge_with_player_and_team_data(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Merge tracking data with player and team information.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.

    Returns:
        pd.DataFrame: DataFrame enriched with player and team information.
    """
    tracking_df = add_player_info(tracking_df)
    tracking_df = add_team_and_play_direction(tracking_df)

    return tracking_df.drop_duplicates(
        subset=["game_id", "play_id", "nfl_id", "frame_id"]
    )


def add_player_info(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Add player information to the tracking DataFrame.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.

    Returns:
        pd.DataFrame: DataFrame enriched with player information.
    """
    players_df = pd.read_csv(settings.PLAYERS_FILE)
    players_df = players_df[["nfl_id", "player_name", "player_position"]]

    return pd.merge(tracking_df, players_df, on="nfl_id", how="left")


def add_team_and_play_direction(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Add team information and play direction to the tracking DataFrame.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.

    Returns:
        pd.DataFrame: DataFrame enriched with team information and play direction.
    """
    plays_df = pd.read_csv(settings.CLEANED_PLAYS_FILE)
    plays_df = plays_df[
        ["game_id", "play_id", "possession_team", "defensive_team", "play_direction"]
    ]

    tracking_with_teams = pd.merge(
        tracking_df, plays_df, on=["game_id", "play_id"], how="left"
    )

    def determine_team(row: pd.Series) -> Any:
        """Helper function to determine the team based on the player's side.

        Args:
            row (pd.Series): A row of the tracking DataFrame.

        Returns:
            str: The team name (offensive or defensive) based on player side.
        """
        if row["player_side"] == "Defense":
            return row["defensive_team"]
        elif row["player_side"] == "Offense":
            return row["possession_team"]
        return ""

    tracking_with_teams["team"] = tracking_with_teams.apply(determine_team, axis=1)
    return tracking_with_teams.drop(columns=["possession_team", "defensive_team"])


def convert_plays_left_to_right(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Flip tracking data where plays are right to left.

    Args:
        tracking_df (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data where all plays are left to right.
    """
    right_to_left_plays = tracking_df[tracking_df.play_direction == "left"].copy()

    right_to_left_plays["x"] = settings.FIELD_LENGTH - right_to_left_plays["x"]
    right_to_left_plays["y"] = settings.FIELD_WIDTH - right_to_left_plays["y"]

    right_to_left_plays["x"] = right_to_left_plays["x"].round(2)
    right_to_left_plays["y"] = right_to_left_plays["y"].round(2)

    if "o" in right_to_left_plays.columns and "dir" in right_to_left_plays.columns:
        right_to_left_plays["o"] = (right_to_left_plays["o"] + 180) % 360
        right_to_left_plays["dir"] = (right_to_left_plays["dir"] + 180) % 360

        right_to_left_plays["o"] = right_to_left_plays["o"].round(2)
        right_to_left_plays["dir"] = right_to_left_plays["dir"].round(2)

        tracking_df.loc[right_to_left_plays.index, ["x", "y", "o", "dir"]] = (
            right_to_left_plays[["x", "y", "o", "dir"]]
        )
    else:
        tracking_df.loc[right_to_left_plays.index, ["x", "y"]] = right_to_left_plays[
            ["x", "y"]
        ]

    return tracking_df


def _clean_tracking_data(week: int, throw_stage: Literal["before", "after"]) -> None:
    """Clean tracking data for a specific week and throw stage (before or after snap).

    Args:
        week (int): The week number to clean data for.
        throw_stage (Literal["before", "after"]): Whether to clean data before or
            after the snap.
    """
    before_snap_df = load_tracking_data(week, throw_stage="before")
    after_snap_df = load_tracking_data(week, throw_stage="after")

    if throw_stage == "before":
        # Clean before snap data
        before_snap_df = before_snap_df[TRACKING_COLS_BEFORE_SNAP]
        merged_df = pd.merge(
            before_snap_df,
            after_snap_df[["game_id", "play_id", "nfl_id"]],
            on=["game_id", "play_id", "nfl_id"],
            how="inner",
        )
        file_prefix = "input"
    else:
        # Clean after snap data
        before_snap_unique = before_snap_df[
            ["game_id", "play_id", "nfl_id", "player_side", "player_role"]
        ]
        merged_df = pd.merge(
            after_snap_df,
            before_snap_unique,
            on=["game_id", "play_id", "nfl_id"],
            how="left",
        )
        file_prefix = "output"

    merged_df = merge_with_player_and_team_data(merged_df)
    merged_df = convert_plays_left_to_right(merged_df)

    output_path = settings.CLEANED_DATA_DIR / f"{file_prefix}_2023_w{week:02d}.csv"
    CSVWriter().write(merged_df, output_path)


def clean_tracking_data() -> None:
    """Clean tracking data for all weeks and both throw stages."""
    for week in range(1, settings.NUM_WEEKS + 1):
        _clean_tracking_data(week, "before")
        _clean_tracking_data(week, "after")


if __name__ == "__main__":
    clean_tracking_data()
