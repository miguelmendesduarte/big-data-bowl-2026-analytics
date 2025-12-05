"""Module for cleaning tracking data."""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...core.settings import get_settings
from ...io.datasets import CSVReader, CSVWriter

settings = get_settings()

TRACKING_COLS_BEFORE_throw = [
    "game_id",
    "play_id",
    "nfl_id",
    "frame_id",
    "play_direction",
    "player_side",
    "player_role",
    "x",
    "y",
    "s",
    "a",
    "dir",
    "o",
]


def filter_before_throw(before_throw_tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Filter tracking data to only include columns relevant before the throw.

    Args:
        before_throw_tracking_df (pd.DataFrame): Tracking data before the throw.

    Returns:
        pd.DataFrame: Filtered tracking data before the throw.
    """
    missing_cols = set(TRACKING_COLS_BEFORE_throw) - set(
        before_throw_tracking_df.columns
    )
    if missing_cols:
        logger.warning(f"Missing columns in before_throw data: {missing_cols}")
    return before_throw_tracking_df[TRACKING_COLS_BEFORE_throw].copy()


def add_before_throw_columns_to_after(
    before_throw_tracking_df: pd.DataFrame, after_throw_tracking_df: pd.DataFrame
) -> pd.DataFrame:
    """Add player_side and player_role from before_throw to after_throw tracking data.

    Merges on game_id, play_id, nfl_id to add the columns.

    Args:
        before_throw_tracking_df (pd.DataFrame): Filtered tracking data before
        the throw.
        after_throw_tracking_df (pd.DataFrame): Tracking data after the throw.

    Returns:
        pd.DataFrame: After throw tracking data with added columns.
    """
    cols_to_add = [
        "game_id",
        "play_id",
        "nfl_id",
        "player_side",
        "player_role",
        "play_direction",
    ]
    missing_cols = set(cols_to_add) - set(before_throw_tracking_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in before_throw data: {missing_cols}"
        )

    before_slice = before_throw_tracking_df[cols_to_add].drop_duplicates()
    merge_keys = ["game_id", "play_id", "nfl_id"]
    result = pd.merge(
        after_throw_tracking_df,
        before_slice,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    if result["player_side"].isnull().any() or result["player_role"].isnull().any():
        logger.warning(
            "Some rows in after_throw data couldn't be matched with before_throw."
        )
    return result


def add_player_info(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Add player name and position to the tracking DataFrame.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.

    Returns:
        pd.DataFrame: DataFrame enriched with player information.
    """
    reader = CSVReader()
    players_df = reader.read(settings.PLAYERS_FILE)
    cols_to_add = ["nfl_id", "player_name", "player_position"]
    missing_cols = set(cols_to_add) - set(players_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in players data: {missing_cols}")

    players_slice = players_df[cols_to_add].drop_duplicates()
    result = pd.merge(
        tracking_df,
        players_slice,
        on="nfl_id",
        how="left",
        validate="many_to_one",
    )
    if result["player_name"].isnull().any() or result["player_position"].isnull().any():
        logger.warning(
            "Some players in tracking data could not be matched with player info."
        )
    return result


def _determine_team(row: pd.Series) -> Optional[str]:
    """Helper function to determine the team based on the player's side.

    Args:
        row (pd.Series): A row of the tracking DataFrame.

    Returns:
        Optional[str]: The team name (offensive or defensive) based on player side,
        or None if undetermined.
    """
    if row["player_side"] == "Defense":
        return row.get("defensive_team")
    elif row["player_side"] == "Offense":
        return row.get("possession_team")
    return None


def add_team_info(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Add team name to the tracking DataFrame based on player side.

    Args:
        tracking_df (pd.DataFrame): DataFrame containing tracking data.

    Returns:
        pd.DataFrame: DataFrame enriched with team information.
    """
    reader = CSVReader()
    plays_df = reader.read(settings.RAW_PLAYS_FILE)
    plays_df = plays_df[["game_id", "play_id", "possession_team", "defensive_team"]]

    merged_df = pd.merge(
        tracking_df,
        plays_df,
        on=["game_id", "play_id"],
        how="left",
    )
    merged_df["team"] = merged_df.apply(_determine_team, axis=1)
    if merged_df["team"].isnull().any():
        logger.warning("Some rows could not determine team based on player_side.")
    return merged_df.drop(
        columns=["possession_team", "defensive_team"], errors="ignore"
    )


def convert_plays_left_to_right(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize play direction to left-to-right.

    Args:
        tracking_df (pd.DataFrame): Tracking data with `play_direction`, `x`, `y`,
        and optionally `o`, `dir`.

    Returns:
        pd.DataFrame: Tracking data with standardized coordinates and directions.
        `play_direction` column is removed.
    """
    if "play_direction" not in tracking_df.columns:
        logger.warning("play_direction column missing — skipping direction flip.")
        return tracking_df.drop(columns=["play_direction"], errors="ignore")

    df = tracking_df.copy()

    rtl_mask = df["play_direction"] == "left"

    if rtl_mask.any():
        logger.info(
            f"Flipping {rtl_mask.sum():,} frames from right-to-left to left-to-right."
        )

        df.loc[rtl_mask, "x"] = settings.FIELD_LENGTH - df.loc[rtl_mask, "x"]
        df.loc[rtl_mask, "y"] = settings.FIELD_WIDTH - df.loc[rtl_mask, "y"]
        df.loc[rtl_mask, ["x", "y"]] = df.loc[rtl_mask, ["x", "y"]].round(2)

        if {"o", "dir"}.issubset(df.columns):
            df.loc[rtl_mask, "o"] = (df.loc[rtl_mask, "o"] + 180) % 360
            df.loc[rtl_mask, "dir"] = (df.loc[rtl_mask, "dir"] + 180) % 360
            df.loc[rtl_mask, ["o", "dir"]] = df.loc[rtl_mask, ["o", "dir"]].round(2)

    return df


def filter_before_throw_to_after_throw_players(
    before_throw_df: pd.DataFrame,
    after_throw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only players pre-throw that also appear post-throw.

    Args:
        before_throw_df (pd.DataFrame): Tracking data before the throw.
        after_throw_df (pd.DataFrame): Tracking data after the throw.

    Returns:
        pd.DataFrame: Filtered tracking data.
    """
    before_count = len(before_throw_df)

    # Merge to keep only matching (game_id, play_id, nfl_id)
    filtered = pd.merge(
        before_throw_df,
        after_throw_df[["game_id", "play_id", "nfl_id"]].drop_duplicates(),
        on=["game_id", "play_id", "nfl_id"],
        how="inner",
    )

    after_count = len(filtered)
    logger.info(
        f"Before-throw: {before_count:,} → {after_count:,} rows"
        f" (kept players in after-throw)"
    )

    return filtered


def sync_players_across_stages(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only players that "survived" the before-throw pipeline.

    Args:
        before_df (pd.DataFrame): Tracking data before the throw.
        after_df (pd.DataFrame): Tracking data after the throw.

    Returns:
        pd.DataFrame: Synchronized tracking data after the throw.
    """
    keep_keys = before_df[["game_id", "play_id", "nfl_id"]].drop_duplicates()

    before_count = len(after_df)
    synced = pd.merge(
        after_df,
        keep_keys,
        on=["game_id", "play_id", "nfl_id"],
        how="inner",
        validate="many_to_one",
    )
    after_count = len(synced)

    logger.info(
        f"Sync-player step: {before_count:,} to {after_count:,} rows "
        f"({before_count - after_count:,} rows removed because the player "
        f"was filtered out pre-throw)."
    )
    return synced


def get_plays_with_one_receiver_and_one_db(df: pd.DataFrame) -> pd.DataFrame:
    """Filter tracking data for 1-on-1 plays (targeted receiver vs. defensive back).

    A Defensive Back (DB) is defined by player_position being in DB_POSITIONS.

    Args:
        df: The input DataFrame containing tracking data.

    Returns:
        A DataFrame containing the tracking data for plays that meet the criteria.
    """
    plays = df[["game_id", "play_id"]].drop_duplicates()

    receiver_counts = (
        df[df["player_role"] == "Targeted Receiver"]
        .groupby(["game_id", "play_id"])["nfl_id"]
        .nunique()
        .reset_index(name="receiver_count")
    )
    db_counts = (
        df[
            (df["player_side"] == "Defense")
            & (df["player_position"].isin(settings.DB_POSITIONS))
        ]
        .groupby(["game_id", "play_id"])["nfl_id"]
        .nunique()
        .reset_index(name="db_count")
    )

    play_counts = pd.merge(
        plays, receiver_counts, on=["game_id", "play_id"], how="left"
    )
    play_counts = pd.merge(
        play_counts, db_counts, on=["game_id", "play_id"], how="left"
    )

    play_counts["receiver_count"] = play_counts["receiver_count"].fillna(0)
    play_counts["db_count"] = play_counts["db_count"].fillna(0)

    # Keep only plays with exactly one receiver and one DB
    filtered_plays_keys = play_counts[
        (play_counts["receiver_count"] == 1) & (play_counts["db_count"] == 1)
    ][["game_id", "play_id"]]

    final_df = pd.merge(df, filtered_plays_keys, on=["game_id", "play_id"], how="inner")

    return final_df


def filter_plays_by_ball_landing_distance(
    tracking_df: pd.DataFrame, plays_df: pd.DataFrame
) -> pd.DataFrame:
    """Filter plays where the ball landing spot is <3 yards of the receiver or defender.

    Args:
        tracking_df: DataFrame with tracking data for the receiver/defender pair,
            usually from get_closest_defender.
        plays_df: DataFrame with ball landing coordinates.

    Returns:
        A DataFrame with tracking data that meet the ball landing distance criteria.
    """
    # Identify the last frame for each play
    last_frame_keys = (
        tracking_df.groupby(["game_id", "play_id"])["frame_id"].max().reset_index()
    )
    last_pos_df = pd.merge(
        tracking_df, last_frame_keys, on=["game_id", "play_id", "frame_id"], how="inner"
    )

    last_pos_pivot = last_pos_df.pivot_table(
        index=["game_id", "play_id"],
        columns="player_side",
        values=["x", "y"],
        aggfunc="first",  # We expect max one of each side per play/frame
    ).reset_index()

    last_pos_pivot.columns = [
        "_".join(str(c) for c in col).strip("_")
        for col in last_pos_pivot.columns.values
    ]

    last_pos_pivot = last_pos_pivot.rename(
        columns={
            "x_Offense": "x_rec_last",
            "y_Offense": "y_rec_last",
            "x_Defense": "x_def_last",
            "y_Defense": "y_def_last",
        }
    )

    merged_with_ball = pd.merge(
        last_pos_pivot,
        plays_df[["game_id", "play_id", "ball_land_x", "ball_land_y"]],
        on=["game_id", "play_id"],
        how="inner",
    )

    # Euclidean distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
    merged_with_ball["dist_rec_to_ball"] = np.sqrt(
        (merged_with_ball["x_rec_last"] - merged_with_ball["ball_land_x"]) ** 2
        + (merged_with_ball["y_rec_last"] - merged_with_ball["ball_land_y"]) ** 2
    )
    merged_with_ball["dist_def_to_ball"] = np.sqrt(
        (merged_with_ball["x_def_last"] - merged_with_ball["ball_land_x"]) ** 2
        + (merged_with_ball["y_def_last"] - merged_with_ball["ball_land_y"]) ** 2
    )

    # Keep play if distance to receiver < 3 OR distance to defender < 3
    plays_to_keep = merged_with_ball[
        (merged_with_ball["dist_rec_to_ball"] < 3)
        | (merged_with_ball["dist_def_to_ball"] < 3)
    ][["game_id", "play_id"]]

    final_filtered_tracking_df = pd.merge(
        tracking_df, plays_to_keep, on=["game_id", "play_id"], how="inner"
    )

    return final_filtered_tracking_df


def get_closest_defender(df: pd.DataFrame) -> pd.DataFrame:
    """Get the defensive player with lowest mean distance to the receiver.

    Args:
        df: The input DataFrame containing tracking data.

    Returns:
        A DataFrame containing only the tracking data rows for the closest
            defender(s) for each unique game_id/play_id combination.
    """
    receiver_data_full = df[df["player_role"] == "Targeted Receiver"].copy()

    receiver_coords = receiver_data_full[
        ["game_id", "play_id", "frame_id", "x", "y"]
    ].rename(columns={"x": "x_rec", "y": "y_rec"})

    defenders_data = df[df["player_side"] == "Defense"].copy()

    merged_df = pd.merge(
        defenders_data,
        receiver_coords,
        on=["game_id", "play_id", "frame_id"],
        how="inner",
    )

    merged_df["distance"] = np.sqrt(
        (merged_df["x"] - merged_df["x_rec"]) ** 2
        + (merged_df["y"] - merged_df["y_rec"]) ** 2
    )

    mean_distance_df = (
        merged_df.groupby(["game_id", "play_id", "nfl_id"])["distance"]
        .mean()
        .reset_index(name="mean_distance")
    )

    closest_defender_indices = mean_distance_df.loc[
        mean_distance_df.groupby(["game_id", "play_id"])["mean_distance"].idxmin()
    ]

    closest_defenders_keys = closest_defender_indices[["game_id", "play_id", "nfl_id"]]

    closest_defender_df = pd.merge(
        defenders_data,
        closest_defenders_keys,
        on=["game_id", "play_id", "nfl_id"],
        how="inner",
    )

    final_filtered_df = pd.concat(
        [closest_defender_df, receiver_data_full], ignore_index=True
    )

    final_filtered_df = final_filtered_df.sort_values(
        ["game_id", "play_id", "frame_id", "player_side"],
        ascending=[
            True,
            True,
            True,
            False,
        ],  # Defense comes after Offense in string sort, so False pushes it up
    )

    return final_filtered_df


def clean_tracking_data() -> None:
    """Clean tracking data for all weeks and both before/after throw stages.

    Processes each week's data separately, applies cleaning steps, and
    saves to cleaned paths.
    """
    reader = CSVReader()
    writer = CSVWriter()

    plays_df = CSVReader().read(settings.CLEANED_PLAYS_FILE)

    for week in range(1, settings.NUM_WEEKS + 1):
        logger.info(f"Cleaning tracking data for week {week}...")

        before_throw_path = settings.get_tracking_data_path(week, "raw", "before")
        after_throw_path = settings.get_tracking_data_path(week, "raw", "after")

        try:
            before_throw_df = reader.read(before_throw_path)
            after_throw_df = reader.read(after_throw_path)
        except FileNotFoundError as e:
            logger.error(f"File not found for week {week}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error reading data for week {week}: {e}")
            continue

        qb_df = before_throw_df[before_throw_df["player_role"] == "Passer"].copy()
        qb_df = qb_df[TRACKING_COLS_BEFORE_throw]
        qb_df = add_player_info(qb_df)
        qb_df = add_team_info(qb_df)

        before_throw_df = filter_before_throw_to_after_throw_players(
            before_throw_df, after_throw_df
        )

        # Process before_throw
        filtered_before = filter_before_throw(before_throw_df)
        filtered_before = convert_plays_left_to_right(filtered_before)
        filtered_before = add_player_info(filtered_before)
        filtered_before = add_team_info(filtered_before)

        # Process after_throw
        after_with_before = add_before_throw_columns_to_after(
            filtered_before, after_throw_df
        )

        after_with_before = get_closest_defender(after_with_before)
        filtered_before = filter_before_throw_to_after_throw_players(
            filtered_before, after_with_before
        )

        after_with_before = sync_players_across_stages(
            filtered_before, after_with_before
        )
        after_with_before = convert_plays_left_to_right(after_with_before)
        after_with_before = add_player_info(after_with_before)
        after_with_before = add_team_info(after_with_before)

        after_with_before = get_plays_with_one_receiver_and_one_db(after_with_before)
        after_with_before = filter_plays_by_ball_landing_distance(
            after_with_before, plays_df
        )
        filtered_before = filter_before_throw_to_after_throw_players(
            filtered_before, after_with_before
        )

        filtered_before = pd.concat([filtered_before, qb_df], ignore_index=True)
        filtered_before = filtered_before[
            filtered_before.groupby(["game_id", "play_id"])["player_role"].transform(
                "nunique"
            )
            == 3
        ].copy()
        filtered_before = filtered_before.sort_values(
            ["game_id", "play_id", "frame_id"],
            ascending=[
                True,
                True,
                True,
            ],
        )

        # Save cleaned data
        cleaned_before_path = settings.get_tracking_data_path(week, "cleaned", "before")
        cleaned_after_path = settings.get_tracking_data_path(week, "cleaned", "after")

        writer = CSVWriter()
        try:
            writer.write(filtered_before, cleaned_before_path)
            writer.write(after_with_before, cleaned_after_path)
            logger.info(f"Cleaned data saved for week {week}.")
        except Exception as e:
            logger.error(f"Error writing cleaned data for week {week}: {e}")


if __name__ == "__main__":
    clean_tracking_data()
