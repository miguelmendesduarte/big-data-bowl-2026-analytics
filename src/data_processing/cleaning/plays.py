"""Module for cleaning and processing plays data."""

from pathlib import Path

import pandas as pd
from loguru import logger

from ...core.settings import get_settings
from ...io.datasets import CSVReader, CSVWriter

settings = get_settings()

PLAY_COLS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "quarter",
    "game_clock",
    "down",
    "home_team_abbr",
    "visitor_team_abbr",
    "play_description",
    "yards_to_go",
    "possession_team",
    "defensive_team",
    "yardline_number",
    "play_nullified_by_penalty",
    "pass_result",
    "pass_length",
    "offense_formation",
    "receiver_alignment",
    "route_of_targeted_receiver",
    "play_action",
    "dropback_type",
    "dropback_distance",
    "team_coverage_man_zone",
    "team_coverage_type",
]

TRACKING_COLS = [
    "game_id",
    "play_id",
    "play_direction",
    "absolute_yardline_number",
    "ball_land_x",
    "ball_land_y",
]

RECEIVER_ROUTES = ["IN", "OUT", "HITCH"]

DEFENSIVE_COVERAGE = "MAN_COVERAGE"


def filter_plays_columns(plays_df: pd.DataFrame) -> pd.DataFrame:
    """Filter columns from plays data.

    Args:
        plays_df (pd.DataFrame): Plays data.

    Returns:
        pd.DataFrame: Filtered plays data.

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
    """
    if plays_df.empty:
        raise ValueError("Input DataFrame is empty")

    missing_cols = [col for col in PLAY_COLS if col not in plays_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return plays_df[PLAY_COLS].drop_duplicates(
        subset=["game_id", "play_id"], ignore_index=True
    )


def filter_receiver_routes(plays_df: pd.DataFrame) -> pd.DataFrame:
    """Filter plays to keep only those with specific receiver routes.

    Args:
        plays_df (pd.DataFrame): Plays data.

    Returns:
        pd.DataFrame: Filtered plays data.

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
    """
    if plays_df.empty:
        raise ValueError("Input DataFrame is empty")
    if "route_of_targeted_receiver" not in plays_df.columns:
        raise ValueError("Missing required column: route_of_targeted_receiver")

    return plays_df[
        plays_df["route_of_targeted_receiver"].isin(RECEIVER_ROUTES)
    ].reset_index(drop=True)


def filter_by_defensive_coverage(plays_df: pd.DataFrame) -> pd.DataFrame:
    """Filter plays to keep only those with the specific defensive scheme.

    Args:
        plays_df (pd.DataFrame): Plays data.

    Returns:
        pd.DataFrame: Filtered plays data.

    Raises:
        ValueError: If the input DataFrame is empty or missing required columns.
    """
    if plays_df.empty:
        raise ValueError("Input DataFrame is empty")
    if "team_coverage_man_zone" not in plays_df.columns:
        raise ValueError("Missing required column: team_coverage_man_zone")

    return plays_df[
        plays_df["team_coverage_man_zone"] == DEFENSIVE_COVERAGE
    ].reset_index(drop=True)


def add_tracking_columns(plays_df: pd.DataFrame) -> pd.DataFrame:
    """Add tracking-related columns to plays data.

    Args:
        plays_df (pd.DataFrame): Plays data.

    Returns:
        pd.DataFrame: Plays data with tracking columns.
    """
    tracking_files = [
        settings.get_tracking_data_path(
            week=week, data_stage="raw", throw_stage="before"
        )
        for week in range(1, settings.NUM_WEEKS + 1)
    ]

    all_tracking_data = []
    for file in tracking_files:
        df_tracking = CSVReader().read(file)

        missing_cols = [col for col in TRACKING_COLS if col not in df_tracking.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {file}: {missing_cols}")

        all_tracking_data.append(df_tracking[TRACKING_COLS])

    if not all_tracking_data:
        raise ValueError("No tracking data found.")

    tracking_df_combined = pd.concat(
        all_tracking_data, ignore_index=True
    ).drop_duplicates(subset=["game_id", "play_id"], ignore_index=True)

    return plays_df.merge(tracking_df_combined, on=["game_id", "play_id"], how="inner")


def adjust_ball_landing_positions(plays_df: pd.DataFrame) -> pd.DataFrame:
    """Adjust ball landing positions based on play direction.

    If the play direction is "left", the ball landing position is adjusted
    to the opposite side of the field (since plays are played right-to-left).

    Args:
        plays_df (pd.DataFrame): Plays data with tracking columns.

    Returns:
        pd.DataFrame: Plays data with adjusted ball landing positions.

    Raises:
        ValueError: If required columns are missing in the data.
    """
    required_cols = ["play_direction", "ball_land_x", "ball_land_y"]
    missing_cols = [col for col in required_cols if col not in plays_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    right_to_left_plays = plays_df[plays_df.play_direction == "left"].copy()

    right_to_left_plays["ball_land_x"] = (
        settings.FIELD_LENGTH - right_to_left_plays["ball_land_x"]
    )
    right_to_left_plays["ball_land_y"] = (
        settings.FIELD_WIDTH - right_to_left_plays["ball_land_y"]
    )

    right_to_left_plays["ball_land_x"] = right_to_left_plays["ball_land_x"].round(2)
    right_to_left_plays["ball_land_y"] = right_to_left_plays["ball_land_y"].round(2)

    plays_df.loc[right_to_left_plays.index, ["ball_land_x", "ball_land_y"]] = (
        right_to_left_plays[["ball_land_x", "ball_land_y"]]
    )

    return plays_df


def process_plays_data(
    input_path: Path = settings.RAW_PLAYS_FILE,
    output_path: Path = settings.CLEANED_PLAYS_FILE,
) -> None:
    """Process and clean plays data, combining with tracking data.

    Args:
        input_path (Path): Path to raw plays CSV file.
            Defaults to settings.RAW_PLAYS_FILE.
        output_path (Path): Path to save cleaned plays CSV file.
            Defaults to settings.CLEANED_PLAYS_FILE.

    Raises:
        ValueError: If input data is invalid or processing fails.
    """
    plays_df = CSVReader().read(input_path)
    if plays_df.empty:
        raise ValueError(f"No data found in {input_path}")
    logger.info(f"Loaded {len(plays_df)} plays")

    plays_df = (
        plays_df.pipe(filter_plays_columns)
        .pipe(filter_receiver_routes)
        # .pipe(filter_by_defensive_coverage)
        .pipe(add_tracking_columns)
        .pipe(adjust_ball_landing_positions)
        .sort_values(by=["game_id", "play_id"])
        .reset_index(drop=True)
    )

    logger.info(f"Processed {len(plays_df)} plays after cleaning")
    CSVWriter().write(plays_df, output_path)


def filter_plays_with_tracking(
    plays_path: Path = settings.CLEANED_PLAYS_FILE,
    output_path: Path = settings.CLEANED_PLAYS_FILE,
) -> None:
    """Filter plays to keep only those present in processed tracking data.

    Args:
        plays_path (Path): Path to the processed plays CSV file.
            Defaults to settings.CLEANED_PLAYS_FILE.
        output_path (Path): Path to save the filtered plays CSV file.
            Defaults to settings.CLEANED_PLAYS_FILE.

    Raises:
        ValueError: If plays data is empty, required columns are missing,
            or no tracking data is found.
    """
    plays_df = CSVReader().read(plays_path)
    if plays_df.empty:
        raise ValueError(f"No data found in {plays_path}")
    missing_cols = [
        col for col in ["game_id", "play_id"] if col not in plays_df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns in plays data: {missing_cols}")
    logger.info(f"Loaded {len(plays_df)} plays")

    tracking_files = [
        settings.get_tracking_data_path(
            week=week, data_stage="cleaned", throw_stage="before"
        )
        for week in range(1, settings.NUM_WEEKS + 1)
    ]

    all_tracking_data = []
    for file in tracking_files:
        df_tracking = CSVReader().read(file)
        missing_cols = [
            col for col in ["game_id", "play_id"] if col not in df_tracking.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in tracking data {file}: {missing_cols}"
            )
        all_tracking_data.append(df_tracking[["game_id", "play_id"]])

    if not all_tracking_data:
        raise ValueError("No tracking data found.")

    tracking_df_combined = pd.concat(
        all_tracking_data, ignore_index=True
    ).drop_duplicates(subset=["game_id", "play_id"], ignore_index=True)
    logger.info(f"Loaded {len(tracking_df_combined)} unique tracking records")

    filtered_plays_df = (
        plays_df.merge(
            tracking_df_combined[["game_id", "play_id"]],
            on=["game_id", "play_id"],
            how="inner",
        )
        .sort_values(by=["game_id", "play_id"])
        .reset_index(drop=True)
    )

    dropped_plays = len(plays_df) - len(filtered_plays_df)
    if dropped_plays > 0:
        logger.warning(f"Dropped {dropped_plays} plays not present in tracking data")
    logger.info(f"Retained {len(filtered_plays_df)} plays after filtering")

    CSVWriter().write(filtered_plays_df, output_path)


if __name__ == "__main__":
    process_plays_data()
