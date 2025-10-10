"""Module for plays data processing."""

import pandas as pd

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


def create_plays_dataset() -> None:
    """Create a dataset of plays merged with tracking data.

    Raises:
        ValueError: If required columns are missing in the data.
    """
    df = CSVReader().read(settings.RAW_PLAYS_FILE)

    if not all(col in df.columns for col in PLAY_COLS):
        raise ValueError(f"Missing required columns in plays data: {PLAY_COLS}")

    plays_df = df[PLAY_COLS]
    plays_df = plays_df.drop_duplicates(
        subset=["game_id", "play_id"], ignore_index=True
    )
    plays_df = plays_df.sort_values(by=["game_id", "play_id"], ascending=True)
    plays_df.reset_index(drop=True, inplace=True)

    all_tracking_data = []

    for week in range(1, settings.NUM_WEEKS + 1):
        input_path = settings.get_tracking_data_path(
            week=week,
            data_stage="raw",
            throw_stage="before",
        )

        df_tracking = CSVReader().read(input_path)

        if not all(col in df_tracking.columns for col in TRACKING_COLS):
            raise ValueError(
                f"Missing required columns in tracking data for week {week}: "
                f"{TRACKING_COLS}"
            )

        tracking_df = df_tracking[TRACKING_COLS]
        tracking_df = tracking_df.drop_duplicates(
            subset=["game_id", "play_id"], ignore_index=True
        )

        all_tracking_data.append(tracking_df)

    if all_tracking_data:
        tracking_df_combined = pd.concat(all_tracking_data, ignore_index=True)

        merged_df = pd.merge(
            plays_df, tracking_df_combined, on=["game_id", "play_id"], how="left"
        )
        merged_df = merged_df.sort_values(by=["game_id", "play_id"], ascending=True)
        merged_df.reset_index(drop=True, inplace=True)

        output_path = settings.CLEANED_DATA_DIR / "plays.csv"
        CSVWriter().write(merged_df, output_path)


if __name__ == "__main__":
    create_plays_dataset()
