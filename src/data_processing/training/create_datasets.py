"""Module for creating the training and testing datasets."""

from typing import Generator, Literal

import pandas as pd

from ...core.settings import get_settings
from ...io.datasets import CSVReader, CSVWriter
from .features import add_features

settings = get_settings()

DataStage = Literal["raw", "cleaned", "processed"]


def get_last_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Get the last frame row for each player in each play.

    Args:
        df (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Dataframe with only the last frames for each play.
    """
    return df.groupby(["game_id", "play_id", "nfl_id"]).tail(1)


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target column.

    The binary target column is 1 if the pass was not a completion, 0 otherwise.

    Args:
        df (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data with binary target column.
    """
    df["is_non_completion"] = (df["pass_result"] != "C").astype(int)

    df = df.drop(columns=["pass_result"])

    return df


def process_single_week(
    tracking_df: pd.DataFrame, plays_df: pd.DataFrame
) -> pd.DataFrame:
    """Process tracking data from a week.

    Args:
        tracking_df (pd.DataFrame): Tracking data.
        plays_df (pd.DataFrame): Plays data.

    Returns:
        pd.DataFrame: Processed tracking data.
    """
    last_frame_df = get_last_frame(tracking_df)

    processed_df = last_frame_df.merge(plays_df, on=["game_id", "play_id"], how="left")
    processed_df = create_binary_target(processed_df)

    processed_df = add_features(processed_df)

    return processed_df


def process_weekly_data_generator(
    week_range: range,
    plays_df: pd.DataFrame,
    reader: CSVReader,
    stage: DataStage = "cleaned",
) -> Generator[pd.DataFrame, None, None]:
    """Process tracking data from multiple weeks.

    Args:
        week_range (range): Range of weeks to process.
        plays_df (pd.DataFrame): Plays data.
        reader (CSVReader): Reader for CSV files.
        stage (DataStage, optional): Type of data to process.
            Defaults to "cleaned".

    Yields:
        Generator[pd.DataFrame, None, None]: Processed tracking data.
    """
    for week in week_range:
        tracking_path = settings.get_tracking_data_path(week, stage, "before")

        weekly_df = reader.read(tracking_path)

        yield process_single_week(weekly_df, plays_df)


def create_datasets() -> None:
    """Create the training and testing datasets."""
    reader = CSVReader()
    writer = CSVWriter()

    plays_df = reader.read(settings.CLEANED_PLAYS_FILE)
    plays_df = plays_df[["game_id", "play_id", "pass_result"]]

    train_weeks = range(1, settings.NUM_TRAIN_WEEKS + 1)
    test_weeks = range(settings.NUM_TRAIN_WEEKS + 1, settings.NUM_WEEKS + 1)

    train_dfs = list(process_weekly_data_generator(train_weeks, plays_df, reader))
    train_df = pd.concat(train_dfs, ignore_index=True)
    writer.write(train_df, settings.TRAIN_DATA_FILE)

    test_dfs = list(process_weekly_data_generator(test_weeks, plays_df, reader))
    test_df = pd.concat(test_dfs, ignore_index=True)
    writer.write(test_df, settings.TEST_DATA_FILE)


if __name__ == "__main__":
    create_datasets()
