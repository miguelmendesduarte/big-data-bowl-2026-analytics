"""Module for creating a dataset for inference."""

from typing import Generator, Literal

import numpy as np
import pandas as pd

from ...core.settings import get_settings
from ...io.datasets import CSVReader, CSVWriter
from ..training.features import _angle_diff, _boundary_distance, _unit_vec_from_angle

settings = get_settings()

DataStage = Literal["raw", "cleaned", "processed"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add final features to the dataset.

    Args:
        df (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data with final engineered features.
    """
    feats = []

    grouped = df.groupby(["game_id", "play_id", "frame_id"], sort=False)

    for (game_id, play_id, frame_id), g in grouped:
        try:
            qb = g.loc[g.player_role == "Passer"].iloc[0]
            rec = g.loc[g.player_role == "Targeted Receiver"].iloc[0]
            deff = g.loc[g.player_side == "Defense"].iloc[0]
        except IndexError:
            continue

        # -----------------------------
        # Vector geometry
        # -----------------------------
        qb_pos = np.array([qb.x, qb.y])
        rec_pos = np.array([rec.x, rec.y])
        def_pos = np.array([deff.x, deff.y])

        vec_qb_to_rec = rec_pos - qb_pos
        vec_def_to_rec = rec_pos - def_pos
        vec_def_to_qb = qb_pos - def_pos

        dist_qb_rec = np.linalg.norm(vec_qb_to_rec)
        dist_def_rec = np.linalg.norm(vec_def_to_rec)
        dist_def_qb = np.linalg.norm(vec_def_to_qb)

        # -----------------------------
        # Base features
        # -----------------------------
        air_yards = rec.x - qb.x

        # Closing speed — using movement direction (dir)
        def_vel = deff.s * _unit_vec_from_angle(deff.dir)
        closing_speed = -np.dot(def_vel, vec_def_to_rec) / np.clip(
            dist_def_rec, 0.01, None
        )

        # Orientation error — using facing orientation (o)
        angle_to_rec = np.rad2deg(np.arctan2(vec_def_to_rec[1], vec_def_to_rec[0]))
        orient_error = _angle_diff(deff.o, angle_to_rec)

        # -----------------------------
        # Boundary distance (includes end zone proximity)
        # -----------------------------
        boundary_dist = _boundary_distance(rec.x, rec.y)

        feats.append(
            {
                "game_id": game_id,
                "play_id": play_id,
                "frame_id": frame_id,
                "receiver_id": rec.nfl_id,
                "defender_id": deff.nfl_id,
                # Distances & air yards
                "air_yards": round(air_yards, 2),
                "separation": round(dist_def_rec, 2),
                "qb_to_rec_dist": round(dist_qb_rec, 2),
                "pressure_dist": round(dist_def_qb, 2),
                # Speeds
                "rec_speed": round(rec.s, 2),
                "def_speed": round(deff.s, 2),
                "qb_speed": round(qb.s, 2),
                # Orientation & angles
                "closing_speed": round(closing_speed, 2),
                "def_orientation_error": round(orient_error, 2),
                "def_back_to_rec": int(orient_error > 120),
                # Receiver geometry
                "rec_boundary_dist": round(boundary_dist, 2),
                "rec_running_away": int(rec.s > 6),
                # Engineered interaction features
                "separation_sq": round(dist_def_rec**2, 2),
                "sep_per_air_yard": round(dist_def_rec / max(air_yards, 1), 2),
                "closing_per_yard": round(closing_speed / max(air_yards, 1), 2),
                # Target / label
                "target": g.pass_result.iloc[0],
            }
        )

    return pd.DataFrame(feats)


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
    processed_df = tracking_df.merge(plays_df, on=["game_id", "play_id"], how="left")

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


def create_inference_dataset() -> None:
    """Create the training and testing datasets."""
    reader = CSVReader()
    writer = CSVWriter()

    plays_df = reader.read(settings.CLEANED_PLAYS_FILE)
    plays_df = plays_df[["game_id", "play_id", "pass_result"]]

    test_weeks = range(settings.NUM_TRAIN_WEEKS + 1, settings.NUM_WEEKS + 1)

    test_dfs = list(process_weekly_data_generator(test_weeks, plays_df, reader))
    test_df = pd.concat(test_dfs, ignore_index=True)
    writer.write(test_df, settings.INFERENCE_DATA_FILE)


if __name__ == "__main__":
    create_inference_dataset()
