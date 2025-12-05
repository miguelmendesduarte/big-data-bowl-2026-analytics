"""Module for creating features for the training and testing datasets."""

import numpy as np
import pandas as pd

from ...core.settings import get_settings

settings = get_settings()


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees.

    Args:
        a (float): First angle in degrees.
        b (float): Second angle in degrees.

    Returns:
        float: Angle difference in degrees.
    """
    diff = (a - b) % 360

    return min(diff, 360 - diff)


def _unit_vec_from_angle(deg: float) -> np.ndarray:
    """Unit vector from angle in degrees.

    Args:
        deg (float): Angle in degrees.

    Returns:
        np.ndarray: Unit vector.
    """
    rad = np.deg2rad(deg)

    return np.array([np.cos(rad), np.sin(rad)])


def _boundary_distance(
    x: float,
    y: float,
    field_length: float = settings.FIELD_LENGTH,
    field_width: float = settings.FIELD_WIDTH,
) -> float:
    """Distance to the nearest endzone or sideline.

    Args:
        x (float): X coordinate.
        y (float): Y coordinate.
        field_length (int): Field length in yards.
        field_width (float): Field width in yards.

    Returns:
        float: Distance to the nearest endzone or sideline.
    """
    dist_sideline = min(y, field_width - y)
    dist_endline = min(x, field_length - x)

    return min(dist_sideline, dist_endline)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add final features to the dataset.

    Args:
        df (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data with final engineered features.
    """
    feats = []

    grouped = df.groupby(["game_id", "play_id"], sort=False)

    for (game_id, play_id), g in grouped:
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
                "target": g.is_non_completion.iloc[0],
            }
        )

    return pd.DataFrame(feats)
