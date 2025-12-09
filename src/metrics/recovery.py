"""Module for computing recovery metric."""

from enum import StrEnum

import pandas as pd

from ..core.settings import get_settings


class PassResult(StrEnum):
    """Enum for pass results."""

    COMPLETE = "C"
    INCOMPLETE = "I"
    INTERCEPTION = "IN"


settings = get_settings()


def get_last_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Get the last frame row for each play.

    Args:
        df (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Dataframe with only the last frames for each play.
    """
    return df.groupby(["game_id", "play_id"]).tail(1)


def calculate_recovery(df: pd.DataFrame) -> pd.Series:
    """Calculate recovery metric for each play.

    The recovery metric is defined as follows:
        - Complete pass: -non_completion_probability
        - Incomplete pass: 1 - non_completion_probability
        - Interception: 2 * (1 - non_completion_probability)

    Args:
        df (pd.DataFrame): Dataframe with last frames for each play.

    Returns:
        pd.Series: Recovery metric for each play.
    """
    recovery = pd.Series(0.0, index=df.index)

    recovery[df["pass_result"] == PassResult.COMPLETE] = -df[
        "non_completion_probability"
    ]
    recovery[df["pass_result"] == PassResult.INCOMPLETE] = (
        1 - df["non_completion_probability"]
    )
    recovery[df["pass_result"] == PassResult.INTERCEPTION] = 1.2 * (
        1 - df["non_completion_probability"]
    )
    # recovery[df["pass_result"] == PassResult.INTERCEPTION] = (
    #     1 - df["non_completion_probability"]
    # )

    return recovery


def compute_recovery_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute recovery score and add it to the dataframe.

    Args:
        df (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Dataframe with recovery score added.
    """
    df_recovery = get_last_frame(df)
    df_recovery = df_recovery.copy()

    df_recovery.loc[:, "recovery_score"] = calculate_recovery(df_recovery)

    return df_recovery
