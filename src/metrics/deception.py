"""Module for computing deception metric."""

import numpy as np
import pandas as pd
import ruptures as rpt  # type: ignore[import-untyped]


def get_last_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Get the last frame row for each play."""
    return df.groupby(["game_id", "play_id"]).tail(1)


def get_first_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Get the first frame row for each play."""
    return df.groupby(["game_id", "play_id"]).head(1)


def detect_change_point(df: pd.DataFrame, column: str) -> int:
    """Detect the change point based on mean values using ruptures.

    This function uses ruptures to detect the change point in the data.
    It then returns the index of the change point in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        column (str): Column name to analyze for change point detection.

    Returns:
        int: Index of the change point in the dataframe.
    """
    signal = df[column].values

    # Apply the change point detection algorithm (using Pelt method)
    model = "l2"  # Using L2 model for detecting mean change
    algo = rpt.Pelt(model=model).fit(signal)
    change_points: list[int] = algo.predict(pen=0.01)

    if len(change_points) <= 1:
        return len(signal)

    change_magnitudes = []

    # Iterate through the indices of the 'actual' change points (i.e.,
    # everything except the last element, which is the signal length).
    # i will go from 0 up to len(change_points) - 2.
    for i in range(len(change_points) - 1):
        # The change point index being measured is change_points[i]
        cp_index = change_points[i]

        # Segment BEFORE the change point (starts from previous change point or 0)
        start_prev_segment = change_points[i - 1] if i > 0 else 0
        first_segment = signal[start_prev_segment:cp_index]

        # This is safe because the loop stops one iteration early.
        second_segment = signal[cp_index : change_points[i + 1]]

        first_segment = pd.Series(first_segment).to_numpy()
        second_segment = pd.Series(second_segment).to_numpy()

        # Only calculate the means if the segments have data
        if (
            first_segment.size > 0 and second_segment.size > 0
        ):  # Use .size for NumPy array check
            magnitude = abs(second_segment.mean() - first_segment.mean())
            change_magnitudes.append(magnitude)
        else:
            change_magnitudes.append(0)

    # Pick the change point with the largest magnitude
    if change_magnitudes:
        # np.argmax is safer than .index(max()) for floats
        most_significant_change_idx = np.argmax(change_magnitudes)

        # The index (0, 1, 2, ...) maps directly to the change_points list element
        # that ISN'T the signal length.
        return change_points[
            most_significant_change_idx
        ]  # Return the actual index in the signal

    return len(signal)


def calculate_deception(df: pd.DataFrame) -> pd.Series:
    """Calculate deception metric based on the change point in the data.

    Args:
        df (pd.DataFrame): Dataframe containing the data.

    Returns:
        pd.Series: Deception metric for each play.
    """
    deception_scores = []

    for _, group in df.groupby(["game_id", "play_id"]):
        change_point = detect_change_point(group, "non_completion_probability")

        if change_point == len(
            group
        ):  # No change point detected, use final minus first
            # If no change point, calculate deception as difference between
            # last and first data points
            first_value = group.iloc[0]["non_completion_probability"]
            last_value = group.iloc[-1]["non_completion_probability"]
            deception_score = last_value - first_value
        else:
            first_segment = group.iloc[:change_point]
            second_segment = group.iloc[change_point:]

            mean_first = first_segment["non_completion_probability"].mean()
            mean_second = second_segment["non_completion_probability"].mean()

            deception_score = mean_second - mean_first

        deception_scores.append(deception_score)

    index = df.groupby(["game_id", "play_id"]).head(1).index
    return pd.Series(deception_scores, index=index, dtype=float)


def compute_deception_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deception score and add it to the dataframe.

    Args:
        df (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Dataframe with deception score added.
    """
    df_deception = get_last_frame(df)
    df_deception = df_deception.copy()

    deception = calculate_deception(df)

    df_deception["deception_score"] = deception.values

    return df_deception
