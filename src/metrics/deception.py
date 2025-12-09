"""Module for computing deception metric."""

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

    # Select the largest change (most significant jump)
    change_magnitudes = []
    for i in range(1, len(change_points)):
        # Ensure the segments are non-empty
        first_segment = signal[change_points[i - 1] : change_points[i]]
        second_segment = signal[
            change_points[i] : change_points[i + 1]
            if i + 1 < len(change_points)
            else len(signal)
        ]

        first_segment = pd.Series(first_segment).to_numpy()
        second_segment = pd.Series(second_segment).to_numpy()

        # Only calculate the means if the segments have data
        if len(first_segment) > 0 and len(second_segment) > 0:
            mean_first = first_segment.mean()
            mean_second = second_segment.mean()
            magnitude = abs(mean_second - mean_first)
            change_magnitudes.append(magnitude)
        else:
            # If segments are empty, handle as needed
            change_magnitudes.append(0)

    # Pick the change point with the largest magnitude
    if change_magnitudes:
        most_significant_change_idx = change_magnitudes.index(max(change_magnitudes))
        return change_points[most_significant_change_idx + 1]

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
