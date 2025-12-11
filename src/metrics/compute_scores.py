"""Module for computing deception and recovery metrics."""

import pandas as pd

from ..core.settings import get_settings
from ..io.datasets import CSVReader, CSVWriter
from .deception import compute_deception_score
from .recovery import compute_recovery_score

settings = get_settings()


def compute_scores() -> None:
    """Compute deception and recovery metrics."""
    reader = CSVReader()

    df = reader.read(settings.INFERENCE_RESULTS_FILE)

    df_deception = compute_deception_score(df)
    df_recovery = compute_recovery_score(df)

    df_combined = pd.merge(
        df_deception[
            [
                "game_id",
                "play_id",
                "frame_id",
                "defender_id",
                "receiver_id",
                "deception_score",
            ]
        ],
        df_recovery[["game_id", "play_id", "frame_id", "recovery_score"]],
        on=["game_id", "play_id", "frame_id"],
        how="left",
    )

    final_df = df_combined[
        [
            "game_id",
            "play_id",
            "defender_id",
            "receiver_id",
            "deception_score",
            "recovery_score",
        ]
    ]

    writer = CSVWriter()
    writer.write(final_df, settings.SCORES_FILE)


if __name__ == "__main__":
    compute_scores()
