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


# if __name__ == "__main__":
#     compute_scores()

#     from ..io.datasets import CSVReader

#     settings = get_settings()
#     reader = CSVReader()

#     df = reader.read(settings.SCORES_FILE)
#     players = reader.read(settings.PLAYERS_FILE)

#     # Valid Players
#     valid_defenders = df.groupby("defender_id").filter(lambda x: len(x) >= 10)
#     valid_receivers = df.groupby("receiver_id").filter(lambda x: len(x) >= 10)

#     # Wide Receivers
#     mean_deception_by_receiver = valid_receivers.groupby("receiver_id")[
#         "deception_score"
#     ].mean()
#     sorted_mean_deception_by_receiver = mean_deception_by_receiver.sort_values(
#         ascending=False
#     )
#     receivers_deception_df = sorted_mean_deception_by_receiver.reset_index().merge(
#         players[["nfl_id", "player_name", "player_position"]],
#         left_on="receiver_id",
#         right_on="nfl_id",
#         how="left",
#     )
#     receivers_deception_df.drop(columns=["nfl_id"], inplace=True)
#     import matplotlib.pyplot as plt

#     # Step 1: Calculate mean deception and recovery for each defender
#     mean_deception_by_defender = valid_defenders.groupby("defender_id")[
#         "deception_score"
#     ].mean()
#     mean_recovery_by_defender = valid_defenders.groupby("defender_id")[
#         "recovery_score"
#     ].mean()

#     defenders_df = pd.DataFrame(
#         {
#             "defender_id": mean_deception_by_defender.index,
#             "mean_deception": mean_deception_by_defender.values,
#             "mean_recovery": mean_recovery_by_defender.values,
#         }
#     )

#     # Merge with player details for names
#     defenders_df = defenders_df.merge(
#         players[["nfl_id", "player_name", "player_position"]],
#         left_on="defender_id",
#         right_on="nfl_id",
#         how="left",
#     )
#     defenders_df.drop(columns=["nfl_id"], inplace=True)

#     # Step 3: Plot the data
#     fig, ax = plt.subplots(figsize=(8, 8))

#     # Scatter plot for Mean Deception vs Mean Recovery
#     ax.scatter(
#         defenders_df["mean_deception"],
#         defenders_df["mean_recovery"],
#         c="blue",
#         edgecolor="black",
#     )

#     # Annotate the players with their names
#     for i, player in enumerate(defenders_df["player_name"]):
#         ax.annotate(
#             player,
#             (
#                 defenders_df["mean_deception"].iloc[i],
#                 defenders_df["mean_recovery"].iloc[i],
#             ),
#             textcoords="offset points",
#             xytext=(0, 5),
#             ha="center",
#         )

#     # Set labels and title
#     ax.set_xlabel("Mean Deception")
#     ax.set_ylabel("Mean Recovery")
#     ax.set_title("Defenders: Mean Deception vs Mean Recovery")

#     # Draw horizontal and vertical lines at 0 (quadrants)
#     ax.axhline(0, color="black", linewidth=1)
#     ax.axvline(0, color="black", linewidth=1)

#     # Set axis limits
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)

#     # Show grid
#     ax.grid(True)

#     # Show the plot
#     plt.show()

#     amazing_play_score = (
#         df["deception_score"].abs() + df["recovery_score"]
#     )
#     df["amazing_play_score"] = amazing_play_score
#     print("Top 10 Amazing Plays:")
#     top_amazing_plays = df.sort_values(
#         by="amazing_play_score", ascending=False
#     ).head(10)
#     print(
#         top_amazing_plays[
#             [
#                 "game_id",
#                 "play_id",
#                 "defender_id",
#                 "receiver_id",
#                 "deception_score",
#                 "recovery_score",
#                 "amazing_play_score",
#             ]
#         ]
#     )
