"""Module for creating a dataset of players."""

import pandas as pd

from ...core.settings import get_settings
from ...io.datasets import CSVReader, CSVWriter

settings = get_settings()

PLAYER_COLS = [
    "nfl_id",
    "player_name",
    "player_height",
    "player_weight",
    "player_birth_date",
    "player_position",
]


def create_players_dataset() -> None:
    """Create a dataset of players.

    Raises:
        ValueError: If required columns are missing in the data.
    """
    all_players = []
    for week in range(1, settings.NUM_WEEKS + 1):
        input_path = settings.get_tracking_data_path(
            week=week,
            data_stage="raw",
            throw_stage="before",
        )

        df = CSVReader().read(input_path)

        if not all(col in df.columns for col in PLAYER_COLS):
            raise ValueError(
                f"Missing required columns in data for week {week}: {PLAYER_COLS}"
            )

        players = df[PLAYER_COLS]
        players = players.drop_duplicates(subset=["nfl_id"])

        all_players.append(players)

    players_df = pd.concat(all_players, ignore_index=True)
    players_df = players_df.drop_duplicates(subset=["nfl_id"], ignore_index=True)
    players_df = players_df.sort_values(by="nfl_id", ascending=True)
    players_df.reset_index(drop=True, inplace=True)

    output_path = settings.CLEANED_DATA_DIR / "players.csv"
    CSVWriter().write(players_df, output_path)


if __name__ == "__main__":
    create_players_dataset()
