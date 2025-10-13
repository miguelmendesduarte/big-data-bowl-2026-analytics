"""Module for cleaning raw data."""

from .players import create_players_dataset
from .plays import create_plays_dataset
from .tracking import clean_tracking_data


def clean_data() -> None:
    """Clean all raw datasets and store in the cleaned data directory."""
    create_players_dataset()
    create_plays_dataset()
    clean_tracking_data()


if __name__ == "__main__":
    clean_data()
