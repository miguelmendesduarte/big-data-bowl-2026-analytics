"""Module for cleaning raw data."""

from .players import create_players_dataset
from .plays import filter_plays_with_tracking, process_plays_data
from .tracking import clean_tracking_data


def clean_data() -> None:
    """Clean all raw datasets and store in the cleaned data directory."""
    create_players_dataset()
    process_plays_data()
    clean_tracking_data()
    filter_plays_with_tracking()


if __name__ == "__main__":
    clean_data()
