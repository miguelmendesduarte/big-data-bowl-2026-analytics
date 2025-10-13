"""Main entry point for the application."""

from .core.logs import configure_logging
from .core.settings import get_settings
from .io.datasets import CSVReader


def main() -> None:
    """Main function to run the application."""
    settings = get_settings()
    configure_logging(settings)

    plays_data = CSVReader().read(settings.CLEANED_PLAYS_FILE)

    print(plays_data.route_of_targeted_receiver.value_counts())

    PASS_LENGTH_MIN_THRESHOLD = 4
    PASS_LENGTH_MAX_THRESHOLD = 200
    ROUTE = "OUT"

    plays_data = plays_data[
        (plays_data["route_of_targeted_receiver"] == ROUTE)
        & (plays_data["pass_length"] > PASS_LENGTH_MIN_THRESHOLD)
        & (plays_data["pass_length"] < PASS_LENGTH_MAX_THRESHOLD)
        & (plays_data["dropback_type"] == "TRADITIONAL")
        & (plays_data["pass_result"] == "C")
    ]
    print(
        f"Found {len(plays_data)} plays with {ROUTE} route "
        f"and pass length > {PASS_LENGTH_MIN_THRESHOLD} yards "
        f"and < {PASS_LENGTH_MAX_THRESHOLD} yards."
    )
    print(plays_data[["game_id", "play_id", "pass_length"]])


if __name__ == "__main__":
    main()
