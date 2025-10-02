"""Main entry point for the application."""

from loguru import logger

from .config.logs import configure_logging
from .config.settings import Settings


def main() -> None:
    """Main function to run the application."""
    settings = Settings()
    configure_logging(settings)

    logger.info(f"Application started with settings: {settings}")


if __name__ == "__main__":
    main()
