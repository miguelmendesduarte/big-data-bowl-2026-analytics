"""Module for visualizing plays."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import typer
from loguru import logger
from matplotlib.animation import PillowWriter
from matplotlib.artist import Artist

from ..core.settings import get_settings
from ..core.teams import TEAMS, Team
from ..io.datasets import CSVReader
from .field import FootballField


class Play:
    """Play class for visualizing a specific play in a game."""

    def __init__(self, game_id: int, play_id: int, save: bool) -> None:
        """Initialize the Play class.

        Args:
            game_id (int): Game ID.
            play_id (int): Play ID.
            save (bool): Whether to save the animation as a GIF.
        """
        self._settings = get_settings()
        self.game_id = game_id
        self.play_id = play_id
        self.home_team: Team | None = None
        self.save = save

    def _read_data(self) -> None:
        """Read the necessary data for the play.

        Raises:
            ValueError: If the game ID or play ID is not found in the plays data.
        """
        reader = CSVReader()

        self.plays_data = reader.read(self._settings.CLEANED_PLAYS_FILE)
        self.plays_data = self.plays_data[
            (self.plays_data["game_id"] == self.game_id)
            & (self.plays_data["play_id"] == self.play_id)
        ]

        if self.game_id not in self.plays_data["game_id"].values:
            raise ValueError(f"Game ID {self.game_id} not found in plays data.")
        if (
            self.play_id
            not in self.plays_data[self.plays_data["game_id"] == self.game_id][
                "play_id"
            ].values
        ):
            raise ValueError(
                f"Play ID {self.play_id} not found for Game ID {self.game_id}."
            )

        self.tracking_data_before = reader.read(
            self._settings.get_tracking_data_path(
                week=self.plays_data["week"].iloc[0],
                data_stage="cleaned",
                throw_stage="before",
            )
        )
        self.tracking_data_before = self.tracking_data_before[
            (self.tracking_data_before["game_id"] == self.game_id)
            & (self.tracking_data_before["play_id"] == self.play_id)
        ]

        self.tracking_data_after = reader.read(
            self._settings.get_tracking_data_path(
                week=self.plays_data["week"].iloc[0],
                data_stage="cleaned",
                throw_stage="after",
            )
        )
        self.tracking_data_after = self.tracking_data_after[
            (self.tracking_data_after["game_id"] == self.game_id)
            & (self.tracking_data_after["play_id"] == self.play_id)
        ]

        max_frame_before = self.tracking_data_before["frame_id"].max()
        self.tracking_data_after = self.tracking_data_after.copy()
        self.tracking_data_after["frame_id"] += max_frame_before

        self.tracking_data = pd.concat(
            [self.tracking_data_before, self.tracking_data_after], ignore_index=True
        ).sort_values(by="frame_id")

    def _init_field(self) -> None:
        """Initialize the football field.

        Raises:
            ValueError: If the home team is not set.
        """
        if self.home_team is None:
            raise ValueError("Home team is not set.")
        self.field = FootballField(home_team=self.home_team)

    def _set_home_team(self) -> None:
        """Set the home team based on the plays data."""
        self.home_team = TEAMS[self.plays_data["home_team_abbr"].iloc[0]]

    def animate(self) -> None:
        """Animate the play."""
        self._read_data()
        self._set_home_team()
        self._init_field()

        if self.field is None:
            raise ValueError("Field is not initialized.")

        fig, ax = self.field.create_field()

        if self.plays_data["play_direction"].iloc[0] == "left":
            line_of_scrimmage = (
                self._settings.FIELD_LENGTH
                - self.plays_data["absolute_yardline_number"].iloc[0]
                - 10
            )
        else:
            line_of_scrimmage = self.plays_data["absolute_yardline_number"].iloc[0] - 10

        first_down = line_of_scrimmage + self.plays_data["yards_to_go"].iloc[0]
        down = self.plays_data["down"].iloc[0]
        play_description = self.plays_data["play_description"].iloc[0]

        logger.info(
            f"Game ID: {self.game_id}, Play ID: {self.play_id}, Down: {down}, "
            f"Line of Scrimmage: {line_of_scrimmage + 10}, First Down: {first_down}, "
            f"Play Description: {play_description}"
        )

        # Line of scrimmage
        ax.plot(
            (line_of_scrimmage + 10, line_of_scrimmage + 10),
            (0, self._settings.FIELD_WIDTH),
            color=self._settings.LINE_OF_SCRIMMAGE_COLOR,
            linewidth=1.2,
            alpha=0.8,
        )

        # First down line (if not goal to go)
        if first_down < 110:
            ax.plot(
                (first_down + 10, first_down + 10),
                (0, self._settings.FIELD_WIDTH),
                color=self._settings.FIRST_DOWN_LINE_COLOR,
                linewidth=1.2,
                alpha=0.8,
            )

        # Ball landing position
        ball_x = self.plays_data["ball_land_x"].iloc[0]
        ball_y = self.plays_data["ball_land_y"].iloc[0]
        ax.text(
            ball_x,
            ball_y,
            "X",
            fontsize=12,
            fontweight="bold",
            color="red",
            ha="center",
            va="center",
            zorder=4,
        )

        player_scatter: list[Artist] = []
        player_texts: list[Artist] = []
        ball_ellipse = None

        def update(frame: int) -> list[Artist]:
            """Update the animation.

            Args:
                frame (int): Frame number.

            Returns:
                list[Artist]: List of artists updated.
            """
            nonlocal player_scatter, ball_ellipse, player_texts
            for scatter in player_scatter:
                scatter.remove()
            player_scatter = []

            for text in player_texts:
                text.remove()
            player_texts = []

            if ball_ellipse is not None:
                ball_ellipse.remove()

            current_frame_data = self.tracking_data[
                self.tracking_data.frame_id == frame
            ]

            # Players
            for _, row in current_frame_data.iterrows():
                x = row.x
                y = row.y
                team = row.team
                team_color = (
                    (TEAMS[team].primary_color, TEAMS[team].secondary_color)
                    if team in TEAMS
                    else ("#FFFFFF", "#000000")
                )
                jersey_number = row["nfl_id"] % 100  # Temporary workaround

                scatter = ax.scatter(
                    x,
                    y,
                    s=110,
                    facecolor=team_color[0],
                    edgecolor=team_color[1],
                    linewidth=1,
                    zorder=2,
                )
                text = ax.text(
                    x,
                    y,
                    f"{int(jersey_number)}",
                    color=team_color[1],
                    ha="center",
                    va="center",
                    fontsize=5,
                    fontweight="bold",
                    zorder=3,
                )

                player_scatter.append(scatter)
                player_texts.append(text)

            return (
                player_scatter + player_texts + ([ball_ellipse] if ball_ellipse else [])
            )

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=sorted(self.tracking_data["frame_id"].unique()),
            interval=100,
            repeat=True,
            repeat_delay=1000,
        )

        if self.save:
            ani.save(
                f"{self.game_id}_{self.play_id}.gif",
                writer=PillowWriter(
                    fps=self._settings.FRAME_RATE, bitrate=self._settings.BITRATE
                ),
            )
        else:
            plt.show()


app = typer.Typer(help="Visualize a specific play from tracking data.")


@app.command()
def visualize_play(
    game_id: int = typer.Option(..., help="Game ID of the play to visualize."),
    play_id: int = typer.Option(..., help="Play ID of the play to visualize."),
    save: bool = typer.Option(
        False, help="Whether to save the animation as a GIF file."
    ),
) -> None:
    """Visualize a specific play using tracking data.

    This function loads tracking and play data for the given game and play ID,
    and either displays or saves an animated visualization of the play.

    Args:
        game_id (int): Game ID of the play to visualize.
        play_id (int): Play ID of the play to visualize.
        save (bool): Whether to save the animation as a GIF file.
            Default is False.
    """
    play = Play(game_id=game_id, play_id=play_id, save=save)
    play.animate()


if __name__ == "__main__":
    app()
