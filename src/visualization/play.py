"""Module for visualizing plays."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import PillowWriter
from matplotlib.artist import Artist
from matplotlib.patches import Ellipse

from ..core.settings import get_settings
from ..core.teams import TEAMS, Team
from .field import FootballField


class Play:
    """Play class for visualizing a specific play in a game."""

    def __init__(
        self,
        game_id: int,
        play_id: int,
        save: bool,
        plays_data: pd.DataFrame,
        tracking_data: pd.DataFrame,
    ) -> None:
        """Initialize the Play class.

        Args:
            game_id (int): Game ID.
            play_id (int): Play ID.
            save (bool): Whether to save the animation as a GIF.
            plays_data (pd.DataFrame): DataFrame containing plays data.
            tracking_data (pd.DataFrame): DataFrame containing tracking data.
        """
        self._settings = get_settings()
        self.game_id = game_id
        self.play_id = play_id
        self.home_team: Team | None = None
        self.visitor_team: Team | None = None
        self.field: FootballField | None = None
        self.save = save
        self.plays_data = plays_data
        self.tracking_data = tracking_data

    def _init_field(self) -> None:
        """Initialize the football field.

        Raises:
            ValueError: If the home team is not set.
        """
        if self.home_team is None:
            raise ValueError("Home team is not set.")
        self.field = FootballField(home_team=self.home_team)

    def _set_teams(self) -> None:
        """Set the home and visitor teams based on the plays data."""
        self.home_team = TEAMS[self.plays_data["home_team_abbr"].iloc[0]]
        self.visitor_team = TEAMS[self.plays_data["visitor_team_abbr"].iloc[0]]

    def animate(self) -> None:
        """Animate the play.

        Raises:
            ValueError: If the game_id or play_id is not found in the plays data.
        """
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

        tracking_data = self.tracking_data[
            (self.tracking_data["game_id"] == self.game_id)
            & (self.tracking_data["play_id"] == self.play_id)
        ]
        plays_data = self.plays_data[
            (self.plays_data["game_id"] == self.game_id)
            & (self.plays_data["play_id"] == self.play_id)
        ]

        self._set_teams()
        self._init_field()

        if self.field is None:
            raise ValueError("Field is not initialized.")
        fig, self.ax = self.field.create_field()

        if tracking_data["play_direction"].iloc[0] == "left":
            line_of_scrimmage = (
                self._settings.FIELD_LENGTH
                - tracking_data["absolute_yardline_number"].iloc[0]
            )
        else:
            line_of_scrimmage = tracking_data["absolute_yardline_number"].iloc[0]

        first_down = line_of_scrimmage + plays_data["yards_to_go"].iloc[0]
        # down = plays_data["down"].iloc[0]
        # play_description = plays_data["play_description"].iloc[0]

        # Line of scrimmage
        self.ax.plot(
            (line_of_scrimmage + 10, line_of_scrimmage + 10),
            (0, self._settings.FIELD_WIDTH),
            color=self._settings.LINE_OF_SCRIMMAGE_COLOR,
            linewidth=1.2,
            alpha=0.8,
        )

        # First down line (if not goal to go)
        if first_down < 110:
            self.ax.plot(
                (first_down + 10, first_down + 10),
                (0, self._settings.FIELD_WIDTH),
                color=self._settings.FIRST_DOWN_LINE_COLOR,
                linewidth=1.2,
                alpha=0.8,
            )

        self.player_scatter: list[Artist] = []
        self.player_texts: list[Artist] = []
        self.ball_ellipse: Ellipse | None = None

        ani = animation.FuncAnimation(
            fig,
            self._update_frame,
            frames=sorted(tracking_data["frame_id"].unique()),
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

    def _update_frame(self, frame: int) -> list[Artist]:
        tracking_data_for_frame = self.tracking_data[
            (self.tracking_data["game_id"] == self.game_id)
            & (self.tracking_data["play_id"] == self.play_id)
            & (self.tracking_data["frame_id"] == frame)
        ]
        player_data_for_frame = tracking_data_for_frame[
            ~tracking_data_for_frame["nfl_id"].isna()
        ]
        ball_data_for_frame = tracking_data_for_frame[
            tracking_data_for_frame["nfl_id"].isna()
        ]

        for scatter in self.player_scatter:
            scatter.remove()
        self.player_scatter.clear()

        for text in self.player_texts:
            text.remove()
        self.player_texts.clear()

        if self.ball_ellipse:
            self.ball_ellipse.remove()
            self.ball_ellipse = None

        # Update the ball position
        if not ball_data_for_frame.empty:
            ball_x = ball_data_for_frame.x.values[0] + 0.31
            ball_y = ball_data_for_frame.y.values[0]

            # Create and store the ball patch so it can be removed on the
            # next frame. The FuncAnimation callback must return the
            # artists that were updated.
            self.ball_ellipse = Ellipse(
                xy=(ball_x, ball_y),
                width=0.8,
                height=0.5,
                facecolor=self._settings.BALL_COLOR,
                edgecolor="black",
                linewidth=1,
                zorder=4,
            )
            self.ax.add_patch(self.ball_ellipse)

        # Update player positions
        for _, player in player_data_for_frame.iterrows():
            x = player["x"]
            y = player["y"]
            if self.home_team is None or self.visitor_team is None:
                raise ValueError("Teams are not set.")
            team_color = (
                (self.home_team.primary_color, self.home_team.secondary_color)
                if player["player_side"] == "Defense"  # This is wrong, change later
                else (
                    self.visitor_team.primary_color,
                    self.visitor_team.secondary_color,
                )
            )
            # jersey_number = player["jersey_number"] # Not available this year
            jersey_number = player["nfl_id"] % 100  # Temporary workaround

            # Plot player position
            scatter = self.ax.scatter(
                x,
                y,
                s=110,
                facecolor=team_color[0],
                edgecolor=team_color[1],
                linewidth=1,
                zorder=2,
            )

            text = self.ax.text(
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

            # Store references to the player markers and text
            self.player_scatter.append(scatter)
            self.player_texts.append(text)

        return (
            self.player_scatter
            + self.player_texts
            + ([self.ball_ellipse] if self.ball_ellipse else [])
        )


if __name__ == "__main__":
    plays_data = pd.read_csv("data/raw/supplementary_data.csv", low_memory=False)
    tracking_data = pd.read_csv("data/raw/input_2023_w01.csv", low_memory=False)
    ball_data = tracking_data[tracking_data["nfl_id"].isna()]

    play = Play(
        game_id=2023090700,
        play_id=877,
        save=False,
        plays_data=plays_data,
        tracking_data=tracking_data,
    )
    play.animate()

    # print("Is there any ball position available?")
    # print("Yes" if tracking_data["nfl_id"].isna().any() else "No")

    tracking_data = pd.read_csv("data/raw/output_2023_w01.csv", low_memory=False)
    print("Is there any ball position available?")
    print("Yes" if tracking_data["nfl_id"].isna().any() else "No")
