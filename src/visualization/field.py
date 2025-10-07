"""Utilities to render a stylized football field using Matplotlib."""

import matplotlib.image as mpl_image
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.settings import get_settings
from ..core.teams import Team


class FootballField:
    """Football field rendering utility."""

    def __init__(self, home_team: Team) -> None:
        """Initialize the FootballField.

        Args:
            home_team (Team): Team used to determine logo path and
                primary/secondary colours for endzones and text effects.
        """
        self._settings = get_settings()
        self.field_length = self._settings.FIELD_LENGTH
        self.field_width = self._settings.FIELD_WIDTH
        self.field_color = self._settings.FIELD_COLOR
        self.line_color = self._settings.LINE_COLOR
        self.figure_size = self._settings.FIGURE_SIZE
        self.home_team = home_team

    def create_field(self) -> tuple[Figure, Axes]:
        """Construct the figure and draw all field elements.

        The method creates a Matplotlib figure/axes pair and draws the
        field background, yard lines, yard numbers, endzones, and an
        optional team logo. The returned ``(fig, ax)`` can be used by
        callers to overlay event/trajectory plots or to save the
        resulting image.

        Returns:
            tuple[Figure, Axes]: Created Matplotlib figure and axes.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figure_size)

        self._draw_field()
        self._draw_field_lines()
        self._add_noise()
        self._draw_team_logo()
        self._draw_endzones()
        self._draw_yard_numbers()

        self._configure_axes()
        plt.tight_layout()

        return self.fig, self.ax

    def _add_noise(self) -> None:
        """Add a subtle noise texture to the field.

        The noise layer is a low-alpha greyscale image used to break up
        the flat field colour and produce a slightly more realistic
        background when the field is displayed or exported.
        """
        noise = np.random.rand(200, 200)
        self.ax.imshow(
            noise,
            extent=(0, self.field_length, 0, self.field_width),
            cmap="Greys",
            alpha=0.05,
            zorder=1,
        )

    def _configure_axes(self) -> None:
        """Configure axis limits and hide axis decorations.

        Sets the x/y limits to match field dimensions and disables the
        axis frame and ticks so only the rendered field is visible.
        """
        self.ax.set_xlim(0, self.field_length)
        self.ax.set_ylim(0, self.field_width)
        self.ax.axis("off")

    def _draw_field(self) -> None:
        """Draw the green field rectangle background.

        Adds a filled rectangle to the axes matching the configured
        field dimensions and field colour.
        """
        self.ax.add_patch(
            patches.Rectangle(
                (0, 0), self.field_length, self.field_width, color=self.field_color
            )
        )

    def _draw_field_lines(self) -> None:
        """Draw yard lines and hash marks on the field.

        Yard lines are drawn every yard with thicker marks at 5 and 10
        yard increments. Hash marks and short line segments are drawn at
        several offsets to approximate the appearance of a regulation
        football field.
        """
        for line in range(10, 111):
            if line % 5 == 0:
                linewidth = 1.5 if line in {10, 110} else 1.0
                self.ax.plot(
                    [line, line],
                    [0, self.field_width],
                    color=self.line_color,
                    linewidth=linewidth,
                    alpha=0.8,
                )
            elif line in {12, 108}:
                self.ax.plot(
                    [line, line],
                    [self.field_width / 2 - 0.25, self.field_width / 2 + 0.25],
                    color=self.line_color,
                    linewidth=0.5,
                )

            offsets = [
                0,
                self.field_width * 4 / 10,
                self.field_width * 6 / 10,
                self.field_width - 0.5,
            ]
            for offset in offsets:
                self.ax.plot(
                    [line, line],
                    [offset, offset + 0.5],
                    color=self.line_color,
                    linewidth=0.5,
                )

    def _draw_team_logo(self) -> None:
        """Overlay the home team's logo at midfield.

        The logo is read from the path provided by the :class:`Team`
        instance and drawn semi-transparently at the centre of the
        field. If the logo file is missing or unreadable, Matplotlib
        will raise an error when attempting to read it.
        """
        logo = mpl_image.imread(self.home_team.get_logo_file_path())
        self.ax.imshow(
            logo,
            aspect="auto",
            zorder=1,
            alpha=0.3,
            extent=(
                self.field_length / 2 - 7,
                self.field_length / 2 + 7,
                self.field_width / 2 - 7,
                self.field_width / 2 + 7,
            ),
        )

    def _draw_endzones(self) -> None:
        """Draw stylized endzone text at each end of the field.

        Displays the team location and team name vertically in the
        endzones using the team's primary and secondary colours and a
        stroke effect to improve contrast against the background.
        """
        self.ax.text(
            5,
            self.field_width / 2,
            self.home_team.location,
            weight="bold",
            color=self.home_team.primary_color,
            fontsize=42,
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            path_effects=[
                path_effects.withStroke(
                    linewidth=2, foreground=self.home_team.secondary_color
                )
            ],
            zorder=1,
        ).set_alpha(0.5)

        self.ax.text(
            115,
            self.field_width / 2,
            self.home_team.name,
            weight="bold",
            color=self.home_team.primary_color,
            fontsize=42,
            rotation=-90,
            verticalalignment="center",
            horizontalalignment="center",
            path_effects=[
                path_effects.withStroke(
                    linewidth=2, foreground=self.home_team.secondary_color
                )
            ],
            zorder=1,
        ).set_alpha(0.5)

    def _draw_yard_numbers(self) -> None:
        """Draw yard numbers and small directional triangles.

        Yard numbers are drawn every 10 yards across the field and
        rotated on the top half so they read correctly from that
        vantage. Small triangular markers (arrows) are drawn next to the
        numbers to visually indicate direction toward the endzones.
        """
        for yard in range(20, 101, 5):
            if yard % 10 == 0:
                number = f"{yard - 10 if yard <= 50 else 120 - yard - 10}"

                # Bottom number
                self.ax.text(
                    yard,
                    self.field_width / 6,
                    number,
                    fontsize=16,
                    color=self.line_color,
                    horizontalalignment="center",
                    zorder=2,
                ).set_alpha(0.8)

                # Top number (rotated)
                self.ax.text(
                    yard,
                    5 * self.field_width / 6,
                    number,
                    fontsize=16,
                    color=self.line_color,
                    horizontalalignment="center",
                    rotation=180,
                    zorder=2,
                ).set_alpha(0.8)

                # Triangles/arrows
                if yard < 60:
                    self._draw_left_triangle(yard, -2.5, self.field_width / 6 + 1.25)
                    self._draw_left_triangle(
                        yard, -2.5, 5 * self.field_width / 6 - 0.25
                    )
                if yard > 60:
                    self._draw_right_triangle(yard, 2.5, self.field_width / 6 + 1.25)
                    self._draw_right_triangle(
                        yard, 2.5, 5 * self.field_width / 6 - 0.25
                    )

    def _draw_left_triangle(self, x: float, x_offset: float, y_offset: float) -> None:
        """Draw a small left-pointing triangle marker.

        Args:
            x (float): X coordinate (yard line) to anchor the triangle.
            x_offset (float): Horizontal offset from ``x`` to place the
                triangle.
            y_offset (float): Vertical position for the triangle's
                centre point.
        """
        triangle = patches.Polygon(
            [
                (x + x_offset, y_offset),
                (x + x_offset + 1, y_offset + 0.25),
                (x + x_offset + 1, y_offset - 0.25),
            ],
            color=self.line_color,
            alpha=0.5,
        )
        self.ax.add_patch(triangle)

    def _draw_right_triangle(self, x: float, x_offset: float, y_offset: float) -> None:
        """Draw a small right-pointing triangle marker.

        Args:
            x (float): X coordinate (yard line) to anchor the triangle.
            x_offset (float): Horizontal offset from ``x`` to place the
                triangle.
            y_offset (float): Vertical position for the triangle's
                centre point.
        """
        triangle = patches.Polygon(
            [
                (x + x_offset, y_offset),
                (x + x_offset - 1, y_offset + 0.25),
                (x + x_offset - 1, y_offset - 0.25),
            ],
            color=self.line_color,
            alpha=0.5,
        )
        self.ax.add_patch(triangle)
