"""Teams module."""

from dataclasses import dataclass
from pathlib import Path

from .settings import get_settings


@dataclass
class Team:
    """Team data class."""

    abbreviation: str
    location: str
    name: str
    primary_color: str
    secondary_color: str

    @property
    def full_name(self) -> str:
        """Get full name of team (location + name).

        Returns:
            str: Team name.
        """
        return f"{self.location} {self.name}"

    def get_logo_file_path(self) -> Path:
        """Get path to team logo file.

        Returns:
            Path: Path to team logo file.
        """
        settings = get_settings()

        team_logo_path: Path = Path(
            settings.LOGO_FILE_TEMPLATE.format(team=self.abbreviation)
        )

        return settings.LOGOS_DIR / team_logo_path


TEAMS = {
    "ARI": Team("ARI", "Arizona", "Cardinals", "#97233F", "#FFB612"),
    "ATL": Team("ATL", "Atlanta", "Falcons", "#A71930", "#000000"),
    "BAL": Team("BAL", "Baltimore", "Ravens", "#241773", "#9E7C0C"),
    "BUF": Team("BUF", "Buffalo", "Bills", "#00338D", "#C60C30"),
    "CAR": Team("CAR", "Carolina", "Panthers", "#0085CA", "#101820"),
    "CHI": Team("CHI", "Chicago", "Bears", "#0B162A", "#C83803"),
    "CIN": Team("CIN", "Cincinnati", "Bengals", "#FB4F14", "#000000"),
    "CLE": Team("CLE", "Cleveland", "Browns", "#311D00", "#FF3C00"),
    "DAL": Team("DAL", "Dallas", "Cowboys", "#003594", "#869397"),
    "DEN": Team("DEN", "Denver", "Broncos", "#FB4F14", "#002244"),
    "DET": Team("DET", "Detroit", "Lions", "#0076B6", "#B0B7BC"),
    "GB": Team("GB", "Green Bay", "Packers", "#203731", "#FFB612"),
    "HOU": Team("HOU", "Houston", "Texans", "#03202F", "#A71930"),
    "IND": Team("IND", "Indianapolis", "Colts", "#002C5F", "#B0B7BC"),
    "JAX": Team("JAX", "Jacksonville", "Jaguars", "#006778", "#D7A22A"),
    "KC": Team("KC", "Kansas City", "Chiefs", "#E31837", "#FFB81C"),
    "LA": Team("LA", "Los Angeles", "Rams", "#003594", "#FFD100"),
    "LAC": Team("LAC", "Los Angeles", "Chargers", "#0080C6", "#FFC20E"),
    "LV": Team("LV", "Las Vegas", "Raiders", "#000000", "#A5ACAF"),
    "MIA": Team("MIA", "Miami", "Dolphins", "#008E97", "#FC4C02"),
    "MIN": Team("MIN", "Minnesota", "Vikings", "#4F2683", "#FFC62F"),
    "NE": Team("NE", "New England", "Patriots", "#002244", "#C60C30"),
    "NO": Team("NO", "New Orleans", "Saints", "#D3BC8D", "#101820"),
    "NYG": Team("NYG", "New York", "Giants", "#0B2265", "#A71930"),
    "NYJ": Team("NYJ", "New York", "Jets", "#125740", "#000000"),
    "PHI": Team("PHI", "Philadelphia", "Eagles", "#004C54", "#A5ACAF"),
    "PIT": Team("PIT", "Pittsburgh", "Steelers", "#FFB612", "#101820"),
    "SEA": Team("SEA", "Seattle", "Seahawks", "#002244", "#69BE28"),
    "SF": Team("SF", "San Francisco", "49ers", "#AA0000", "#B3995D"),
    "TB": Team("TB", "Tampa Bay", "Buccaneers", "#D50A0A", "#0A0A08"),
    "TEN": Team("TEN", "Tennessee", "Titans", "#0C2340", "#4B92DB"),
    "WAS": Team("WAS", "Washington", "Commanders", "#773141", "#FFB612"),
}
