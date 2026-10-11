"""Settings and the playstyle feature definitions. Paths can be overridden with
environment variables so the same code runs locally and in a container."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def _path(env: str, default: Path) -> Path:
    return Path(os.getenv(env, str(default)))


RAW_DIR = _path("RAW_DIR", ROOT / "data" / "raw")
ARTIFACT_DIR = _path("ARTIFACT_DIR", ROOT / "artifacts")
REPORT_DIR = _path("REPORT_DIR", ROOT / "reports")

SOURCE_URL = "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/Data"
SOURCE_FILES = {"advanced": "Advanced.csv", "shooting": "Player Shooting.csv", "per_game": "Player Per Game.csv"}


@dataclass(frozen=True)
class Feature:
    column: str       # column name in the raw data
    key: str          # short name used in code and prototypes
    label: str        # chart label
    high: str         # plain-English trait when well above league average
    low: str          # ... and when well below


# Style, not quality: every feature is a share or a rate, so a bench player and
# an All-Star who play the same way end up close together.
FEATURES: List[Feature] = [
    Feature("percent_fga_from_x0_3_range", "rim", "Shots at rim", "lives at the rim", "rarely shoots at the rim"),
    Feature("percent_fga_from_x3_10_range", "short", "Short paint shots", "works the short paint", "avoids the short paint"),
    Feature("percent_fga_from_x10_16_range", "mid", "Mid-range shots", "takes mid-range jumpers", "avoids the mid-range"),
    Feature("percent_fga_from_x16_3p_range", "long2", "Long twos", "takes long twos", "avoids long twos"),
    Feature("percent_fga_from_x3p_range", "three", "Three-point share", "shoots a lot of threes", "rarely shoots threes"),
    Feature("percent_corner_3s_of_3pa", "corner3", "Corner-three share", "spots up in the corners", "shoots threes above the break"),
    Feature("percent_assisted_x2p_fg", "ast2", "Assisted twos", "gets set up for twos", "creates his own twos"),
    Feature("percent_assisted_x3p_fg", "ast3", "Assisted threes", "catch-and-shoot threes", "creates his own threes"),
    Feature("percent_dunks_of_fga", "dunk", "Dunk share", "dunks a lot", "rarely dunks"),
    Feature("f_tr", "ftr", "Free-throw rate", "draws a lot of fouls", "rarely gets to the line"),
    Feature("usg_percent", "usg", "Usage", "high-usage scorer", "low-usage role player"),
    Feature("ast_percent", "astp", "Assist rate", "primary playmaker", "rarely assists"),
    Feature("tov_percent", "tov", "Turnover rate", "high turnover rate", "protects the ball"),
    Feature("orb_percent", "orb", "Offensive rebounds", "crashes the offensive glass", "stays off the offensive glass"),
    Feature("drb_percent", "drb", "Defensive rebounds", "cleans the defensive glass", "light defensive rebounder"),
    Feature("stl_percent", "stl", "Steal rate", "gets a lot of steals", "few steals"),
    Feature("blk_percent", "blk", "Block rate", "protects the rim", "rarely blocks shots"),
]
FEATURE_COLUMNS = [f.column for f in FEATURES]
FEATURE_KEYS = [f.key for f in FEATURES]
KEY_TO_COLUMN = {f.key: f.column for f in FEATURES}

# Archetype prototypes: the direction (in standardized feature space) each
# named archetype points in. Cluster centroids are matched to these with the
# Hungarian algorithm so every cluster gets a distinct, readable name.
PROTOTYPES: Dict[str, Dict[str, float]] = {
    "Primary Creator": {"usg": 1.0, "astp": 1.0, "ast2": -1.0, "ast3": -0.7, "ftr": 0.5, "tov": 0.4, "orb": -0.5},
    "Floor General": {"astp": 1.2, "tov": 0.6, "usg": 0.1, "ast2": -0.4, "orb": -0.6, "blk": -0.6, "dunk": -0.5, "rim": -0.3},
    "Spot-Up Shooter": {"three": 1.2, "corner3": 0.8, "ast3": 0.8, "usg": -0.6, "rim": -0.6, "ftr": -0.5, "astp": -0.3, "orb": -0.4},
    "3-and-D Wing": {"three": 0.8, "stl": 0.9, "ast3": 0.6, "corner3": 0.5, "usg": -0.5, "astp": -0.2},
    "Rim-Running Big": {"rim": 1.0, "dunk": 1.0, "orb": 0.8, "blk": 0.7, "ast2": 0.8, "three": -1.0, "usg": -0.3, "astp": -0.4},
    "Stretch Big": {"three": 0.7, "drb": 0.9, "blk": 0.6, "orb": 0.2, "astp": -0.3, "rim": -0.2, "stl": -0.3, "corner3": -0.2},
    "Post Scorer": {"short": 1.0, "usg": 0.7, "orb": 0.6, "drb": 0.4, "ast2": -0.6, "three": -0.9},
    "Mid-Range Scorer": {"mid": 1.0, "long2": 1.0, "usg": 0.7, "ast2": -0.7, "three": -0.5, "rim": -0.3},
    "Slasher": {"rim": 0.9, "ftr": 0.9, "usg": 0.5, "dunk": 0.4, "three": -0.4, "ast2": -0.2},
    "Glass Cleaner": {"orb": 1.0, "drb": 1.0, "blk": 0.5, "usg": -0.4, "three": -0.7, "astp": -0.3},
    "Perimeter Pest": {"stl": 1.0, "astp": 0.4, "three": 0.3, "usg": -0.2, "blk": -0.2},
    "Off-Ball Wing": {"long2": 0.8, "mid": 0.5, "ast2": 0.6, "ast3": 0.3, "rim": -0.6, "orb": -0.6, "astp": -0.3, "ftr": -0.3},
    "Shot-Creating Scorer": {"usg": 1.0, "ast2": -0.7, "ast3": -0.6, "three": 0.3, "mid": 0.4, "astp": 0.2},
}
FALLBACK_ARCHETYPE = "Versatile Role Player"
MIN_PROTOTYPE_MATCH = 0.30


@dataclass
class TrainConfig:
    first_season: int = int(os.getenv("FIRST_SEASON", 2016))
    last_season: int = int(os.getenv("LAST_SEASON", 2026))
    min_minutes: int = int(os.getenv("MIN_MINUTES", 500))
    min_games: int = int(os.getenv("MIN_GAMES", 20))
    k: int | None = int(os.getenv("N_CLUSTERS")) if os.getenv("N_CLUSTERS") else 10
    k_range: Tuple[int, int] = (4, 14)
    random_state: int = 42
    z_clip: float = 4.0
    seasons: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.seasons = list(range(self.first_season, self.last_season + 1))
