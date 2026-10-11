import pandas as pd

from playstyle import config
from playstyle.data import build_player_seasons, collapse_traded, load_raw


def _rows(rows):
    return pd.DataFrame(rows, columns=["season", "lg", "player", "player_id", "team", "g", "mp"])


def test_traded_player_keeps_combined_row_and_lists_teams():
    df = _rows([
        [2025, "NBA", "Luka", "l1", "2TM", 50, 1769],
        [2025, "NBA", "Luka", "l1", "DAL", 22, 785],
        [2025, "NBA", "Luka", "l1", "LAL", 28, 984],
        [2017, "NBA", "Old", "o1", "TOT", 70, 2000],
        [2017, "NBA", "Old", "o1", "SAC", 40, 1100],
        [2017, "NBA", "Old", "o1", "NOP", 30, 900],
        [2025, "NBA", "Stay", "s1", "BOS", 80, 2500],
        [2025, "BAA", "Other", "x1", "BOS", 80, 2500],
    ])
    out = collapse_traded(df).set_index("player_id")
    assert len(out) == 3                                    # non-NBA row dropped
    assert out.loc["l1", "team"] == "2TM" and out.loc["l1", "mp"] == 1769
    assert out.loc["l1", "teams"] == "DAL/LAL"
    assert out.loc["o1", "teams"] == "SAC/NOP"
    assert out.loc["s1", "teams"] == "BOS"


def test_bundled_data_builds_one_row_per_player_season():
    cfg = config.TrainConfig()
    df = build_player_seasons(load_raw(), cfg)
    assert not df.duplicated(["season", "player_id"]).any()
    assert df["mp"].min() >= cfg.min_minutes and df["g"].min() >= cfg.min_games
    assert set(df["season"]) == set(cfg.seasons)
    assert all(c in df.columns for c in config.FEATURE_COLUMNS)
