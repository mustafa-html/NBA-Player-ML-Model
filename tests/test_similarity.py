import numpy as np
import pandas as pd

from playstyle.similarity import SimilarityIndex, normalize_name


def test_normalize_name_strips_accents_and_punctuation():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("  De'Aaron  Fox ") == "deaaron fox"
    assert normalize_name("P.J. Washington") == "pj washington"


def _index():
    table = pd.DataFrame({
        "player": ["A", "A", "B", "C"], "player_id": ["a", "a", "b", "c"], "season": [2025, 2026, 2026, 2026],
    })
    Z = np.array([[1.0, 0.0], [1.0, 0.1], [2.0, 0.2], [-1.0, 0.0]])
    return SimilarityIndex(table, Z)


def test_cosine_ignores_magnitude_and_excludes_self():
    idx = _index()
    res = idx.query(np.array([1.0, 0.1]), n=3, exclude_player_id="a")
    assert res.iloc[0]["player"] == "B" and res.iloc[0]["similarity"] == 100.0
    assert res.iloc[-1]["player"] == "C" and res.iloc[-1]["similarity"] < 0


def test_find_defaults_to_latest_season_and_filters():
    idx = _index()
    assert idx.find("a") == 1
    assert idx.find("A", 2025) == 0
    assert idx.find("Nobody") is None


def test_real_data_matches(engine):
    i = engine.index.find("Rudy Gobert", 2026)
    top = engine.similar(i, n=5)
    assert engine.table.loc[i, "archetype"] == "Rim-Running Big"
    assert (top["archetype"] == "Rim-Running Big").sum() >= 4
    assert "Rudy Gobert" not in top["player"].tolist()
    assert engine.index.suggest("Lebron")[0] == "LeBron James"
