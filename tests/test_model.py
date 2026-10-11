import numpy as np

from playstyle import config
from playstyle.model import name_clusters


def _prototype(name):
    v = np.zeros(len(config.FEATURE_KEYS))
    for k, w in config.PROTOTYPES[name].items():
        v[config.FEATURE_KEYS.index(k)] = w
    return v


def test_centroids_matching_prototypes_get_those_names():
    names = ["Rim-Running Big", "Spot-Up Shooter", "Primary Creator"]
    out = name_clusters(np.vstack([_prototype(n) * 1.5 for n in names]))
    assert [a["name"] for a in out] == names


def test_names_are_unique_and_weak_matches_fall_back():
    centroids = np.vstack([_prototype("Stretch Big"), _prototype("Stretch Big") * 0.9, np.zeros(len(config.FEATURE_KEYS))])
    out = name_clusters(centroids)
    assert len({a["name"] for a in out}) == 3
    assert out[2]["name"].startswith(config.FALLBACK_ARCHETYPE)


def test_trained_model_is_sensible(trained):
    s = trained["summary"]
    assert s["k"] == 10
    assert len({a["name"] for a in s["archetypes"]}) == 10
    assert sum(a["size"] for a in s["archetypes"]) == s["player_seasons"]
    for f in ("k_selection.png", "playstyle_map.png", "archetype_heatmap.png", "archetype_trend.png"):
        assert (trained["reports"] / f).stat().st_size > 10_000
