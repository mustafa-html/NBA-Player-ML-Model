import numpy as np
import pandas as pd

from playstyle import config
from playstyle.features import describe, fit_season_stats, raw_feature_matrix, standardize, vector_from_raw


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for season, shift in ((2016, 0.0), (2026, 0.2)):
        for _ in range(60):
            rows.append({"season": season, **{c: rng.normal(0.3 + shift, 0.1) for c in config.FEATURE_COLUMNS}})
    df = pd.DataFrame(rows)
    df.loc[0, "percent_assisted_x3p_fg"] = np.nan  # e.g. a centre with no made threes
    return df


def test_missing_rates_filled_with_season_median():
    df = _frame()
    X = raw_feature_matrix(df)
    assert not X.isna().any().any()
    col = "percent_assisted_x3p_fg"
    assert X.loc[0, col] == df[df.season == 2016][col].median()


def test_standardized_within_each_season():
    df = _frame()
    X = raw_feature_matrix(df)
    Z = standardize(X, df["season"], fit_season_stats(X, df["season"]), clip=10)
    for season in (2016, 2026):
        block = Z[(df["season"] == season).to_numpy()]
        assert np.allclose(block.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(block.std(axis=0), 1, atol=1e-9)


def test_inference_vector_uses_season_reference_and_accepts_both_key_styles():
    df = _frame()
    X = raw_feature_matrix(df)
    stats = fit_season_stats(X, df["season"])
    _, mean, _ = stats[2026]
    z = vector_from_raw({"usg": mean[config.FEATURE_KEYS.index("usg")], "ast_percent": 10.0}, 2026, stats)
    assert abs(z[config.FEATURE_KEYS.index("usg")]) < 1e-9
    assert z[config.FEATURE_KEYS.index("astp")] > 3


def test_describe_reads_extremes_in_plain_english():
    z = np.zeros(len(config.FEATURES))
    z[config.FEATURE_KEYS.index("three")] = 2.5
    z[config.FEATURE_KEYS.index("rim")] = -1.8
    assert describe(z, n=2) == ["shoots a lot of threes", "rarely shoots at the rim"]
    assert describe(np.zeros(len(config.FEATURES))) == ["close to league average across the board"]
