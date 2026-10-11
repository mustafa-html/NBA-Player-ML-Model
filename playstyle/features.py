"""Turn raw rates into standardized playstyle vectors.

Features are standardized *within each season*. The league's three-point rate
climbed steeply over the last decade, so raw rates would make almost every 2016
player look like a non-shooter next to a 2026 player. Per-season z-scores
describe a player relative to the league he actually played in.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from playstyle import config

SeasonStats = Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]  # season -> (median, mean, std)


def raw_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[config.FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # Undefined rates (e.g. "% of threes assisted" for a centre with no made
    # threes) get the season median, so they don't read as extreme either way.
    X = X.groupby(df["season"].values).transform(lambda c: c.fillna(c.median()))
    return X.fillna(X.median())


def fit_season_stats(X: pd.DataFrame, seasons: pd.Series) -> SeasonStats:
    stats: SeasonStats = {}
    for season, block in X.groupby(seasons.values):
        std = block.std(ddof=0).replace(0, 1.0)
        stats[int(season)] = (block.median().to_numpy(), block.mean().to_numpy(), std.to_numpy())
    return stats


def standardize(X: pd.DataFrame, seasons: pd.Series, stats: SeasonStats, clip: float = 4.0) -> np.ndarray:
    Z = np.empty(X.shape, dtype=float)
    for i, (season, row) in enumerate(zip(seasons.values, X.to_numpy(dtype=float))):
        median, mean, std = stats[int(season)]
        row = np.where(np.isnan(row), median, row)
        Z[i] = (row - mean) / std
    return np.clip(Z, -clip, clip)


def vector_from_raw(raw: Dict[str, float], season: int, stats: SeasonStats, clip: float = 4.0) -> np.ndarray:
    """Standardize one player's raw stats (inference path). Keys may be feature
    keys ("usg") or source column names ("usg_percent"); missing values get the
    season median."""
    if season not in stats:
        raise KeyError(f"No reference data for season {season}. Available: {min(stats)}-{max(stats)}")
    median, mean, std = stats[season]
    values = []
    for f, med in zip(config.FEATURES, median):
        v = raw.get(f.key, raw.get(f.column))
        values.append(float(v) if v is not None else med)
    return np.clip((np.array(values) - mean) / std, -clip, clip)


def describe(z: np.ndarray, n: int = 3, threshold: float = 0.75) -> list[str]:
    """The n most distinctive traits of a standardized vector, in plain English."""
    order = np.argsort(-np.abs(z))
    traits = []
    for i in order[:n]:
        if abs(z[i]) < threshold:
            break
        f = config.FEATURES[i]
        traits.append(f.high if z[i] > 0 else f.low)
    return traits or ["close to league average across the board"]
