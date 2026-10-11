"""Load Basketball Reference season tables and turn them into one clean row per
player-season."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from playstyle import config

KEYS = ["season", "player_id"]
MULTI_TEAM = re.compile(r"^(TOT|\dTM)$")  # Basketball Reference combined rows: "TOT" (old) or "2TM", "3TM" (2025+)
DISPLAY_COLUMNS = ["mp_per_game", "pts_per_game", "trb_per_game", "ast_per_game", "x3pa_per_game", "fg_percent"]


def fetch_raw(raw_dir: Path = config.RAW_DIR, seasons: Iterable[int] | None = None) -> Dict[str, Path]:
    """Download the source CSVs and keep only the seasons we train on."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(seasons) if seasons else None
    out = {}
    for name, filename in config.SOURCE_FILES.items():
        url = f"{config.SOURCE_URL}/{urllib.request.quote(filename)}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            df = pd.read_csv(resp)
        if wanted:
            df = df[df["season"].isin(wanted)]
        path = raw_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        out[name] = path
    return out


def load_raw(raw_dir: Path = config.RAW_DIR) -> Dict[str, pd.DataFrame]:
    missing = [n for n in config.SOURCE_FILES if not (raw_dir / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data in {raw_dir}: {', '.join(missing)}. Run `python -m playstyle.cli fetch` first."
        )
    return {name: pd.read_csv(raw_dir / f"{name}.csv") for name in config.SOURCE_FILES}


def collapse_traded(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season.

    A traded player has a row per team plus a combined season row. Keep the
    combined row (it covers the whole season) and record the teams he played for.
    """
    df = df[df["lg"] == "NBA"].copy()
    df["team"] = df["team"].astype(str)
    combined = df["team"].str.match(MULTI_TEAM)

    teams = (
        df[~combined].groupby(KEYS)["team"].apply(lambda s: "/".join(dict.fromkeys(s))).rename("teams")
    )
    has_combined = df.loc[combined, KEYS].drop_duplicates().assign(_has=True)
    df = df.merge(has_combined, on=KEYS, how="left")
    keep = combined.values | df["_has"].isna().values
    out = df[keep].drop(columns="_has").drop_duplicates(subset=KEYS, keep="first")
    return out.merge(teams, on=KEYS, how="left").reset_index(drop=True)


def build_player_seasons(raw: Dict[str, pd.DataFrame], cfg: config.TrainConfig) -> pd.DataFrame:
    """Merge advanced, shooting and per-game tables for qualifying player-seasons."""
    adv = collapse_traded(raw["advanced"])
    adv = adv[adv["season"].isin(cfg.seasons)]
    adv = adv[(adv["mp"] >= cfg.min_minutes) & (adv["g"] >= cfg.min_games)]

    adv_cols = KEYS + ["player", "age", "pos", "teams", "g", "mp"] + [
        c for c in config.FEATURE_COLUMNS if c in raw["advanced"].columns
    ]
    shoot = collapse_traded(raw["shooting"])
    shoot_cols = KEYS + [c for c in config.FEATURE_COLUMNS if c in raw["shooting"].columns]
    pg = collapse_traded(raw["per_game"])[KEYS + DISPLAY_COLUMNS]

    df = adv[adv_cols].merge(shoot[shoot_cols], on=KEYS, how="inner").merge(pg, on=KEYS, how="left")
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Source data is missing feature columns: {missing}")
    df["season"] = df["season"].astype(int)
    return df.sort_values(["season", "player"]).reset_index(drop=True)
