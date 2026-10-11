"""Nearest-neighbour search over standardized playstyle vectors."""

from __future__ import annotations

import difflib
import unicodedata
from typing import List, Optional

import numpy as np
import pandas as pd


def normalize_name(name: str) -> str:
    """'Luka Dončić' -> 'luka doncic' so searches work without accents."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(ascii_name.lower().replace(".", "").replace("'", "").split())


class SimilarityIndex:
    """Cosine similarity: two players match when their strengths and weaknesses
    point the same way, whatever their overall level of extremeness."""

    def __init__(self, table: pd.DataFrame, Z: np.ndarray):
        self.table = table.reset_index(drop=True)
        self.Z = Z
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        self.unit = Z / np.maximum(norms, 1e-9)
        self.table["_name_key"] = self.table["player"].map(normalize_name)

    def find(self, player: str, season: Optional[int] = None) -> Optional[int]:
        """Row index for a player (latest qualifying season if none given)."""
        key = normalize_name(player)
        rows = self.table[self.table["_name_key"] == key]
        if season is not None:
            rows = rows[rows["season"] == season]
        if rows.empty:
            return None
        return int(rows.sort_values("season").index[-1])

    def suggest(self, player: str, n: int = 5) -> List[str]:
        names = self.table.drop_duplicates("_name_key").set_index("_name_key")["player"]
        keys = difflib.get_close_matches(normalize_name(player), names.index.tolist(), n=n, cutoff=0.6)
        return [names[k] for k in keys]

    def query(
        self,
        vector: np.ndarray,
        n: int = 10,
        season: Optional[int] = None,
        exclude_player_id: Optional[str] = None,
    ) -> pd.DataFrame:
        v = vector / max(np.linalg.norm(vector), 1e-9)
        sims = self.unit @ v
        mask = np.ones(len(sims), dtype=bool)
        if season is not None:
            mask &= (self.table["season"] == season).to_numpy()
        if exclude_player_id is not None:
            mask &= (self.table["player_id"] != exclude_player_id).to_numpy()
        idx = np.where(mask)[0]
        top = idx[np.argsort(-sims[idx])[:n]]
        out = self.table.loc[top].drop(columns="_name_key").copy()
        out["similarity"] = np.round(sims[top] * 100, 1)
        return out
