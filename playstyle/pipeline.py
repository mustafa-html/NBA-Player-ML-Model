"""Training and inference entry points shared by the CLI and the web app."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from playstyle import config, reports
from playstyle.data import build_player_seasons, load_raw
from playstyle.features import describe, fit_season_stats, raw_feature_matrix, standardize, vector_from_raw
from playstyle.model import PlaystyleModel, fit_model
from playstyle.similarity import SimilarityIndex

log = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
TABLE_FILE = "player_seasons.csv"
Z_FILE = "playstyle_vectors.npy"
SUMMARY_FILE = "summary.json"


def train(
    cfg: Optional[config.TrainConfig] = None,
    raw_dir: Path = config.RAW_DIR,
    artifact_dir: Path = config.ARTIFACT_DIR,
    report_dir: Path = config.REPORT_DIR,
) -> Dict:
    cfg = cfg or config.TrainConfig()
    table = build_player_seasons(load_raw(raw_dir), cfg)
    X = raw_feature_matrix(table)
    stats = fit_season_stats(X, table["season"])
    Z = standardize(X, table["season"], stats, clip=cfg.z_clip)

    model = fit_model(Z, stats, cfg)
    table["cluster"] = model.kmeans.labels_
    table["archetype"] = [model.archetypes[c]["name"] for c in table["cluster"]]
    pcs = model.pca.transform(Z)
    table["pc1"], table["pc2"] = pcs[:, 0], pcs[:, 1]

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifact_dir / MODEL_FILE)
    table.to_csv(artifact_dir / TABLE_FILE, index=False)
    np.save(artifact_dir / Z_FILE, Z)

    reports.k_selection(model.k_scores, model.k, report_dir / "k_selection.png")
    reports.archetype_map(table, model.archetypes, season=cfg.last_season, path=report_dir / "playstyle_map.png")
    reports.centroid_heatmap(model.kmeans.cluster_centers_, model.archetypes, report_dir / "archetype_heatmap.png")
    reports.archetype_trend(table, model.archetypes, report_dir / "archetype_trend.png")

    summary = {
        "player_seasons": int(len(table)),
        "players": int(table["player_id"].nunique()),
        "seasons": cfg.seasons,
        "k": model.k,
        "silhouette": round(model.k_scores[model.k]["silhouette"], 3),
        "explained_variance_2d": round(float(model.pca.explained_variance_ratio_.sum()), 3),
        "archetypes": [
            {**a, "size": int((table["cluster"] == a["cluster"]).sum())} for a in model.archetypes
        ],
    }
    (artifact_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary


@dataclass
class Engine:
    """Loaded artifacts plus the queries the app needs."""

    model: PlaystyleModel
    table: pd.DataFrame
    Z: np.ndarray
    index: SimilarityIndex

    @classmethod
    def load(cls, artifact_dir: Path = config.ARTIFACT_DIR) -> "Engine":
        model = PlaystyleModel.load(artifact_dir / MODEL_FILE)
        table = pd.read_csv(artifact_dir / TABLE_FILE, dtype={"player_id": str})
        Z = np.load(artifact_dir / Z_FILE)
        return cls(model=model, table=table, Z=Z, index=SimilarityIndex(table, Z))

    @property
    def seasons(self) -> list[int]:
        return sorted(self.table["season"].unique().tolist())

    def player_profile(self, idx: int) -> Dict:
        row = self.table.loc[idx]
        arch = self.model.archetypes[int(row["cluster"])]
        return {
            "player": row["player"],
            "player_id": row["player_id"],
            "season": int(row["season"]),
            "age": int(row["age"]) if pd.notna(row["age"]) else None,
            "position": row["pos"],
            "teams": row.get("teams"),
            "games": int(row["g"]),
            "per_game": {
                "minutes": _num(row.get("mp_per_game")), "points": _num(row.get("pts_per_game")),
                "rebounds": _num(row.get("trb_per_game")), "assists": _num(row.get("ast_per_game")),
            },
            "archetype": arch["name"],
            "cluster": int(row["cluster"]),
            "traits": describe(self.Z[idx]),
            "playstyle": dict(zip(self.model.feature_keys, np.round(self.Z[idx], 2).tolist())),
        }

    def similar(self, idx: int, n: int = 10, scope: str = "season") -> pd.DataFrame:
        row = self.table.loc[idx]
        return self.index.query(
            self.Z[idx], n=n, season=int(row["season"]) if scope == "season" else None,
            exclude_player_id=row["player_id"],
        )

    def infer(self, raw_stats: Dict[str, float], season: int, n: int = 10) -> Dict:
        """Place a stat line that isn't in the dataset: standardize it against its
        season, assign an archetype and return the closest real players."""
        z = vector_from_raw(raw_stats, season, self.model.season_stats)
        cluster = int(self.model.predict(z)[0])
        matches = self.index.query(z, n=n, season=season)
        return {
            "season": season,
            "archetype": self.model.archetypes[cluster]["name"],
            "cluster": cluster,
            "traits": describe(z),
            "playstyle": dict(zip(self.model.feature_keys, np.round(z, 2).tolist())),
            "similar": records(matches),
        }


def _num(v) -> Optional[float]:
    return None if v is None or pd.isna(v) else round(float(v), 1)


def records(df: pd.DataFrame) -> list[Dict]:
    cols = ["player", "season", "teams", "pos", "archetype", "similarity", "pts_per_game", "ast_per_game", "trb_per_game"]
    out = df[[c for c in cols if c in df.columns]].copy()
    return json.loads(out.to_json(orient="records"))


def ensure_artifacts(artifact_dir: Path = config.ARTIFACT_DIR, report_dir: Path = config.REPORT_DIR) -> None:
    """Train on first run so the app works straight after `pip install`."""
    if not (artifact_dir / MODEL_FILE).exists() or not (report_dir / "playstyle_map.png").exists():
        log.info("No trained model in %s; training now", artifact_dir)
        train(artifact_dir=artifact_dir, report_dir=report_dir)
