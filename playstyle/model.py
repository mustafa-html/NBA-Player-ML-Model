"""K-Means clustering of playstyle vectors, archetype naming, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from playstyle import config
from playstyle.features import SeasonStats, describe


@dataclass
class PlaystyleModel:
    kmeans: KMeans
    pca: PCA
    season_stats: SeasonStats
    archetypes: List[Dict]          # per cluster: name, traits, match
    feature_keys: List[str]
    k_scores: Dict[int, Dict[str, float]]
    train_config: Dict

    @property
    def k(self) -> int:
        return self.kmeans.n_clusters

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(np.atleast_2d(Z))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PlaystyleModel":
        return joblib.load(path)


def score_k(Z: np.ndarray, k_min: int, k_max: int, random_state: int = 42) -> Dict[int, Dict[str, float]]:
    """Inertia (elbow) and silhouette for each k, for choosing and reporting k."""
    scores = {}
    sample = min(len(Z), 3000)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=sample, random_state=random_state)
        scores[k] = {"inertia": float(km.inertia_), "silhouette": float(sil)}
    return scores


def name_clusters(centroids: np.ndarray) -> List[Dict]:
    """Give each cluster a distinct archetype name by matching its centroid to
    hand-defined prototype directions (Hungarian assignment on cosine similarity)."""
    names = list(config.PROTOTYPES)
    P = np.zeros((len(names), len(config.FEATURE_KEYS)))
    for i, name in enumerate(names):
        for key, w in config.PROTOTYPES[name].items():
            P[i, config.FEATURE_KEYS.index(key)] = w
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    C = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
    sim = C @ P.T  # clusters x prototypes

    rows, cols = linear_sum_assignment(-sim)
    assigned = dict(zip(rows, cols))
    out = []
    fallback_count = 0
    for c in range(len(centroids)):
        p = assigned.get(c)
        match = float(sim[c, p]) if p is not None else 0.0
        if p is None or match < config.MIN_PROTOTYPE_MATCH:
            fallback_count += 1
            name = config.FALLBACK_ARCHETYPE + ("" if fallback_count == 1 else f" {fallback_count}")
        else:
            name = names[p]
        out.append({"cluster": c, "name": name, "match": round(match, 3), "traits": describe(centroids[c], n=3, threshold=0.4)})
    return out


def fit_model(Z: np.ndarray, season_stats: SeasonStats, cfg: config.TrainConfig) -> PlaystyleModel:
    k_scores = score_k(Z, *cfg.k_range, random_state=cfg.random_state)  # for the report and --k auto
    k = cfg.k or max(k_scores, key=lambda kk: k_scores[kk]["silhouette"])
    kmeans = KMeans(n_clusters=k, n_init=25, random_state=cfg.random_state).fit(Z)
    pca = PCA(n_components=2, random_state=cfg.random_state).fit(Z)
    return PlaystyleModel(
        kmeans=kmeans,
        pca=pca,
        season_stats=season_stats,
        archetypes=name_clusters(kmeans.cluster_centers_),
        feature_keys=list(config.FEATURE_KEYS),
        k_scores=k_scores,
        train_config={
            "seasons": [cfg.first_season, cfg.last_season],
            "min_minutes": cfg.min_minutes,
            "min_games": cfg.min_games,
            "k": k,
        },
    )
