"""Matplotlib reports. Uses the Agg backend so it runs on headless servers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from playstyle import config  # noqa: E402

PALETTE = plt.get_cmap("tab20").colors
LABELS = [f.label for f in config.FEATURES]


def _save(fig, path: Optional[Path] = None) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def k_selection(k_scores: Dict[int, Dict[str, float]], chosen: int, path: Optional[Path] = None) -> bytes:
    ks = sorted(k_scores)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ks, [k_scores[k]["inertia"] for k in ks], "o-", color="#1d4ed8", label="Inertia")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster spread)", color="#1d4ed8")
    ax2 = ax1.twinx()
    ax2.plot(ks, [k_scores[k]["silhouette"] for k in ks], "s--", color="#c2410c", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="#c2410c")
    ax1.axvline(chosen, color="#555", lw=1, ls=":")
    ax1.set_title(f"Choosing k (chosen: {chosen})")
    return _save(fig, path)


def archetype_map(table: pd.DataFrame, archetypes: List[Dict], highlight: Optional[List[int]] = None,
                  season: Optional[int] = None, path: Optional[Path] = None) -> bytes:
    """2-D PCA projection of every player-season, coloured by archetype."""
    data = table if season is None else table[table["season"] == season]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for a in archetypes:
        pts = data[data["cluster"] == a["cluster"]]
        ax.scatter(pts["pc1"], pts["pc2"], s=10, alpha=0.55, color=PALETTE[a["cluster"] % 20], label=a["name"])
    if highlight:
        for i, idx in enumerate(highlight):
            r = table.loc[idx]
            ax.scatter(r["pc1"], r["pc2"], s=140 if i == 0 else 70, facecolors="none",
                       edgecolors="black", linewidths=2 if i == 0 else 1.2, zorder=5)
            ax.annotate(f"{r['player']} ({r['season']})", (r["pc1"], r["pc2"]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8 if i else 9, weight="bold" if i == 0 else "normal")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Playstyle map" + (f", {season - 1}-{str(season)[2:]}" if season else ""))
    ax.legend(fontsize=7, markerscale=2, loc="best", frameon=False)
    return _save(fig, path)


def centroid_heatmap(centroids: np.ndarray, archetypes: List[Dict], path: Optional[Path] = None) -> bytes:
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(archetypes) + 1.8))
    im = ax.imshow(centroids, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(archetypes)), [a["name"] for a in archetypes], fontsize=8)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="Std. devs from league average")
    ax.set_title("What defines each archetype")
    return _save(fig, path)


def archetype_trend(table: pd.DataFrame, archetypes: List[Dict], path: Optional[Path] = None) -> bytes:
    """Share of qualifying players in each archetype, by season."""
    names = {a["cluster"]: a["name"] for a in archetypes}
    share = pd.crosstab(table["season"], table["cluster"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(share.index, share.T.values, labels=[names[c] for c in share.columns],
                 colors=[PALETTE[c % 20] for c in share.columns], alpha=0.85)
    ax.set_xlim(share.index.min(), share.index.max())
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of qualifying players")
    ax.set_xlabel("Season (ending year)")
    ax.set_title("How the league's playstyles have shifted")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    return _save(fig, path)


def radar(z_a: np.ndarray, label_a: str, z_b: Optional[np.ndarray] = None, label_b: str = "",
          path: Optional[Path] = None) -> bytes:
    """Compare one or two playstyle vectors (z-scores clipped to +/-3 for display)."""
    n = len(LABELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"polar": True})
    for z, label, color in ((z_a, label_a, "#1d4ed8"), (z_b, label_b, "#c2410c")):
        if z is None:
            continue
        vals = np.clip(z, -3, 3).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, lw=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1], LABELS, fontsize=7)
    ax.set_ylim(-3, 3)
    ax.set_yticks([-2, 0, 2], ["-2", "avg", "+2"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), fontsize=8, frameon=False)
    return _save(fig, path)
